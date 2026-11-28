"""Minimal OpenAI-compatible chat server for inference-parity evals.

    python utils/serve_openai.py --model <HF export> --engine {hf,ours} --dtype {bfloat16,float32} --port 8001

Serves `/health` and `/v1/chat/completions` (greedy only) so the unchanged eval client
that talks to `vllm serve` can be pointed at another inference engine:

- `hf`:   transformers `Qwen3_5ForConditionalGeneration.generate` (sdpa).
- `ours`: this repo's training model (`models/qwen3_5`), greedy decoding without a KV
          cache: the prompt is embedded once (vision tower included), then every step
          reruns the decoder over prompt + generated tokens.

Preprocessing mirrors `vllm serve` as used by the eval: the model dir's chat template
renders the request messages as given, images come from the data URLs, and the HF
processor resizes them with `--mm-processor-kwargs` (the eval's min/max pixels).
Generation stops at <|im_end|> or <|endoftext|> and decodes with special tokens skipped.
Requests are served one at a time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
PAD_TO = 512


def load_image(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data: URLs are supported")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


class Engine:
    def __init__(self, args):
        from transformers import AutoProcessor

        self.args = args
        self.dtype = getattr(torch, args.dtype)
        self.dev = torch.device("cuda")
        self.proc = AutoProcessor.from_pretrained(args.model)
        tok = self.proc.tokenizer
        self.stop_ids = {tok.convert_tokens_to_ids(t) for t in STOP_TOKENS}
        self.mm_kwargs = json.loads(args.mm_processor_kwargs) if args.mm_processor_kwargs else {}
        if args.engine == "hf":
            from transformers import Qwen3_5ForConditionalGeneration

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                args.model, dtype=self.dtype, attn_implementation="sdpa").to(self.dev).eval()
        else:
            from models.qwen3_5.checkpoint import build_meta, load_hf, materialize

            model = build_meta(args.model, seq_len=args.max_model_len)
            materialize(model, self.dev)
            load_hf(model, args.model)
            for p in model.parameters():
                p.data = p.data.to(self.dtype)
            model.retie_weights()
            self.model = model.eval()
            cfg = json.loads((Path(args.model) / "config.json").read_text())
            self.image_id, self.video_id = cfg["image_token_id"], cfg["video_token_id"]
            self.merge = cfg["vision_config"]["spatial_merge_size"]

    def encode(self, body):
        messages = body["messages"]
        images = [load_image(p["image_url"]["url"]) for m in messages
                  if isinstance(m["content"], list) for p in m["content"]
                  if p.get("type") == "image_url"]
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(body.get("chat_template_kwargs") or {}))
        return self.proc(text=[text], images=images or None, return_tensors="pt",
                         **self.mm_kwargs).to(self.dev)

    @torch.no_grad()
    def generate(self, inputs, max_new: int) -> tuple[list[int], str]:
        if self.args.engine == "hf":
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new,
                                      eos_token_id=sorted(self.stop_ids), pad_token_id=None)
            new = out[0, inputs["input_ids"].shape[1]:].tolist()
        else:
            new = self._ours_greedy(inputs, max_new)
        finish = "length"
        for i, t in enumerate(new):
            if t in self.stop_ids:
                new, finish = new[: i + 1], "stop"
                break
        return new, finish

    @contextmanager
    def _embeddings_in(self):
        # Qwen35Model.forward embeds tokens only when tok_embeddings is set; with it
        # unset, `tokens` is taken as the input embeddings (the pipeline-stage path)
        emb = self.model.tok_embeddings
        self.model.tok_embeddings = None
        try:
            yield emb
        finally:
            self.model.tok_embeddings = emb

    def _ours_greedy(self, inputs, max_new: int) -> list[int]:
        from data.model_batch import mrope_positions

        m = self.model
        ids = inputs["input_ids"][0]
        grid = inputs.get("image_grid_thw")
        pv = inputs.get("pixel_values")
        x = m._prepare_multimodal_embeds(
            ids, pixel_values=pv.to(self.dtype) if pv is not None else None,
            grid_thw=grid,
            special_tokens={"image_id": self.image_id})
        new: list[int] = []
        pad_id = self.proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        with self._embeddings_in() as emb:
            for _ in range(max_new):
                all_ids = torch.cat([ids, ids.new_tensor(new)]) if new else ids
                n = all_ids.shape[0]
                # Right-pad to a multiple of PAD_TO: the model is causal, so the
                # padding cannot change the logits at position n-1, and the kernels
                # see a few lengths instead of a new one (and a new autotune) per step.
                pad = -n % PAD_TO
                ids_p = torch.cat([all_ids, all_ids.new_full((pad,), pad_id)])
                x_p = torch.cat([x, emb(ids_p[n:])]) if pad else x
                mrope = mrope_positions(
                    ids_p.cpu(), torch.tensor([0, n + pad]),
                    grid.cpu() if grid is not None else None,
                    image_token_id=self.image_id, video_token_id=self.video_id,
                    spatial_merge_size=self.merge).to(self.dev)
                pos = torch.arange(n + pad, device=self.dev)
                inp, _, extra = m.preprocess_inputs(
                    {"input": x_p, "labels": ids_p, "positions": pos, "mrope_positions": mrope})
                logits = m(inp, **extra)
                nxt = int(logits[n - 1].float().argmax())
                new.append(nxt)
                if nxt in self.stop_ids:
                    break
                x = torch.cat([x, emb(ids.new_tensor([nxt]))])
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["hf", "ours"], required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--served-model-name", default="model")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--mm-processor-kwargs", default="")
    args = ap.parse_args()
    if args.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    engine = Engine(args)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._send(404, {})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if float(body.get("temperature", 0.0)) != 0.0 or float(body.get("repetition_penalty", 1.0)) != 1.0:
                    raise ValueError("only greedy decoding without repetition penalty is supported")
                with lock:
                    inputs = engine.encode(body)
                    new, finish = engine.generate(inputs, int(body.get("max_tokens", 256)))
                text = engine.proc.tokenizer.decode(new, skip_special_tokens=True)
                self._send(200, {
                    "id": f"cmpl-{time.time_ns()}", "object": "chat.completion",
                    "model": args.served_model_name,
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                              "completion_tokens": len(new)},
                })
            except Exception as exc:  # the client records failures and retries on rerun
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    print(f"serving {args.engine}/{args.dtype} {args.model} on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
