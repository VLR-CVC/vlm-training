import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

class LocalDispatchMetadata:
    def __init__(self, token_indices_experts_sorted, top_scores_experts_sorted):
        self.token_indices_experts_sorted = token_indices_experts_sorted
        self.top_scores_experts_sorted = top_scores_experts_sorted

class AllToAllDispatchMetadata:
    def __init__(self, token_indices_experts_sorted, top_scores_experts_sorted, input_shape, permuted_indices, input_splits, output_splits):
        self.token_indices_experts_sorted = token_indices_experts_sorted
        self.top_scores_experts_sorted = top_scores_experts_sorted
        self.input_shape = input_shape
        self.permuted_indices = permuted_indices
        self.input_splits = input_splits
        self.output_splits = output_splits

class _AllToAllSingleAutograd(torch.autograd.Function):
    """
    wrapper around all-to-all to add backward pass
    """
    @staticmethod
    def forward(ctx, input_, output_splits, input_splits, group):
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        
        out_total = int(sum(output_splits)) if output_splits else 0
        out = torch.empty((out_total, input_.size(1)), dtype=input_.dtype, device=input_.device)
        
        if group is not None:
            dist.all_to_all_single(out, input_, output_splits, input_splits, group=group.get_group())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        in_total = int(sum(ctx.input_splits)) if ctx.input_splits else 0
        grad_input = torch.empty((in_total, grad_output.size(1)), dtype=grad_output.dtype, device=grad_output.device)
        
        if ctx.group is not None:
            dist.all_to_all_single(grad_input, grad_output, ctx.input_splits, ctx.output_splits, group=ctx.group.get_group())
        return grad_input, None, None, None

def all_to_all_single_autograd(input_, output_splits, input_splits, group):
    return _AllToAllSingleAutograd.apply(input_, output_splits, input_splits, group)

class TokenDispatcher:
    """Consolidated EP/SP dispatcher. Handles local token reorder and all-to-all."""
    
    def __init__(self, num_experts: int, top_k: int, score_before_experts: bool = True):
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_before_experts = score_before_experts
        
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int = -1

    def _split_along_sp(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for t in tensors:
            local_num_tokens = t.shape[0] // self.sp_size
            offset = self.sp_rank * local_num_tokens
            results.append(t[offset : offset + local_num_tokens])
        return results

    def _permute(self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts):
        device = num_tokens_per_expert_group.device
        total = num_tokens_per_expert_group.sum().item()

        t_mat = num_tokens_per_expert_group.view(ep_size, num_local_experts)
        input_starts = (num_tokens_per_expert_group.cumsum(0) - num_tokens_per_expert_group).view(ep_size, num_local_experts)

        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(segment_lens.long())
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (input_starts[seg_ids] + torch.arange(total, device=device) - output_starts[seg_ids]).long()

        num_tokens_per_expert = t_mat.sum(0)
        return routed_input.shape, routed_input[permuted_indices, :], permuted_indices, num_tokens_per_expert

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted

    def dispatch(self, x: torch.Tensor, top_scores: torch.Tensor, selected_experts_indices: torch.Tensor):
        if self.sp_size > 1:
            x, top_scores, selected_experts_indices = self._split_along_sp(x, top_scores, selected_experts_indices)

        flat_experts = selected_experts_indices.view(-1)
        num_tokens_per_expert = torch.bincount(flat_experts, minlength=self.num_experts).float()

        token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k
        routed_input = x[token_indices_experts_sorted]

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        # Skip all-to-all logic entirely if ep_mesh is missing (EP=1)
        if self.ep_mesh is None:
            metadata = AllToAllDispatchMetadata(
                token_indices_experts_sorted, top_scores_experts_sorted, 
                None, None, None, None
            )
            return routed_input, num_tokens_per_expert, metadata

        ep_size = self.ep_mesh.size()
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=self.ep_mesh.get_group())

        input_splits = num_tokens_per_expert.view(ep_size, -1).sum(dim=1).int().cpu().tolist()
        output_splits = num_tokens_per_expert_group.view(ep_size, -1).sum(dim=1).int().cpu().tolist()

        routed_input = all_to_all_single_autograd(routed_input, output_splits, input_splits, self.ep_mesh)

        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
        input_shape, routed_input, permuted_indices, num_tokens_per_expert_group = self._permute(
            routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted, top_scores_experts_sorted, 
            input_shape, permuted_indices, input_splits, output_splits
        )
        return routed_input, num_tokens_per_expert_group, metadata

    def combine(self, routed_output: torch.Tensor, metadata: "AllToAllDispatchMetadata", x: torch.Tensor, shared_experts: torch.nn.Module | None = None) -> torch.Tensor:
        if self.ep_mesh is not None:
            routed_output = self._unpermute(routed_output, metadata.input_shape, metadata.permuted_indices)
            routed_output = all_to_all_single_autograd(routed_output, metadata.input_splits, metadata.output_splits, self.ep_mesh)

        out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (routed_output.to(torch.float32) * metadata.top_scores_experts_sorted.reshape(-1, 1)).to(routed_output.dtype)

        token_indices_experts_sorted = metadata.token_indices_experts_sorted
        if self.sp_size > 1:
            local_num_tokens = x.shape[0] // self.sp_size
            token_indices_experts_sorted = token_indices_experts_sorted + local_num_tokens * self.sp_rank

        out.index_add_(0, token_indices_experts_sorted, routed_output)
        return out


class MockLocalExperts(nn.Module):
    """A dummy expert implementation holding only local weights."""
    def __init__(self, num_local_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.w1 = nn.Parameter(torch.randn(num_local_experts, hidden_dim, intermediate_dim))
        self.w2 = nn.Parameter(torch.randn(num_local_experts, intermediate_dim, hidden_dim))

    def forward(self, x, num_tokens_per_expert):
        """
        x: (total_routed_tokens, hidden_dim)
        num_tokens_per_expert: (num_local_experts,)
        """
        total_tokens = int(num_tokens_per_expert.sum().item())
        if total_tokens == 0:
            return torch.empty_like(x)
        if total_tokens == x.shape[0] and self.num_local_experts == 1:
            return torch.relu(x @ self.w1[0]) @ self.w2[0]

        offsets = torch.zeros(self.num_local_experts + 1, dtype=torch.long, device=x.device)
        offsets[1:] = num_tokens_per_expert.cumsum(0)
        
        outputs = []
        for i in range(self.num_local_experts):
            start, end = int(offsets[i]), int(offsets[i+1])
            if end > start:
                chunk = x[start:end]
                hidden = torch.relu(chunk @ self.w1[i])
                outputs.append(hidden @ self.w2[i])
        
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)

    forward_compiled = None

class MockLocalExperts(nn.Module):
    def __init__(self, num_local_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.w1 = nn.Parameter(torch.randn(num_local_experts, hidden_dim, intermediate_dim))
        self.w2 = nn.Parameter(torch.randn(num_local_experts, intermediate_dim, hidden_dim))
    
    def forward(self, x, num_tokens_per_expert):
        total_tokens = int(num_tokens_per_expert.sum().item())
        if total_tokens == 0:
            return torch.empty_like(x)
        if total_tokens == x.shape[0] and self.num_local_experts == 1:
            return torch.relu(F.linear(x, self.w1[0].t())) @ self.w2[0].t()

        offsets = torch.zeros(self.num_local_experts + 1, dtype=torch.long, device=x.device)
        offsets[1:] = num_tokens_per_expert.cumsum(0)
        
        outputs = []
        for i in range(self.num_local_experts):
            start, end = int(offsets[i]), int(offsets[i+1])
            if end > start:
                chunk = x[start:end]
                hidden = torch.relu(F.linear(chunk, self.w1[i].t()))
                outputs.append(F.linear(hidden, self.w2[i].t()))
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)
        return compiled_forward
    
    def forward(self, x, num_tokens_per_expert):
        total_tokens = int(num_tokens_per_expert.sum().item())
        if total_tokens == 0:
            return torch.empty_like(x)
        if total_tokens == x.shape[0] and self.num_local_experts == 1:
            return torch.relu(F.linear(x, self.w1[0].t()) @ self.w2[0].t())

        offsets = torch.zeros(self.num_local_experts + 1, dtype=torch.long, device=x.device)
        offsets[1:] = num_tokens_per_expert.cumsum(0)
        
        outputs = []
        for i in range(self.num_local_experts):
            start, end = int(offsets[i]), int(offsets[i+1])
            if end > start:
                chunk = x[start:end]
                hidden = torch.relu(F.linear(chunk, self.w1[i].t()))
                outputs.append(F.linear(hidden, self.w2[i].t()))
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)

def run_ep_benchmark(ep_size: int, num_experts: int = 16, hidden_dim: int = 4096, batch_size: int = 8, seq_len: int = 2048):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    
    dp_size = world_size // ep_size
    global_mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_mesh = global_mesh["ep"]
    
    num_local_experts = num_experts // ep_size
    intermediate_dim = 14336 
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    experts = MockLocalExperts(num_local_experts, hidden_dim, intermediate_dim).to(device)
    shared_expert = nn.Linear(hidden_dim, hidden_dim).to(device) 
    dispatcher = TokenDispatcher(num_experts=num_experts, top_k=2, score_before_experts=True)
    
    if ep_size > 1:
        dispatcher.ep_mesh = ep_mesh
    
    num_tokens = batch_size * seq_len
    x = torch.randn(num_tokens, hidden_dim, device=device, requires_grad=True)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    top_scores, selected_experts_indices = torch.topk(torch.softmax(router_logits, dim=-1), k=2, dim=-1)

    # WARMUP
    for _ in range(3):
        r_in, counts, meta = dispatcher.dispatch(x, top_scores, selected_experts_indices)
        e_out = experts(r_in, counts)
        out = dispatcher.combine(e_out, meta, x, shared_experts=shared_expert)
        out.sum().backward()

    torch.cuda.synchronize(device)
    
    # BENCHMARK
    iters = 10
    dispatch_times = []
    expert_times = []
    shared_fwd_times = []
    index_add_times = []
    backward_times = []
    
    for _ in range(iters):
        torch.cuda.synchronize(device)
        
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        
        t0.record()
        r_in, counts, meta = dispatcher.dispatch(x, top_scores, selected_experts_indices)
        t1.record()
        e_out = experts(r_in, counts)
        t2.record()
        
        out = shared_expert(x)
        t3.record()
        
        routed = e_out
        if dispatcher.ep_mesh is not None:
            routed = dispatcher._unpermute(routed, meta.input_shape, meta.permuted_indices)
            routed = all_to_all_single_autograd(routed, meta.input_splits, meta.output_splits, dispatcher.ep_mesh)
        
        token_indices_experts_sorted = meta.token_indices_experts_sorted
        if dispatcher.sp_size > 1:
            local_num_tokens = x.shape[0] // dispatcher.sp_size
            token_indices_experts_sorted = token_indices_experts_sorted + local_num_tokens * dispatcher.sp_rank
        
        out.index_add_(0, token_indices_experts_sorted, routed)
        t4.record()
        
        out.sum().backward()
        t5.record()
        
        torch.cuda.synchronize(device)
        dispatch_times.append(t0.elapsed_time(t1))
        expert_times.append(t1.elapsed_time(t2))
        shared_fwd_times.append(t2.elapsed_time(t3))
        index_add_times.append(t3.elapsed_time(t4))
        backward_times.append(t4.elapsed_time(t5))
    
    avg_dispatch = sum(dispatch_times) / iters
    avg_expert = sum(expert_times) / iters
    avg_shared_fwd = sum(shared_fwd_times) / iters
    avg_index_add = sum(index_add_times) / iters
    avg_backward = sum(backward_times) / iters
    avg_time_ms = avg_dispatch + avg_expert + avg_backward
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    # FLOPs Math: FWD is 2*H*I per weight matrix, BWD is ~2x FWD. Total = 12 * H * I per token.
    # Total routed tokens globally = num_tokens * dp_size * top_k
    global_routed_tokens = num_tokens * dp_size * dispatcher.top_k
    expert_flops_per_iter = 12 * global_routed_tokens * hidden_dim * intermediate_dim
    
    # Include shared expert FLOPs
    shared_flops_per_iter = 12 * (num_tokens * dp_size) * hidden_dim * hidden_dim
    total_flops = expert_flops_per_iter + shared_flops_per_iter
    
    tflops_per_sec = (total_flops / (avg_time_ms / 1000.0)) / (1e12)
    tflops_per_gpu = tflops_per_sec / world_size

    dist.barrier()
    
    if rank == 0:
        print(f"--- Configuration: DP={dp_size}, EP={ep_size} ---")
        print(f"Peak Memory Allocated: {peak_mem_gb:.2f} GB")
        print(f"Dispatch: {avg_dispatch:.2f}ms, Expert: {avg_expert:.2f}ms, SharedFwd: {avg_shared_fwd:.2f}ms, IndexAdd: {avg_index_add:.2f}ms, Backward: {avg_backward:.2f}ms")
        print(f"Average Time / Iteration: {avg_time_ms:.2f} ms")
        print(f"Total TFLOPS (Cluster): {tflops_per_sec:.2f} TFLOPS")
        print(f"Per-GPU TFLOPS: {tflops_per_gpu:.2f} TFLOPS\n")
        
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    assert world_size == 4, "This test is designed to run exactly on 4 devices."
    
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    if dist.get_rank() == 0:
        print("Starting EP Memory Benchmark...\n")

    # Baseline: EP=1 (All 16 experts on every GPU)
    run_ep_benchmark(ep_size=1)

    # Run EP=2
    run_ep_benchmark(ep_size=2)
    
    # Run EP=4
    run_ep_benchmark(ep_size=4)
    
    dist.destroy_process_group()
