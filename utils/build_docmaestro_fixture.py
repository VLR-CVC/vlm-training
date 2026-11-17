"""Build the tiny DocMaestro fixture that `models/tests/test_cooker_docmaestro.py` reads.

Runs DocMaestro's own render pass and composers on a couple of processed papers, so
the fixture is exactly what a real DocMaestro run ships: shards with asset
*references*, a page store, and the extraction tree's figure crops.

Needs a DocMaestro checkout and an environment with its compose/render deps
(pypdfium2, PyYAML, Pillow; NOT this repo's env), e.g.:

    PY=/data/151-1/users/allabres/envs/docvqa2027-base/bin/python
    $PY utils/build_docmaestro_fixture.py \\
        --docmaestro ~/DocMaestro \\
        --processed-root /data/151-1/datasets/DocMaestroData/processed \\
        --pdf-root /data/151-1/datasets/DocMaestroData/data \\
        --out /data/151-1/users/allabres/t11_docmaestro_fixture

Output layout (point the tests at it with DOCMAESTRO_FIXTURE=<out>):

    <out>/processed/<doc_id>/...   copied XML + MinerU crops (the crops root)
    <out>/pages/<doc_id>/pNNN.jpg  DocMaestro page store
    <out>/shards/<split>/<task>-00000.tar
    <out>/forms/<doc_name>.fields.json   synthetic CommonForms sidecar

`form_filling` reads a CommonForms sidecar, not the XML, and `run_doctasks.py` has no
flag for it, so it is composed here through the API from a synthetic sidecar (two
fields on page 1 of the first document). Its labels are fake; its asset contract
(one page render) is the real one, which is all the cooker tests look at.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DOCS = (
    # 11 pages, 11 <page-number>, 8 figures: covers page_sorting redaction and the
    # three figure tasks
    "CVPR/2020/15_Keypoints_Is_All_You_Need",
    "CVPR/2020/3D_Packing_for_Self-Supervised_Monocular_Depth_Estimation",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docmaestro", type=Path, required=True, help="DocMaestro checkout")
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--pdf-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", nargs="*", default=list(DEFAULT_DOCS), help="doc_ids (VENUE/YEAR/stem)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dm = a.docmaestro.expanduser().resolve()
    out = a.out.resolve()
    processed, pdfs, pages, shards, forms = (out / d for d in ("processed", "pdfs", "pages", "shards", "forms"))
    for d in (processed, pdfs, pages, shards, forms):
        d.mkdir(parents=True, exist_ok=True)

    for doc_id in a.docs:
        src = a.processed_root / doc_id
        if not (processed / doc_id).exists():
            shutil.copytree(src, processed / doc_id)
        pdf = pdfs / f"{doc_id}.pdf"  # append, never with_suffix: stems contain dots
        pdf.parent.mkdir(parents=True, exist_ok=True)
        if not pdf.exists():
            shutil.copyfile(a.pdf_root / f"{doc_id}.pdf", pdf)

    env_py = sys.executable
    run = lambda *args: subprocess.run([env_py, *map(str, args)], check=True, cwd=dm,  # noqa: E731
                                       env={"PYTHONPATH": str(dm / "src"), "PATH": "/usr/bin:/bin"})
    run("scripts/render_pages.py", "--processed-root", processed, "--pdf-root", pdfs,
        "--pages-root", pages)
    run("scripts/run_doctasks.py", "--output-dir", processed, "--pages-root", pages,
        "--shard-dir", shards, "--no-json", "--seed", a.seed)

    # form_filling: synthetic sidecar, composed through the API (see module docstring)
    doc_id = a.docs[0]
    doc_name = Path(doc_id).name
    (forms / f"{doc_name}.fields.json").write_text(json.dumps({"pages": [{
        "page": 1, "width": 1000, "height": 1300,
        "fields": [{"category": "text_input", "bbox": [100, 100, 400, 140]},
                   {"category": "checkbox", "bbox": [100, 200, 130, 230]}],
    }]}))
    code = f"""
import xml.etree.ElementTree as ET
from pathlib import Path
from doc_maestro.task_composer.tasks.form_filling import FormFillingComposer
from doc_maestro.task_composer.shard_writer import ShardWriter
from doc_maestro.training.splits import assign_split
xml = sorted(Path({str(processed / doc_id)!r}).glob("*.xml"))[0]
root = ET.parse(xml).getroot()
samples = FormFillingComposer(sidecar_root=Path({str(forms)!r})).compose_document(root, xml.parent, doc_id={doc_id!r})
assert samples, "synthetic sidecar produced no form_filling sample"
split = assign_split({doc_id!r}, seed={a.seed})
ShardWriter(Path({str(shards)!r}) / split, "form_filling").write(samples)
print("form_filling:", len(samples), "sample(s) ->", split)
"""
    subprocess.run([env_py, "-c", code], check=True, env={"PYTHONPATH": str(dm / "src"), "PATH": "/usr/bin:/bin"})

    for split_dir in sorted(p for p in shards.iterdir() if p.is_dir()):
        print(split_dir.name, sorted(p.name for p in split_dir.glob("*.tar")))


if __name__ == "__main__":
    main()
