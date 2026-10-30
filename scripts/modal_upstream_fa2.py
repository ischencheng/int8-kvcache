"""The FA2 empty-row repro on upstream flash-attn (Dao-AILab release wheel) instead of vLLM's fork.

    modal run scripts/modal_upstream_fa2.py --run-id 22-fa2-lse-oob-upstream
    modal run scripts/modal_upstream_fa2.py --run-id 22-fa2-lse-oob-upstream-min --cmd "bash scripts/fa2_lse_oob_min.sh"

Results land in results/<run-id>/ like modal_app.py.
"""
import io
import os
import subprocess
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent
WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/"
         "flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")  # devel: compute-sanitizer
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .pip_install("torch==2.8.0", "numpy", WHEEL)
    .env({"OUT": "/tmp/out", "PYTHONUNBUFFERED": "1", "FA_IMPL": "upstream"})
    .add_local_dir(REPO / "scripts", "/root/repo/scripts", ignore=["**/__pycache__", "_prev_kernel"])
)
app = modal.App("int8-kvcache-upstream-fa2", image=image)


@app.function(gpu="A100-80GB", timeout=3600)
def run(cmd: str) -> bytes:
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["bash", "-c", cmd], cwd="/root/repo", capture_output=True, text=True)
    (out / "output.log").write_text(r.stdout + r.stderr + f"\n[modal_upstream_fa2] exit={r.returncode}\n")
    print(r.stdout[-4000:])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main(run_id: str, cmd: str = "bash scripts/repro_fa2_empty_rows.sh"):
    dest = REPO / "results" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(run.remote(cmd))) as tar:
        tar.extractall(dest, filter="data")
    print(f"[modal_upstream_fa2] results in {dest}")
