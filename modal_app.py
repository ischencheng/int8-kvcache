"""Run one shell command from this repo on a Modal GPU, on top of vLLM 0.32.0.

    modal run modal_app.py --cmd "python scripts/check_env.py" --run-id env --gpu A100-80GB

The repo is mounted at /root/repo (read-only copy, so local edits need no image
rebuild). The image applies patches/vllm-0.32.0-int8-static-kv.patch to the
installed vLLM and registers the `kvq` package as a vLLM general plugin, so every
vLLM process (engine core, TP workers, API server) loads our backend.

Whatever the command writes to $OUT (= /tmp/out) comes back to results/<run-id>/
together with output.log, also when the command fails or times out.
"""
import io
import os
import subprocess
import tarfile
import time
from pathlib import Path

import modal

REPO = Path(__file__).parent
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"
PATCH = "patches/vllm-0.32.0-int8-static-kv.patch"

image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")  # Modal looks for `python`
    .apt_install("patch")
    .run_commands("python3 -m pip install pytest matplotlib pandas datasets safetensors human-eval")
    .add_local_file(REPO / "patches/install_plugin_stub.sh", "/tmp/stub.sh", copy=True)
    .run_commands("bash /tmp/stub.sh")
    .add_local_file(REPO / PATCH, "/tmp/kvq.patch", copy=True)
    .run_commands(
        "cd $(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
        " && if [ -s /tmp/kvq.patch ]; then patch -p1 --forward < /tmp/kvq.patch; fi"
    )
    .apt_install("cuda-nsight-compute-13-0")  # ncu for the kernel analysis (run with --clock-control none)
    .env({"HF_HOME": "/cache/hf", "OUT": "/tmp/out", "PYTHONUNBUFFERED": "1",
          "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO, "/root/repo", ignore=["results", ".git", "**/__pycache__", "*.pyc"])
)

app = modal.App("int8-kvcache", image=image)
cache = modal.Volume.from_name("int8-kvcache-cache", create_if_missing=True)


def _pack(out_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out_dir, arcname=".")
    return buf.getvalue()


@app.function(gpu="A100-80GB", volumes={"/cache": cache}, timeout=3 * 3600,
              cpu=8.0, memory=64 * 1024)
def run(cmd: str, limit_s: int) -> bytes:
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "output.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd="/root/repo", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            if time.time() - t0 > limit_s:
                proc.kill()
                log.write(f"\n[modal_app] killed after {limit_s}s\n")
                break
        proc.wait(timeout=60)
    finally:
        log.write(f"\n[modal_app] exit={proc.returncode} wall={time.time() - t0:.0f}s\n")
        log.close()
        cache.commit()
    return _pack(out)


@app.local_entrypoint()
def main(cmd: str, run_id: str, gpu: str = "A100-80GB", limit_s: int = 3 * 3600 - 300):
    fn = run.with_options(gpu=None if gpu == "none" else gpu)
    data = fn.remote(cmd, limit_s)
    dest = REPO / "results" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")
    print(f"[modal_app] results in {dest}")
