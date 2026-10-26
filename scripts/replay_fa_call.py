"""Replay a dumped FlashAttention call of the long-query path (KVQ_DEBUG_DUMP) with random tensors of
the same shapes, one variant per subprocess (an illegal access poisons the CUDA context).

    python3 scripts/replay_fa_call.py fa_call.pt
"""
import subprocess
import sys

import torch

VARIANTS = ["as_is", "dumped_strides", "no_empty_rows", "wider_table", "only_long_rows"]


def stats(d):
    cu, su, bt = d["cu_seqlens_q"], d["seqused_k"], d["block_table"]
    ql = cu[1:] - cu[:-1]
    long = ql > 8
    print({k: v for k, v in d.items() if not torch.is_tensor(v)})
    print(f"requests {ql.numel()}, tokens {int(ql.sum())}, long {int(long.sum())}, "
          f"short {int(((ql > 0) & ~long).sum())}, empty {int((ql == 0).sum())}")
    print("long q_len", ql[long].tolist()[:16], "long seqused", su[long].tolist()[:16])
    print("block_table", tuple(bt.shape), "max id", int(bt.max()), "temp pages", d["kt_shape"][0])
    print("seqused < q_len on long rows:", int((su[long] < ql[long]).sum()))


def run(d, variant):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    dev = "cuda"
    cu, su, bt = d["cu_seqlens_q"].to(dev), d["seqused_k"].to(dev).clone(), d["block_table"].to(dev).clone()
    T, H, D = d["q_shape"]
    max_q, max_k = d["max_seqlen_q"], d["max_seqlen_k"]
    if variant == "dumped_strides":
        qs, os_ = d["q_stride"], d["out_stride"]
        q = torch.randn(T * qs[0], dtype=torch.bfloat16, device=dev).as_strided((T, H, D), qs)
        out = torch.empty(T * os_[0], dtype=torch.bfloat16, device=dev).as_strided((T, H, D), os_)
    else:
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(q)
    kt = torch.randn(d["kt_shape"], dtype=torch.bfloat16, device=dev)
    vt = torch.randn_like(kt)
    ql = cu[1:] - cu[:-1]
    if variant == "no_empty_rows":
        su = torch.where(su == 0, torch.ones_like(su), su)
    if variant == "wider_table":
        bt = torch.cat([bt, torch.zeros(bt.shape[0], 64, dtype=bt.dtype, device=dev)], 1)
    if variant == "only_long_rows":
        keep = torch.nonzero(ql > 8).flatten()
        starts = cu[:-1][keep]
        idx = torch.cat([torch.arange(int(s), int(s) + int(n), device=dev) for s, n in zip(starts, ql[keep])])
        q = q[idx].contiguous()
        out = torch.empty_like(q)
        cu = torch.zeros(keep.numel() + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.cumsum(ql[keep], 0)
        su, bt = su[keep].contiguous(), bt[keep].contiguous()
    flash_attn_varlen_func(q=q, k=kt, v=vt, out=out, cu_seqlens_q=cu, max_seqlen_q=max_q, seqused_k=su,
                           max_seqlen_k=max_k, softmax_scale=D ** -0.5, causal=True, block_table=bt, fa_version=2)
    torch.cuda.synchronize()
    print(f"{variant}: ok")


if __name__ == "__main__":
    d = torch.load(sys.argv[1])
    if len(sys.argv) == 3:
        run(d, sys.argv[2])
    else:
        stats(d)
        for v in VARIANTS:
            r = subprocess.run([sys.executable, __file__, sys.argv[1], v], capture_output=True, text=True)
            err = [line for line in (r.stdout + r.stderr).splitlines() if "Error" in line or "ok" in line]
            print(f"{v}: exit {r.returncode}", err[-1:] if err else "")
