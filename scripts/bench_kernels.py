"""Stage 7/8 microbenchmarks: decode paged attention and the KV write path.

Decode attention, Qwen2.5-7B shapes (28 q heads, 4 kv heads, D 128, block 16), sweeping
batch x context. Same split-KV policy for every Triton kernel (vLLM's: 16 segments when
batch <= 128 / kv_heads). Implementations:
  fa2_bf16          vLLM's FlashAttention-2 kernel (what A100 serving uses today)
  vllm_triton_bf16  vLLM's unified Triton attention, bf16 cache
  vllm_int8_pth     vLLM's unified Triton attention, int8_per_token_head cache (dequantize on load)
  ours_bf16         our kernel, bf16 cache (same code as the int8 modes: the fair baseline)
  ours_int8_deq     our kernel, int8 cache, dequantize each tile (implementation A)
  ours_int8         our kernel, int8 cache, INT8 tensor-core QK/PV (implementation B)
  ours_int8_g32     ... with one q scale per 32 channels
  cuda_int8         our CUDA kernel (mma.sync m16n8k32 s8), implementation B
  fa3_fp8           (H100 only) FlashAttention-3 with an FP8 cache
Effective bandwidth = KV bytes the kernel must read / time.
"""
import argparse
import json
import os

import torch
import triton
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

from kvq import cuda_kernels as ck
from kvq import quant
from kvq.triton_kernels.int8_attention import paged_attention
from kvq.triton_kernels.quant_cache import quant_reshape_and_cache
from kvq.triton_kernels.quant_q import quant_q

HQ, HKV, D, BS = 28, 4, 128, 16
SCALE = D ** -0.5
THRESH_3D = 128 // HKV


def decode_case(batch, ctx, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = (ctx + BS - 1) // BS
    nb = batch * pages + 1
    bt = (torch.randperm(nb - 1, generator=g, device="cuda")[: batch * pages] + 1).to(torch.int32).view(batch, pages)
    q = torch.randn(batch, HQ, D, generator=g, device="cuda").to(torch.bfloat16)
    k = torch.randn(nb, BS, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
    v = torch.randn(nb, BS, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    sl = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    return q, k, v, cu, sl, bt


def static_for(k, v):
    kf = k.flatten(0, 1).float()
    mu = kf.mean(0)
    return {"k_scale": quant.scale_from_absmax((kf - mu).abs().amax(0)).contiguous(),
            "v_scale": quant.scale_from_absmax(v.flatten(0, 1).float().abs().amax(0)).contiguous(),
            "k_mu": mu.contiguous()}


def segm(batch):
    if batch > THRESH_3D:
        return None
    return (torch.empty(THRESH_3D, HQ, 16, D, device="cuda"), torch.empty(THRESH_3D, HQ, 16, device="cuda"),
            torch.empty(THRESH_3D, HQ, 16, device="cuda"))


def bench(fn):
    fn()
    return triton.testing.do_bench(fn, warmup=20, rep=80) * 1e3  # us


def decode_row(batch, ctx, fp8):
    q, k, v, cu, sl, bt = decode_case(batch, ctx)
    st = static_for(k, v)
    kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
    vq = quant.quantize(v, st["v_scale"])
    out = torch.empty_like(q)
    sg = segm(batch)
    common = dict(cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=sl, max_seqlen_k=ctx, softmax_scale=SCALE,
                  causal=True, block_table=bt)
    seg_kw = dict(seq_threshold_3D=THRESH_3D, num_par_softmax_segments=16,
                  softmax_segm_output=torch.empty(THRESH_3D, HQ, 16, D, device="cuda"),
                  softmax_segm_max=torch.empty(THRESH_3D, HQ, 16, device="cuda"),
                  softmax_segm_expsum=torch.empty(THRESH_3D, HQ, 16, device="cuda"))
    nb = k.shape[0]
    ksc = torch.rand(nb, BS, HKV, device="cuda") * 0.05
    vsc = torch.rand(nb, BS, HKV, device="cuda") * 0.05
    fns = {
        "fa2_bf16": lambda: flash_attn_varlen_func(q=q, k=k, v=v, out=out, fa_version=2, **common),
        "vllm_triton_bf16": lambda: unified_attention(q, k, v, out, cu, 1, sl, ctx, SCALE, True, (-1, -1), bt,
                                                      0, None, None, None, **seg_kw),
        "vllm_int8_pth": lambda: unified_attention(q, kq, vq, out, cu, 1, sl, ctx, SCALE, True, (-1, -1), bt,
                                                   0, None, None, None, **seg_kw,
                                                   kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                                                   k_scale_cache=ksc, v_scale_cache=vsc),
        "ours_bf16": lambda: paged_attention(q, k, v, out, cu, sl, bt, 1, SCALE, segm=sg),
        "ours_int8_deq": lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, int8_math=False, segm=sg),
        "ours_int8": lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, segm=sg),
        "ours_int8_g32": lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, q_groups=4, segm=sg),
        "cuda_int8": lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=sg),
    }
    if fp8:
        kf8, vf8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        ones = torch.ones(batch, HKV, device="cuda")
        fns["fa3_fp8"] = lambda: flash_attn_varlen_func(q=q.to(torch.float8_e4m3fn), k=kf8, v=vf8, out=out,
                                                        fa_version=3, q_descale=ones, k_descale=ones,
                                                        v_descale=ones, **common)
    row = {"batch": batch, "ctx": ctx}
    for name, fn in fns.items():
        try:
            row[name] = bench(fn)
        except Exception as e:  # noqa: BLE001 - report and keep going
            row[name] = None
            row[f"{name}_error"] = repr(e)[:200]
    kv_elems = batch * ctx * HKV * D * 2
    row["kv_bytes_bf16"] = kv_elems * 2
    row["kv_bytes_int8"] = kv_elems
    return row


def write_path_rows():
    """KV write per step: our fused quant+cache vs a two-pass torch version vs vLLM's kernels."""
    rows = []
    for T in (64, 256, 2048, 8192):
        g = torch.Generator(device="cuda").manual_seed(0)
        nb = T // BS + 64
        key = torch.randn(T, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
        val = torch.randn(T, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
        slots = torch.randperm(nb * BS, generator=g, device="cuda")[:T].long()
        kc = torch.zeros(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
        vc = torch.zeros_like(kc)
        kci, vci = torch.zeros(nb, BS, HKV, D, dtype=torch.int8, device="cuda"), torch.zeros(nb, BS, HKV, D,
                                                                                              dtype=torch.int8,
                                                                                              device="cuda")
        st = static_for(key[:, None], val[:, None])
        ksc, vsc = torch.ones(nb, BS, HKV, device="cuda"), torch.ones(nb, BS, HKV, device="cuda")
        one = torch.ones((), device="cuda")

        def torch_two_pass():
            kq = quant.quantize(key.float() - st["k_mu"], st["k_scale"])
            vq = quant.quantize(val, st["v_scale"])
            kci[slots // BS, slots % BS] = kq
            vci[slots // BS, slots % BS] = vq

        rows.append({
            "tokens": T,
            "vllm_bf16_write": bench(lambda: triton_reshape_and_cache_flash(key, val, kc, vc, slots, "auto", one, one)),
            "vllm_int8_pth_write": bench(lambda: triton_reshape_and_cache_flash_per_token_head_quant(
                key, val, kci, vci, ksc, vsc, slots, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)),
            "ours_bf16_write": bench(lambda: quant_reshape_and_cache(key, val, kc, vc, slots, None)),
            "ours_int8_fused": bench(lambda: quant_reshape_and_cache(key, val, kci, vci, slots, st)),
            "torch_int8_two_pass": bench(torch_two_pass),
            "cuda_int8_fused": bench(lambda: ck.quant_reshape_and_cache(key, val, kci, vci, slots, st)),
        })
        print(json.dumps(rows[-1]))
    return rows


def q_quant_rows():
    """Standalone q quantization kernel vs folding it into the attention prologue."""
    rows = []
    for batch, ctx in ((32, 4096), (128, 4096), (256, 1024)):
        q, k, v, cu, sl, bt = decode_case(batch, ctx)
        st = static_for(k, v)
        kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
        vq = quant.quantize(v, st["v_scale"])
        out = torch.empty_like(q)
        sg = segm(batch)
        rows.append({
            "batch": batch, "ctx": ctx,
            "quant_q_kernel": bench(lambda: quant_q(q, st["k_scale"])),
            "cuda_quant_q": bench(lambda: ck.quant_q(q, st["k_scale"])),
            "attn_fused_q": bench(lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, segm=sg)),
            "attn_prequant_q": bench(lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, segm=sg,
                                                             q_prequant=quant_q(q, st["k_scale"]))),
        })
        print(json.dumps(rows[-1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=torch.cuda.get_device_name().split()[-1].lower())
    args = ap.parse_args()
    fp8 = torch.cuda.get_device_capability()[0] >= 9
    rows = []
    for batch in (1, 8, 32, 64, 128, 256):
        for ctx in (1024, 4096, 16384, 32768):
            if batch * ctx > 4 * 1024 * 1024:
                continue
            rows.append(decode_row(batch, ctx, fp8))
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()
                              if not k.endswith("_error")}))
    res = {"gpu": torch.cuda.get_device_name(), "decode": rows, "write": write_path_rows(), "q_quant": q_quant_rows()}
    with open(f"{os.environ['OUT']}/bench_kernels_{args.tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
