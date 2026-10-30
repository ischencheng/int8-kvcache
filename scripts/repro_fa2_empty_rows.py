"""FlashAttention-2 (vllm_flash_attn, fa_version=2) varlen + paged KV: rows with q_len >= 1 and
seqused_k = 0 write softmax_lse out of bounds.

mha_varlen_fwd allocates softmax_lse unpadded as (h, total_q) (unpadded_lse=true) and, with a block table,
always runs the split-KV kernel. That kernel's early exit (no key block to visit) writes +inf to
softmax_lse_ptr + ((bidb * h + bidh) * max_seqlen_q + m_block * kBlockM + row), the padded
(b, h, max_seqlen_q) offset; its normal epilogue uses bidh * total_q + q_offset. One variant per
subprocess (an illegal access poisons the CUDA context).

    python3 scripts/repro_fa2_empty_rows.py                    # every variant, plain
    python3 scripts/repro_fa2_empty_rows.py serving            # one variant
    python3 scripts/repro_fa2_empty_rows.py --check serving memcheck_serving.log
"""
import os
import re
import subprocess
import sys

import torch

KBLOCK_M, KBLOCK_N = 64, 128  # split-KV kernel tiles for head_dim 128 (flash_fwd_launch_template.h)
H, HK, D = 28, 4, 128  # Qwen2.5-7B
UPSTREAM = os.environ.get("FA_IMPL") == "upstream"  # Dao-AILab wheel instead of vllm_flash_attn
PAGE = 256 if UPSTREAM else 16  # upstream wants pages of 256
VARIANTS = ["serving", "serving_control", "serving_low", "serving_high", "small", "small_control", "lse", "big",
            "big_control"]


def serving_batch(seed=0):
    """The dumped call (results/18-replay-fa): 256 requests, 6442 tokens, 68 prompt chunks (q_len 39..190,
    16 cached keys each) among 188 decodes with zero keys, interleaved; max_seqlen_q 190, max_seqlen_k 206."""
    g = torch.Generator().manual_seed(seed)
    long_q = [39, 190] + torch.randint(40, 150, (66,), generator=g).tolist()
    i = 2
    while sum(long_q) != 6442 - 188:  # same token count as the dump
        long_q[i] += 1 if sum(long_q) < 6254 else -1
        long_q[i] = min(max(long_q[i], 40), 189)
        i = 2 + (i - 1) % 66
    q = long_q + [1] * 188
    k = [n + 16 for n in long_q] + [0] * 188
    perm = torch.randperm(256, generator=g).tolist()
    return [q[j] for j in perm], [k[j] for j in perm], 190, 206, 29


def small_batch():
    """Unit-test size (tests/conftest.py CASES["mixed"]); decodes get zero keys."""
    q = [1, 1, 7, 33, 128, 1, 300, 9, 4]
    k = [0, 0, 7, 500, 128, 0, 900, 2000, 64]
    return q, k, 300, 2000, 125


def lse_batch():
    """Empty row first, then long rows: every early-exit write lands inside softmax_lse, on a slot of
    another row, and the head that owns that slot ran earlier in the grid, so the +inf survives."""
    q = [1] + [512] * 7
    k = [0] + [512] * 7
    return q, k, 512, 512, 32


def big_batch():
    """One 8k prefill chunk among 511 decodes with zero keys: the bad offset grows with max_seqlen_q and
    reaches ~470 MB past softmax_lse."""
    return [8192] + [1] * 511, [8192] + [0] * 511, 8192, 8192, 512


def batch(variant):
    if variant.startswith("big"):
        q, k, mq, mk, w = big_batch()
    elif variant.startswith("small"):
        q, k, mq, mk, w = small_batch()
    elif variant == "lse":
        q, k, mq, mk, w = lse_batch()
    else:
        q, k, mq, mk, w = serving_batch()
    if variant.endswith("control"):
        k = [max(n, 1) for n in k]
    if variant in ("serving_low", "serving_high"):  # zero keys only where every head writes in (out of) bounds
        oob = predict(q, k, mq)[1]
        bad = {b for b, _, _ in oob}
        allin = {b for b in range(len(q)) if k[b] == 0} - bad
        allout = {b for b in bad if all((b, hh, 0) in oob for hh in range(H))}
        keep = allin if variant == "serving_low" else allout
        k = [n if (n > 0 or b in keep) else 1 for b, n in enumerate(k)]
    return q, k, mq, mk, w


def early_exit_writes(q, k, mq):
    """(bidb, bidh, m_block, row, lse float index) of every LSE write of the early exit (num_splits = 1,
    causal, window_size_right = 0)."""
    out = []
    for b, (ql, kl) in enumerate(zip(q, k)):
        for m in range((ql + KBLOCK_M - 1) // KBLOCK_M):
            n_max = min((kl + KBLOCK_N - 1) // KBLOCK_N, -(-((m + 1) * KBLOCK_M + kl - ql) // KBLOCK_N))
            if n_max > 0:
                continue
            for r in range(min(KBLOCK_M, ql - m * KBLOCK_M)):
                out += [(b, hh, m, r, (b * H + hh) * mq + m * KBLOCK_M + r) for hh in range(H)]
    return out


def predict(q, k, mq):
    size = H * sum(q)
    w = early_exit_writes(q, k, mq)
    oob = {(b, hh, m): idx for b, hh, m, r, idx in w if idx >= size}
    return w, oob, size


def summary(variant):
    q, k, mq, mk, _ = batch(variant)
    w, oob, size = predict(q, k, mq)
    past = (max(oob.values()) + 1 - size) * 4 if oob else 0
    print(f"{variant}: requests {len(q)}, tokens {sum(q)}, zero-key rows {sum(n == 0 for n in k)}, "
          f"max_seqlen_q {mq}, softmax_lse ({H}, {sum(q)}) = {size * 4} bytes; early-exit LSE writes {len(w)}, "
          f"out of bounds {len(oob)} (rows {sorted({b for b, _, _ in oob})[:6]}..., up to {past} bytes past the end)")


def fa_call(q, kc, vc, out, cu, mq, seqused, mk, bt, lse=False):
    if UPSTREAM:  # no seqused_k in the public API: row lengths through cu_seqlens_k
        from flash_attn import flash_attn_varlen_func

        cu_k = torch.zeros_like(cu)
        cu_k[1:] = torch.cumsum(seqused, 0)
        r = flash_attn_varlen_func(q, kc, vc, cu, cu_k, mq, mk, softmax_scale=D ** -0.5, causal=True,
                                   block_table=bt, return_attn_probs=lse)
        return (r[0], r[1]) if lse else r
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=mq, seqused_k=seqused,
                                  max_seqlen_k=mk, softmax_scale=D ** -0.5, causal=True, block_table=bt,
                                  fa_version=2, return_softmax_lse=lse)


def run(variant):
    dev = "cuda"
    q_lens, k_lens, mq, mk, width = batch(variant)
    g = torch.Generator(device=dev).manual_seed(0)
    pages = [(n + PAGE - 1) // PAGE for n in k_lens]
    # Plain runs of serving / small pass on A100 here: what follows softmax_lse is mapped (the dumped call did
    # fault, results/18-replay-fa). memcheck with the caching allocator off sees each tensor's bounds.
    q = torch.randn(sum(q_lens), H, D, generator=g, device=dev, dtype=torch.bfloat16)
    kc = torch.randn(max(sum(pages), 1), PAGE, HK, D, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    out = torch.empty_like(q)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(torch.tensor(q_lens, device=dev), 0)
    seqused = torch.tensor(k_lens, dtype=torch.int32, device=dev)
    width = max(pages) if UPSTREAM else width
    bt = torch.zeros(len(q_lens), width, dtype=torch.int32, device=dev)  # zero-key rows point at page 0
    off = 0
    for i, p in enumerate(pages):
        bt[i, :p] = torch.arange(off, off + p, dtype=torch.int32)
        off += p
    if variant != "lse":
        fa_call(q, kc, vc, out, cu, mq, seqused, mk, bt)
        torch.cuda.synchronize()
        print(f"{variant}: ok")
        return
    # Same call with the empty row given one key: no other row's LSE may change.
    _, lse = fa_call(q, kc, vc, torch.empty_like(q), cu, mq, seqused, mk, bt, lse=True)
    _, ref = fa_call(q, kc, vc, torch.empty_like(q), cu, mq, seqused.clamp(min=1), mk, bt, lse=True)
    torch.cuda.synchronize()
    lse, ref = lse.flatten().cpu(), ref.flatten().cpu()
    own = torch.zeros_like(lse, dtype=torch.bool)  # the empty row's own slots: (h, total_q)[:, 0]
    own[::int(cu[-1])] = True
    diff = set(torch.nonzero((lse != ref) & ~own).flatten().tolist())
    pred = {idx for _, _, _, _, idx in early_exit_writes(q_lens, k_lens, mq)} - set(torch.nonzero(own).flatten().tolist())
    print(f"lse: softmax_lse {tuple(lse.shape)}, {len(pred)} predicted foreign slots "
          f"{sorted(pred)[:8]}...; slots that differ from the reference: {len(diff)}, "
          f"all predicted {diff <= pred}, values {sorted({float(lse[i]) for i in diff})}")


ERR = re.compile(r"Invalid __global__ write of size (\d+) bytes")
KER = re.compile(r"at (?:void )?([\w:]+)")
BLK = re.compile(r"by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)")
ADDR = re.compile(r"(?:Access to|Address) (0x[0-9a-f]+) is out of bounds")
NEAR = re.compile(r"and is ([\d,]+) bytes (after|before) the nearest allocation at (0x[0-9a-f]+) of size ([\d,]+) bytes")


def check(variant, log):
    """Match every memcheck error against the formula: blockIdx = (m_block, bidb, bidh) for num_splits = 1;
    the LSE base implied by address - 4 * index must be one allocation, of softmax_lse's size."""
    q, k, mq, _, _ = batch(variant)
    _, oob, size = predict(q, k, mq)
    lines = open(log, errors="replace").read().splitlines()
    errs, cur = [], None
    for line in lines:
        if m := ERR.search(line):
            cur = {"size": int(m[1])}
            errs.append(cur)
        elif cur is not None:
            if "kernel" not in cur and (m := KER.search(line)) and "at " in line:
                cur["kernel"] = m[1]
            if m := BLK.search(line):
                cur["block"] = tuple(int(x) for x in m.groups()[3:])
            if m := ADDR.search(line):
                cur["addr"] = int(m[1], 16)
            if m := NEAR.search(line):
                cur["near"] = (int(m[3], 16), int(m[4].replace(",", "")))
    kernels = sorted({e.get("kernel", "?").split("<")[0] for e in errs})
    other = [l for l in lines if "=========" in l and ("Invalid" in l or "Program hit" in l) and not ERR.search(l)]
    print(f"--check {variant}: {len(errs)} invalid global writes, sizes {sorted({e['size'] for e in errs})}, "
          f"kernels {kernels}; other errors {len(other)} {other[:2]}")
    summary_line = [l for l in lines if "ERROR SUMMARY" in l]
    print("   ", summary_line[-1].strip("= ") if summary_line else "no ERROR SUMMARY line")
    if not errs:
        print(f"    predicted out-of-bounds writes {len(oob)}")
        return
    hit = [e for e in errs if "block" in e and (e["block"][1], e["block"][2], e["block"][0]) in oob]
    bases = {e["addr"] - 4 * oob[(e["block"][1], e["block"][2], e["block"][0])] for e in hit if "addr" in e}
    nears = {e["near"] for e in errs if "near" in e}
    print(f"    errors whose block (m_block, bidb, bidh) is a predicted out-of-bounds early exit: {len(hit)}/{len(errs)}"
          f" (predicted {len(oob)}; the launch is aborted after the first errors, so memcheck reports a subset)")
    print(f"    LSE base implied by address - 4*index: {[hex(b) for b in sorted(bases)][:4]} ({len(bases)} distinct)")
    print(f"    nearest allocation(s) reported: {[(hex(a), s) for a, s in sorted(nears)][:4]}; "
          f"softmax_lse is {size * 4} bytes")
    if len(bases) == 1:
        b = bases.pop()
        nxt = b + -(-size * 4 // 512) * 512  # allocations are 512-byte aligned
        print(f"    implied base is a reported allocation of softmax_lse's size: {(b, size * 4) in nears}; "
              f"or the allocation right after it (upstream allocates rng_state there): "
              f"{any(a == nxt for a, _ in nears)}")
    ex = next(e for e in errs if "addr" in e)
    blk = ex["block"]
    print(f"    e.g. block {blk}: address {hex(ex['addr'])} = base + 4 * (({blk[1]}*{H} + {blk[2]})*{mq} + "
          f"{blk[0]}*{KBLOCK_M}) = base + {4 * oob.get((blk[1], blk[2], blk[0]), -1)} bytes")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        check(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        for v in VARIANTS:
            summary(v)
        for v in VARIANTS:
            r = subprocess.run([sys.executable, __file__, v], capture_output=True, text=True)
            lines = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l or l.startswith(v)]
            print(f"{v}: exit {r.returncode}", lines[-1:] if lines else "")
