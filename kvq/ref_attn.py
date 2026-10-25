"""Stage 3: the bf16 golden reference for vLLM's paged attention, in plain PyTorch.

Same contract as vLLM's flash_attn_varlen_func with a paged KV cache:
  q            [total_q, Hq, D]            queries of all requests, concatenated
  k/v_cache    [num_blocks, block, Hkv, D] (any strides; vLLM hands us views)
  cu_seqlens_q [B + 1]                     query offsets
  seq_lens     [B]                         tokens of KV each request attends to
  block_table  [B, max_pages]              logical page -> physical block
Causal masking is bottom-right aligned: the last query of a request sits at
position seq_len - 1, so a decode token sees the whole context.

run_paged() does the paging (gather pages, pad, mask) and calls an `attend`
function on dense tensors, so the bf16 reference here and the INT8 variants in
int8_ref.py share exactly the same addressing code. Decode requests are batched
together; prefill requests go one at a time in row chunks to bound memory.
"""
import torch


def gather_pages(cache, block_table, max_len):
    """[num_blocks, bs, H, D] + [B, pages] -> [B, max_len, H, D] (padded rows are garbage)."""
    bs = cache.shape[1]
    pages = (max_len + bs - 1) // bs
    return cache[block_table[:, :pages].long()].flatten(1, 2)[:, :max_len]


def run_paged(attend, q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal=True, max_rows=512,
              attend_multi=None):
    """Call attend(qb [b,R,Hq,D], k [b,K,Hkv,D], v [b,K,Hkv,D], mask [b,R,K], valid [b,K]) -> [b,R,Hq,Dv].

    attend_multi, if given, is used instead for requests with more than one query token."""
    attend_multi = attend_multi or attend
    out = torch.empty(q.shape[0], q.shape[1], v_cache.shape[-1], dtype=q.dtype, device=q.device)
    dev = q.device
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    starts = cu_seqlens_q[:-1].tolist()
    lens = seq_lens.tolist()
    dec = [i for i, n in enumerate(q_lens) if n == 1]
    if dec:
        idx = torch.tensor(dec, device=dev)
        max_k = max(lens[i] for i in dec)
        rows = cu_seqlens_q[idx].long()
        valid = torch.arange(max_k, device=dev)[None] < seq_lens[idx, None]
        out[rows] = attend(q[rows][:, None], gather_pages(k_cache, block_table[idx], max_k),
                           gather_pages(v_cache, block_table[idx], max_k), valid[:, None], valid)[:, 0].to(q.dtype)
    for i, n in enumerate(q_lens):
        if n <= 1:
            continue
        k = gather_pages(k_cache, block_table[i:i + 1], lens[i])
        v = gather_pages(v_cache, block_table[i:i + 1], lens[i])
        kpos = torch.arange(lens[i], device=dev)
        valid = torch.ones(1, lens[i], dtype=torch.bool, device=dev)
        for s in range(0, n, max_rows):
            e = min(n, s + max_rows)
            qpos = lens[i] - n + torch.arange(s, e, device=dev)
            mask = (kpos[None] <= qpos[:, None]) if causal else valid.expand(e - s, -1)
            out[starts[i] + s:starts[i] + e] = attend_multi(q[starts[i] + s:starts[i] + e][None], k, v,
                                                            mask[None], valid)[0].to(q.dtype)
    return out


def group_heads(qb, num_kv_heads):
    """GQA without copying K/V: q head h reads kv head h // G, so view q as [.., Hkv, G, D]."""
    b, r, hq, d = qb.shape
    return qb.reshape(b, r, num_kv_heads, hq // num_kv_heads, d)


def masked_softmax(scores, mask):
    """fp32 softmax over keys. scores [b, Hkv, G, R, K], mask [b, R, K]."""
    return torch.softmax(scores.masked_fill(~mask[:, None, None], float("-inf")), dim=-1)


def qk_scores(qb, k):
    """[b, R, Hq, D] x [b, K, Hkv, D] -> [b, Hkv, G, R, K] in fp32."""
    return torch.einsum("brhgd,bkhd->bhgrk", group_heads(qb, k.shape[2]).float(), k.float())


def pv(p, v):
    """[b, Hkv, G, R, K] x [b, K, Hkv, D] -> [b, R, Hq, D]."""
    o = torch.einsum("bhgrk,bkhd->brhgd", p, v.float())
    return o.flatten(2, 3)


def attend_bf16(qb, k, v, mask, softmax_scale):
    return pv(masked_softmax(qk_scores(qb, k) * softmax_scale, mask), v)


def paged_attention_ref(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, softmax_scale, causal=True):
    return run_paged(lambda qb, k, v, mask, valid: attend_bf16(qb, k, v, mask, softmax_scale),
                     q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal)


def compare(ref, out):
    """Error summary used by every test and report in this repo."""
    ref, out = ref.float(), out.float()
    d = (ref - out).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_l2": ((ref - out).norm() / ref.norm().clamp(min=1e-12)).item(),
        "cos": torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item(),
    }
