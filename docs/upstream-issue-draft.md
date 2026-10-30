<!-- 给 Dao-AILab/flash-attention 的 issue 草稿，未发；原文要用户确认后才发。
     证据：results/22-fa2-lse-oob-upstream-min（下面的脚本原样跑的）、results/22-fa2-lse-oob-upstream、results/22-fa2-lse-oob（vLLM fork）。 -->

**Title:** `flash_attn_varlen_func` with `block_table`: rows with no keys write softmax_lse out of bounds

**Body:**

With `block_table`, a row that has q_len >= 1 but 0 keys writes its LSE out of bounds. Small batches silently write +inf somewhere past the LSE buffer, bigger ones crash with an illegal memory access. I hit it in a custom vLLM attention backend that gave 0 keys to decode rows it overwrites later.

Repro (flash-attn 2.8.3.post2, torch 2.8.0, 1x A100 80GB):

```python
import torch
from flash_attn import flash_attn_varlen_func

h, hk, d, page = 28, 4, 128, 256
q_lens = [8192] + [1] * 511
k_lens = [8192] + [0] * 511  # decode rows with no keys
cu_q = torch.tensor([0] + q_lens, device="cuda").cumsum(0).int()
cu_k = torch.tensor([0] + k_lens, device="cuda").cumsum(0).int()
q = torch.randn(sum(q_lens), h, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(8192 // page, page, hk, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
block_table = torch.zeros(len(q_lens), 8192 // page, dtype=torch.int32, device="cuda")
block_table[0] = torch.arange(8192 // page)
flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8192, 8192, causal=True, block_table=block_table)
torch.cuda.synchronize()
```

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Under `compute-sanitizer --tool memcheck` with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`:

```
========= Invalid __global__ write of size 4 bytes
=========     at void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, (int)4, ...>(flash::Flash_fwd_params)+0xac0
=========     by thread (0,0,0) in block (0,3,0)
=========     Address 0x2b9e76cb1400 is out of bounds
...
========= ERROR SUMMARY: 16 errors
```

With a block table, `mha_varlen_fwd` always runs the split-KV kernel and allocates `softmax_lse` as `(num_heads, total_q)` with `unpadded_lse = true`. A row with no key block to visit takes the early exit in `compute_attn_1rowblock_splitkv`, and that path writes `INFINITY` at the padded `(b, h, seqlen_q)` offset:

https://github.com/Dao-AILab/flash-attention/blob/819ef26aae39a1838d0597090880e044230d9261/csrc/flash_attn/src/flash_fwd_kernel.h#L552

The normal epilogue of the same kernel uses `bidh * total_q + q_offset` when `unpadded_lse` is set:

https://github.com/Dao-AILab/flash-attention/blob/819ef26aae39a1838d0597090880e044230d9261/csrc/flash_attn/src/flash_fwd_kernel.h#L1030-L1032

So the early exit writes at `(bidb * h + bidh) * max_seqlen_q`, which goes past `h * total_q` once bidb is large. In the run above all 14 bad writes are `lse_base + 4 * (bidb * 28 + bidh) * 8192` for their block, with the same base for all of them. With at least 1 key per row memcheck reports 0 errors. When the bad offset stays inside the buffer it lands on another row's slot, so `return_attn_probs=True` can return +inf LSE for rows that do have keys.

Fix idea: use the epilogue's offset in the early exit too (not built or tested yet):

```cpp
const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
        ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
    ) + m_block * kBlockM;
```

vllm-project/flash-attention has the same line.
