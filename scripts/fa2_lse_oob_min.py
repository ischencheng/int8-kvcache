# Minimal repro for docs/upstream-issue-draft.md (upstream flash-attn): one 8k prefill row plus 511 rows
# with q_len 1 and no keys, paged KV. Illegal memory access on A100.
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
print("ok")
