# 实现日志（每一步做了什么、遇到什么问题）

## Step 1 环境与基线（阶段一）

- 平台：Modal A100-SXM4-80GB（sm_80：有 INT8 Tensor Core，没有 FP8，正好对应"没有 FP8 的卡只能用 int8"这个项目背景）。
- 软件：镜像 `vllm/vllm-openai:v0.32.0`，torch 2.13.0+cu130，Triton 3.7.1，vLLM 0.32.0。
- 基线：Qwen2.5-1.5B-Instruct（12 个 q head、2 个 kv head、head_dim 128，和课程用的 Qwen2-1.5B 同构），
  `enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_model_len=256`，跑通。
- PagedAttention 调用点（vLLM 0.32.0 源码）：
  - A100 默认 backend 是 `FLASH_ATTN`（FA2）。prefill 和 decode 都走
    `vllm/v1/attention/backends/flash_attn.py` 里 `FlashAttentionImpl.forward` 调的
    `flash_attn_varlen_func(q, key_cache, value_cache, ..., block_table=...)`（约 1400 行），
    它是 vllm_flash_attn 仓库的 FA2 kernel。
  - KV 写入：同文件 `FlashAttentionImpl.do_kv_cache_update` → `reshape_and_cache_flash`。
  - 0.32.0 默认用 V2 model runner（日志 `Using V2 Model Runner`）。
- KV cache 布局（和课程文档写的都不一样，以代码为准）：每层一个 4D 逻辑张量
  `[num_blocks, num_kv_heads, block_size, 2*head_dim]`，K 和 V 拼在最后一维（前 128 是 K、后 128 是 V）；
  默认物理布局 `LBNHC`，stride `[8192, 256, 512, 1]`，即物理上是 `[block, slot, head, K|V]`。
  所以 kernel 一律按 stride 访问，不假设连续。

遇到的问题：
1. `vllm/vllm-openai` 镜像里只有 `python3`，Modal 的 `uv_pip_install` 和函数启动都要找 `python` → 加软链。
2. Modal 的 `run_commands` 不能传以换行开头的多行脚本（会生成空 RUN）→ 插件桩脚本单独放成文件。
3. 发现 vLLM 0.32.0 已经原生有 `int8_per_token_head`（Triton backend，每个 token 每个 head 一个动态 scale，
   scale 以 fp32 内联存在 KV 页里）。它和本项目的"静态 per-channel"方案正好形成对照，后面作为基线之一。
