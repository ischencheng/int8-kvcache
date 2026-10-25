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

## Step 2 采集 KV 并选量化维度（阶段二）

做法：`kvq/capture.py` 由 vLLM 插件装进每个 vLLM 进程，包住 FlashAttention backend 的
`do_kv_cache_update`（K/V 写 cache 的唯一入口）和 `forward`：
- 每层记录 |K|、|V| 的每 (kv_head, channel) 最大值，并抽样保存 16384 个 token 的 K/V 行；
- 第一个"纯 decode 且上下文 ≥1536"的 batch，在第 0/14/27 层把 q、用到的 KV 页（压缩重编号）、
  block_table、seq_lens、cu_seqlens_q、输出整份存成 safetensors（给阶段三/四做真实数据对拍）。
  课程给的是 pickle 全量 dump（79 MB 里 99.9% 是没用到的页），这里只存用到的页，也不用 pickle。

数据：Qwen2.5-7B-Instruct 和 1.5B，AIME24 30 题 + GSM8K 200 题，greedy，max_tokens 2048，
7B 共 117k token、最长序列 2289。

结论（`results/02-capture/kv_analysis_qwen2.5-7b.md`，每层 INT8 往返相对误差）：

| 粒度 | K 平均误差 | V 平均误差 | 能否静态 |
|---|---|---|---|
| per_tensor | 2.46% | 2.50% | 能 |
| per_head | 1.90% | 1.96% | 能 |
| per_channel `[H, D]` | **0.61%** | **1.06%** | 能 |
| per_token | 1.82% | 1.04% | 不能 |
| per_token_head（vLLM 原生 int8） | 1.29% | 0.79% | 不能 |

- K 有强烈的**通道**离群：第 0 层同一 head 内通道最大/中位数 40 倍，第 27 层 |K| 最大 422、
  通道比 74、head 间比 21。离群集中在每个 head 的 52–63 和 116–127 通道（neox RoPE 的低频维度），
  跨 token 几乎是常数（`k_channel_box_qwen2.5-7b.png`），来源是 Qwen2 的 qkv bias。
  → K 用 per_channel。
- V 没有明显通道结构（通道比 2–5），按 token 变化更多：动态 per_token_head 最好，静态里 per_channel 最好。
  → V 也用 per_channel（静态方案要求 scale 与 token 无关）。
- 99.9% 分位截断反而更差（第 4 层 K 0.68%→0.91%，最后几层 V 1.1%→2.0%）：K/V 的离群值是有用信号，
  截掉的误差比省下的分辨率大。先用 max，端到端再比一次（扩展 3.5(a)）。

遇到的问题 / 修正：
1. **"往返误差"这个指标对 K 有误导**。K 里跨 token 不变的分量 μ（每通道均值）对 softmax 不可见：
   `softmax(q·(k_t+μ)) = softmax(q·k_t + q·μ)`，q·μ 对同一个 query 的所有 key 相同。
   把误差改成只算 attention 看得见的部分 `k - μ` 之后：
   per_channel 平均 1.53%（第 0 层 6.6%、第 27 层 9.5%），**per_token_head 平均 5.0%（第 27 层 46.9%）**。
   离群通道吃掉了 scale，真正有信息的小变化被量化掉了。这预示 vLLM 原生 int8_per_token_head
   在 Qwen2.5 上可能掉点，后面端到端验证。
2. 由此得到一个零成本的改进：**smooth-K**，校准时记下每通道均值 μ，写 cache 前存 `k - μ` 的 int8，
   attention 里完全不用加回来（softmax 平移不变）。可见误差 1.53% → **0.90%**，第 27 层 9.5% → 1.0%。
   代价是所有参与同一个 softmax 的 K 都必须减同一个 μ（prefill 新算的 K 和 cache 里读出来的 K 不能混用）。
3. Python 钩子只在 eager 下生效：full CUDA graph 回放时不会回到 Python，所以采集必须 `enforce_eager=True`。

## Step 3 bf16 PyTorch 金标准 + 三级验证（阶段三）

`kvq/ref_attn.py`：和 `flash_attn_varlen_func`（分页模式）同语义的纯 PyTorch 实现。
- `run_paged()` 负责分页寻址：按 `block_table` 把页 gather 成连续张量、padding、建 mask，
  然后调一个只看稠密张量的 `attend()`。阶段四的 int8 版本复用同一个 `run_paged()`，
  只换 `attend()`，保证"计算流程一行不动，只改数值部分"。
- causal mask 右下角对齐：请求最后一个 query 的位置是 `seq_len - 1`，decode token 能看到全部上下文。
- GQA 用分组 einsum：q 视为 `[Hkv, G, D]`，K/V 不复制（q head h 读 kv head h // G）。

三级验证（`results/03-golden-ref/`）：
1. 真实数据：vLLM FA2 用 dump 的输入重算，6 份 dump（7B/1.5B × 第 0/14/27 层，batch 8–10，上下文 1536）
   **逐位相等**，说明 dump 和调用方式都没问题；PyTorch 金标准对 dump 输出 rel_l2 ≈ 0.8–1.2e-3，cos > 0.999999。
2. 随机数据：随机分页（页打散在 2 倍大的池子里）、GQA 28/4 和 12/2、纯 decode 批和"decode + 7/33/128 长度
   prefill 块"混合批，对 FA2 rel_l2 0.9–1.8e-3。
3. 整网集成：插件把 `FlashAttentionImpl.forward` 里的 kernel 换成 PyTorch 金标准（`KVQ_ATTN=ref`），
   Qwen2.5-7B GSM8K 250 题 greedy：FA2 91.6%，金标准 91.6%（2 题变对、2 题变错），47.6% 的输出逐字相同。
   差异来自 fp32 参考 vs FA2 内部 bf16 的 P·V，属于正常数值翻转。

遇到的问题：
1. 第一版参考实现把所有请求 padding 到最长再算，prefill 块进来时 scores 张量过大，整网跑时 PyTorch 反复 OOM 重试。
   改成"decode 请求一起向量化，prefill 请求逐个按 512 行分块"。
2. GQA 用 `repeat_interleave` 把 K/V 扩到 28 个 head 再转 fp32，250 个请求一起 decode 时一次 3–4 GB，
   仍然 OOM 告警；改成分组 einsum 后告警消失，整网替换从 119 s 降到 50 s。
3. `index_put` 要求 dtype 一致：attend 输出 fp32、out 是 bf16，要先 `.to(q.dtype)`。
4. 课程的参考实现没把 `softmax_scale` 传进去（固定 1/√d），这里显式传入；dump 里也存了 scale。

## Step 4 两类 int8 实现与误差拆分（阶段四）

`kvq/int8_ref.py`，都挂在阶段三的 `run_paged()` 上，只换 `attend()`：
- **实现 A**：K/V 量化 → 立刻反量化（先 fp32 再 bf16）→ bf16 金标准。误差只来自 K/V 的 int8 舍入，
  也是"读出来就反量化"的 kernel 和 chunked prefill 的真实计算路径。
- **实现 B**：真 int8 算术。
  - QK：scale 能不能提到点积外，只看它是否依赖被缩并的 head_dim。per_head / per_token_head 的 K scale 不依赖 →
    `scores = (q_i8·k_i8)·s_q·s_k`；per_channel 依赖 → 先把 `s_k[h,:]` 乘进 q（GQA 下按 q head → kv head 取行），再量化 q。
  - PV：缩并 token 维，per_channel/per_head 的 V scale 不依赖 token → 乘在输出上。
  - P：对 `exp(s - rowmax)` 按（行, 64 个 key 的 tile）动态量化，模拟 flash kernel 每个 tile 做一次 int8 PV mma。
  - 整数点积在 float64 里算（int8×int8×128 项 < 2^53，精确），避免 TF32 干扰。
- 单测 `tests/test_int8_ref.py`（28 个）：舍入误差 ≤ scale/2；q、P 不量化时 B 必须和 fp64 下"反量化后再算"
  完全一致（rel < 1e-6），专门检查 scale 折叠/提取的代数，per_head / per_channel / per_token_head / per_tensor 都过。

真实 decode dump（7B/1.5B × 第 0/14/27 层，上下文 1536）上 attention 输出相对误差，参考为 FA2 捕获输出
（`results/04-int8-ref/int8_error_table.md`，动态 scale = 每个请求自己的 K/V）：

| 配置 | 平均 | 最差层 |
|---|---|---|
| fp32 金标准（参照） | 0.10% | 0.12% |
| A per_head | 12.27% | 28.46% |
| A per_token_head（vLLM 原生做法） | 11.10% | 26.98% |
| A per_channel | 2.47% | 4.42% |
| **A per_channel + smooth-K** | **1.06%** | 1.78% |
| A per_token_head + smooth-K | 1.17% | 1.52% |
| B per_channel | 8.29% | 13.53% |
| B per_channel + smooth-K（q 每 token-head 一个 scale） | 2.21% | 3.36% |
| **B per_channel + smooth-K + q 每 32 通道一个 scale** | **1.71%** | 2.46% |
| B per_channel，q per-tensor | 36.98% | 72.73% |
| B per_channel + smooth-K，P 固定 scale 1/127 | 6.40% | 9.34% |

误差拆分（B per_channel + smooth-K = 2.21%）：KV 量化 1.06%（= 实现 A）；P 的 tile 量化约 +0.2%
（不量化 P 是 2.03%）；q 量化约 +0.9%（q 不量化是 1.32%）。B 相对 A 的增量 ≈ 1.2%，远在课程说的 5–10% 以内。

遇到的问题：
1. **实现 B 不加 smooth-K 会掉到 8.3%**，比实现 A 差 3 倍多，而且不是 P 的问题（不量化 P 还是 8.2%）。
   原因：per_channel 的 `s_k` 在离群通道上大 40–74 倍，折进 q 之后 q' 的通道间动态范围被撑大，
   一个 scale 量化 q' 时正常通道被压没了；而离群通道贡献的主要是对 softmax 不可见的常数项。
   smooth-K 先去掉常数偏移，离群通道的 `s_k` 变小，q' 范围恢复正常 → 2.2%。所以在 Qwen2.5 上，
   **per_channel 的真 int8 QK 必须配 smooth-K**。
2. per_token_head 的问题同源：离群通道的常数偏移吃掉了每个 token 的 scale。加 smooth-K 后从 11.1% 到 1.17%。
   这说明 vLLM 原生 int8_per_token_head 的主要问题在 Qwen2.5 的 K 偏移，不在粒度本身。
3. **P 不能用固定 scale 1/127**：`exp(s-m)` 虽然在 [0,1]，但注意力分散或有 attention sink 时，大部分 token 的 p
   远小于 1/254，一律舍成 0，分子丢掉大量质量（1.5B 第 27 层 1.9% → 9.4%）。按 tile 取最大值做 scale 才稳。
4. q 用 per-tensor（课程正文的写法）不可用：动态 scale 被整个 batch 里最大的 q 决定，平均 37%。
5. 为了让 scale 计算不碰到 padding 位置的脏数据（gather 出来超出 seq_len 的行是别的请求的页），
   动态 absmax/均值都必须先按 `valid` mask。
