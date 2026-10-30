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

## Step 6a 静态量化的框架改造（阶段六 a–e，PyTorch 版）

改动分两层：
1. **vLLM 源码 patch**（`patches/vllm-0.32.0-int8-static-kv.patch`，镜像构建时打到已安装的 vLLM 上），只做"类型登记"：
   - `vllm/config/cache.py`：`CacheDType` 加 `"int8_static"`（pydantic 用 Literal 校验，插件里改不了，只能改源码）；
   - `vllm/utils/torch_utils.py`：`"int8_static" → torch.int8`，于是 KV cache 按 1 字节/元素分配，block 数自动翻倍；
   - `vllm/v1/kv_cache_interface.py`：`KVQuantMode.INT8_STATIC`，让 `is_quantized_kv_cache()` 等判断把它当成量化 cache。
2. **自定义 attention backend**（`kvq/vllm_int8/backend.py`），插件里 `register_backend(AttentionBackendEnum.CUSTOM, ...)`，
   运行时 `kv_cache_dtype="int8_static", attention_config={"backend": "CUSTOM"}`。继承 Triton backend 的元数据：
   - (a) cache dtype 是 int8（由 patch 决定）；
   - (b)(e) `do_kv_cache_update`：写 cache 前用静态 scale 量化（K 先减 smooth-K 均值），prefill/decode 同一入口；
   - (c) 多 token 请求（prefill、chunked prefill、投机解码验证）读 int8 → 反量化（实现 A）；
   - (d) decode 走 int8 算术（实现 B），q 动态量化，K/V 静态 scale；
   - 静态 scale 在 **metadata builder 的 `__init__`** 里交给每层：它在模型加载之后、CUDA graph capture 之前、
     每个 KV cache group 只构造一次，并且拿得到 `layer_names` 和 `static_forward_context`
     （impl 的构造函数拿不到层名，forward 里懒加载又可能撞上 graph capture）。TP 下按 rank 切 kv head。

校准（`scripts/calibrate.py` + `kvq/scales.py`）：在 FA2 上开采集，统计每 (kv_head, channel) 的 |x| 最大值、
有符号 max/min 和 sum（smooth-K 需要均值 μ 和 max|k-μ| = max(kmax-μ, μ-kmin)），一次生成
per_channel/per_head × max/99.9%/99.99% × 是否 smooth 共 6 种 scale 文件。校准集不碰评测集
（数学用 GSM8K **train**，另备一份 wikitext 文本做跨域对照）。

冒烟测试（`results/06-smoke/`，GSM8K train 64 题校准，test 50 题评测）：
- `GPU KV cache size`：bf16 968,272 token → int8_static **1,936,544 token**（正好 2 倍，同样的显存）。
- 真 int8 存储（CUSTOM backend）92%；"bf16 cache + 读时按同一静态 scale 量化"（课程 3.3 的二分工具，
  `KVQ_ATTN=static:B_channel_smooth`）92%，**50 题输出逐字相同**。说明写时量化/int8 存取/读出的框架改动和
  纯数学实现严格等价；以后静态量化掉点，就可以用这个工具二分是 scale 的问题还是框架的问题。

遇到的问题：
1. 我们模块的 logger 用 `init_logger(__name__)` 时 INFO 不打印：vLLM 只给 `vllm.*` 命名空间配了 handler，改名 `vllm.kvq`。
2. `AttentionBackendEnum[backend.get_name()]` 会在 Attention 构造时被调用，自定义 backend 的 `get_name()` 必须返回
   注册时用的枚举名 `"CUSTOM"`，不能沿用父类的 `"TRITON_ATTN"`。

## Step 7a 三个 Triton kernel（阶段七）

`kvq/triton_kernels/`：
1. **`quant_cache.py` 量化 + reshape_and_cache 融合**：每个 program 处理一个 token 的所有 kv head，
   读一次 bf16 K/V → 减 μ → `div_rn` 除 scale → `rint`（四舍六入五成双，和 torch.round 一致）→ clamp → 按
   slot_mapping 写进分页 cache；slot < 0（padding）直接返回。`static=None` 时就是普通的 bf16 写 cache。
2. **`quant_q.py` q 量化**：q × s_k[h // G, :]（折叠 per-channel K scale）后按行或按 32 通道一组求 scale 量化。
3. **`int8_attention.py` 统一的分页 attention**（prefill / chunked prefill / decode / 投机验证一个 kernel）：
   - 一个 program 负责 BLOCK_M = BLOCK_Q 个 token × G 个 q head（同一个 kv head，GQA 下 K/V tile 只读一次给 G 个 head 用），
     按 block_table 以 64 token 为一个 tile 走页，online softmax；
   - 小批量纯 decode 时把 KV 范围切 16 段（flash-decoding），第二个 kernel 合并；直接复用 vLLM Triton backend
     metadata builder 里已经分配好的 segment 缓冲区（CUDA graph 下地址不变）；
   - MODE 0 = bf16 cache（同一份代码的 bf16 基线，用来公平对比）；MODE 1 = int8 cache 逐 tile 反量化成 bf16 再算
     （实现 A，prefill 用）；MODE 2 = int8 Tensor Core（实现 B，decode 用）：序言里折叠 s_k 并量化 q，
     `tl.dot(int8, int8) → int32`，P 按（行, tile）量化成 int8 再和 int8 V 做 `tl.dot`，最后乘 s_v；
   - `q_groups=4` 时 QK 拆成 4 个 k=32 的 int8 点积，每段乘自己的 q scale（对应阶段四的 group32）；
   - smooth-K 在 kernel 里什么都不用做（cache 里存的就是 k-μ）。

测试（`tests/test_triton_kernels.py`, `tests/test_cudagraph.py`，A100 和 L4 上 57 个全过，`results/07-kernels-test-l4/`）：
- 融合写 cache 与 PyTorch "量化 + scatter" **逐位相等**（per_channel / per_head，含 padding slot）；q 量化逐位相等；
- MODE 0 对 fp32 金标准 rel 1–2e-3（decode / 混合 / 纯 prefill，GQA 28/4 和 12/2）；MODE 1 对实现 A 8e-4；
- MODE 2 对 PyTorch 实现 B ≈ 1e-5（融合 q、预量化 q、group32、16 段切分、per_head 都测了）；投机验证批（每请求 5 个 query）也一致；
- 真实 dump 上 int8 kernel 对 FA2 输出 1.5%–3.3%（token-head q），group32 1.1%–2.7%，和阶段四的数一致；
- CUDA graph：capture 后换输入 replay 与 eager 逐位相等。

遇到的问题：
1. **用 dump 算"静态 scale"时第 27 层误差 14.5%**，查下来是我把压缩后 cache 的整页都算进了统计：
   每个请求最后一页没写满的 slot 里是别的请求留下的旧数据，把 μ 和 max 都带偏了。改成只用 block_table/seq_lens
   覆盖到的有效 token 后降到 3.3%。vLLM 里的采集是在写 cache 时对新 token 统计的，天然没有这个问题；但凡是"直接读 cache 做统计"
   的工具都要注意。
2. **CUDA graph 里 scale 必须原地更新**：graph 记录的是张量地址。`static["v_scale"] = new` 换对象之后 replay 仍读旧值
   （测试里输出完全不变），只有 `copy_/mul_` 原地修改才生效。所以静态 scale 要在 capture 前分配好，之后只原地改。
3. Triton 的 `/` 是近似除法，量化写 cache 用 `tl.div_rn` 才能和 PyTorch 逐位对上；取整用 `libdevice.rint`（不是 `.to(int)` 的截断）。
4. int8 的 `tl.dot` 要求 K 维 ≥ 32：QK 的 K 维是 head_dim（128 没问题），PV 的 K 维是 tile 长度（取 64）；
   group32 只能用在 head_dim 128（每组 32），head_dim 64 的模型（Qwen2.5-0.5B）只能用整行 q scale。

## Step 7b CUDA 版 kernel（阶段七，CUDA 实现）

Triton 版之外，三个 kernel 都用 CUDA 重写了一遍（`kvq/cuda/int8_kv.cu`，`kvq/cuda_kernels.py` 用
`torch.utils.cpp_extension.load` 在容器里 JIT 编译），backend 里 `KVQ_KERNEL=cuda` 切换。

1. `quant_cache_kernel`：一个 block 一个 token，每个线程 8 个通道：16 B 读 bf16、8 B 写 int8；
   `__fdiv_rn` + `__float2int_rn`（四舍六入五成双）保证和 PyTorch 逐位一致。
2. `quant_q_kernel`：一个 warp 一个 (token, q head)，每 lane 4 个通道，warp shuffle 求 amax；支持整行 / 32 通道一组。
3. `int8_decode_attn_kernel`（decode，每请求 1 个 query token）：
   - grid = (请求, kv head, split)，block = 4 warp；m16 的 16 行 = 这个 kv head 的 G 个 q head（G ≤ 16，多余行置零）。
   - 序言：128 线程 8 个一组处理一行，折叠 s_k、求行 amax、量化 q，写 smem；每个 warp 读出 4 个 k-step 的 A fragment 常驻寄存器。
   - 主循环：warp w 处理第 w, w+4, ... 个 32-token chunk（4 个 warp 各自独立做 online softmax，最后在 smem 合并，
     循环里只需要 `__syncwarp`）。加载时 8 个 lane 读一行（16 B × 8），一条指令 4 个 token、完全合并访存。
   - QK：`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`，4 个 n-tile（8 token）× 4 个 k-step（32 通道）= 16 条 mma；
     K 在 smem 里行距 144 B（128+16 padding），B fragment 的 32 位读取无 bank 冲突。
   - **P 不经过任何寄存器搬移就当 PV 的 A 操作数**：QK 累加器里线程 (g,t) 拿到的是 token 8j+2t+e，
     而 PV 的 A fragment 需要 k-slot 4t..4t+3、16+4t..16+4t+3。PV 是对 token 求和，只要 P 和 V 用同一个 token 置换就等价，
     所以让 P 按原寄存器顺序填 slot，把 V 转置进 smem 时按同一置换放列（`slot_of_token`），B fragment 就是一条 32 位读取。
   - P 按（行, 32-token chunk）量化：四个线程（同一个 quad 持有同一行）`__shfl_xor` 求行最大值；分母用量化后的 P 求和。
   - PV：16 个 n-tile（每个 8 个通道）× 1 个 k-step（32 个置换后的 token）= 16 条 mma，int32 结果乘 chunk 的 P scale 累加到 fp32。
   - 合并：每个 warp 的 (m, l) 写 smem，按 `exp2(m_w - M)` 加权，acc 用 smem 原子加（复用 K/V 缓冲区），
     收尾时乘 s_v 写 bf16；小批量 decode 时 grid.z 切 16 段，`split_reduce_kernel` 合并（复用 vLLM 的 segment 缓冲区）。
   - smem 约 40 KB / block（不用申请超过 48 KB 的动态 smem）。

测试（`tests/test_cuda_kernels.py`，23 个，L4 上一次编译通过，`results/07b-cuda-test/`）：
- 融合写 cache、q 量化与 PyTorch 逐位相等；
- decode attention 对 PyTorch 实现 B（P 每 32 token 量化的配置）rel ≈ 0–8e-5（GQA 28/4、12/2、32/8、16/1，切分与不切分），
  per_head scale 约 1e-3；与 Triton 版差 2e-3（Triton 每 64 token 量化一次 P）；
- 真实 dump 上对 FA2 输出 1.0%–3.2%，和 Triton 版一致。

遇到的问题：
1. `ATen/cuda/CUDAContext.h` 会 include cuSPARSE/cuBLAS 的头文件，vLLM 镜像里只有 nvcc 和 runtime 头文件 → 编译失败；
   取 stream 只需要 `c10/cuda/CUDAStream.h`。
2. vLLM 镜像预设了 `TORCH_CUDA_ARCH_LIST`（7 个架构），JIT 编译会全编一遍；改成只编当前 GPU 的架构。
3. 这个 CUDA kernel 只做 decode：prefill / chunked prefill 是计算密集，int8 访存优势不明显，课程方案里也是反量化后走 bf16，
   所以多 token 批次仍走 Triton 的反量化 kernel。group32 的 q scale 只在 Triton 版里实现了。

## Step 5 动态量化端到端验精度（阶段五）

做法：插件把 FA2 的 attention 换成阶段四的 PyTorch 实现（`KVQ_ATTN=<配置名>`），KV cache 仍是 bf16，每步读出来按
请求现场求 scale 量化（动态）。Qwen2.5-7B-Instruct，enforce_eager，greedy：
- GSM8K test 全量 1319 题（`\boxed{}` 抽答案），逐题和 bf16 对比翻转数；
- ΔNLL：用 bf16 自己的 greedy 回答（GSM8K 前 200 条 + AIME24 30 条，后者是 2k 级长回答）做固定文本，
  teacher-forcing 算回答部分的平均 NLL 与 bf16 的差。连续量，噪声远小于准确率。

结果（`results/05-dynamic/`，`docs/tables/stage5_dynamic.md`）：

| 配置 | GSM8K | 翻转（对/错） | ΔNLL gsm8k | ΔNLL aime24 |
|---|---|---|---|---|
| bf16 FA2 | 92.19% | – | 0 | 0 |
| A per_head | 89.16% | +17 / −57 | +0.0016 | +0.0035 |
| A per_token_head（vLLM 原生做法） | 92.72% | +21 / −14 | +0.0016 | +0.0022 |
| A per_channel | 91.89% | +10 / −14 | +0.0002 | +0.0006 |
| A per_channel + smooth-K | 91.58% | +6 / −14 | **+0.0001** | **+0.0003** |
| B per_head | 91.36% | +25 / −36 | +0.0004 | +0.0038 |
| **B per_channel（不加 smooth-K）** | **63.08%** | +13 / **−397** | +0.0221 | +0.0161 |
| B per_channel + smooth-K | 91.43% | +13 / −23 | +0.0005 | +0.0006 |
| B per_channel + smooth-K + group32 q | **92.04%** | +11 / −13 | +0.0002 | +0.0003 |

结论：
- 技术路线可行：per_channel + smooth-K 在 KV 量化（A）和真 int8 计算（B + group32）两种实现下都和 bf16 打平
  （±1 个点内，ΔNLL ≤ 0.0003）。
- 阶段四的算子级误差排序在端到端完全对得上：B per_channel 不加 smooth-K 的 8.3% 算子误差变成 **GSM8K 掉 29 个点**。
- per_head 掉 3 个点，长文本（AIME 回答）上 ΔNLL 更大（+0.0035/+0.0038）：误差随上下文累积。

遇到的问题：
1. **准确率会骗人**：per_token_head 的 92.72% 比 bf16 还高，但它的 ΔNLL（+0.0016）和掉了 3 个点的 per_head 一样差，
   只是在 1319 题上恰好翻对的比翻错的多。同一个配置的翻转有 30–60 题，±1 个点基本是噪声；排序以 ΔNLL 为准。
2. PyTorch 版动态量化很慢（每步对整段上下文 gather + 量化，一个配置 GSM8K 要 12 分钟），8 个配置串行会超过 3 小时的
   函数时限。改成先单独生成 bf16 参考答案存进 Volume，再 3 台 A100 并行。第一次串行跑了 50 分钟才发现，停掉重来浪费了约 $2。

## Step 7c 用 ncu 把 CUDA kernel 调到超过 Triton（阶段七性能）

第一版 CUDA kernel 正确但慢：A100 上 batch 32 × ctx 32k 要 1514 µs（Triton int8 892，FA2 1402），
小 batch 还有约 60 µs 的固定开销。按"先保精度再做性能"的顺序，用 ncu 一轮一轮定位：

| 版本 | 改了什么 | b32×32k | b64×4k | b32×4k（16 段切分） | b1×4k |
|---|---|---|---|---|---|
| v1 | 逐字节写 V^T 到 smem，读完再算，smem 原子加合并 | 1514 | 405 | 417 | 80 |
| v2 | cp.async 两级流水；K 行置换 + 寄存器内 4×4 字节转置，去掉 V^T 缓冲 | 1173 | 372 | 560 | 64 |
| v3 | 合并去掉 smem 原子加；block_size 用移位，每个 chunk 只查 2 次页表并提前一轮取 | **743** | **246** | **154** | **24.7** |
| Triton int8 | （对照） | 892 | 256 | 162 | 24.5 |
| FA2 bf16 | （对照） | 1402 | 485 | 224 | 37 |

每一步的依据（`results/08-ncu-cuda-hot/`、`results/08-graph-vs-eager/`）：
1. v1 的 ncu：每个 scheduler 只有 3 个 warp，每发射一条指令平均等 9.4 周期，43% 在等 L1TEX（全局读 + local memory），
   DRAM 只有 34%。原因：每个 chunk 读完才开始算（访存延迟完全暴露）；V 转置是每 lane 每 chunk 128 次逐字节 smem 写；
   局部数组做 `reinterpret_cast` 被放进 local memory。→ v2：cp.async 预取下一个 chunk；
   把置换挪到 K 的读取行号上（不花指令），V 保持原顺序，每个线程用 4 条 32 位 LDS + 8 条 `__byte_perm` 拿到 4 个 n-tile 的 B fragment，
   输出通道随之置换（`chan_of`），合并时放回原位。
2. v2 大 batch 变快了，但切分模式反而更慢。先用 CUDA graph 回放排除 host 开销（graph 和 eager 一样慢），
   再用 ncu 源码级采样：**75% 的 stall 在合并阶段的 smem `atomicAdd(float)`**。sm_80 上 shared memory 的 float 原子加
   编译成 `ATOMS.CAS` 循环（FADD + CAS + ISETP + BRA），4 个 warp 抢同一批地址；切分模式每个 block 的主循环很短，合并就成了大头。
   不切分时 stall 则主要是整数指令（SHF/IMAD/IADD3）：每个 token 都在用运行时 block_size 做除法取模和 64 位乘法。
   → v3：每个 warp 把自己的部分和写进独立的 smem 区（行距 129 个 float，32 个 lane 落在不同 bank），收尾时 4 份相加；
   block_size 只支持 2 的幂、用移位，一个 32-token chunk 最多跨 2 页，只查 2 次 block_table，并且提前一轮把下一批页号读进寄存器。

v3 在 A100 上的有效带宽 1.45 TB/s（峰值约 2.0 TB/s），比 FA2 bf16 快 1.9–2.0 倍，比 Triton int8 快 0–24%。
23 个 CUDA 测试在每一版都重新跑过。

## Step 6b 完整静态量化：校准、max vs 分位数、跨数据集（阶段六 + 扩展 3.5(a)）

校准（`scripts/calibrate.py`）：数学集 GSM8K **train** 512 题（模型作答，19.4 万 token，含 prefill 和 decode 的 KV），
文本集 wikitext-2 train 256 段（只 prefill，36.0 万 token）。主方案的 scale 文件放在 `scales/`（每个约 170 KB）。
评测：`int8_static` + CUSTOM backend + Triton kernel，GSM8K test 全量（`results/06-static/`，`docs/tables/stage6_static.md`）：

| 配置 | GSM8K | 翻转 |
|---|---|---|
| bf16 FA2 | 92.19% | – |
| **per_channel max + smooth-K，decode int8 算术（主方案）** | **91.89%** | +10 / −14 |
| 同上，decode 也反量化（实现 A） | 92.12% | +14 / −15 |
| 同上，q 每 32 通道一个 scale | 91.66% | +9 / −16 |
| per_channel max，不加 smooth-K | 89.46% | +14 / −50 |
| per_channel **99.9% 分位** + smooth-K | **78.24%** | +27 / −211 |
| per_channel **99.99% 分位** + smooth-K | **78.70%** | +28 / −206 |
| per_head max + smooth-K | 92.27% | +18 / −17 |
| wikitext 校准 → GSM8K 评测（跨域） | 91.58% | +9 / −17 |
| PyTorch 路径 vs Triton 路径（同一 scale，250 题） | 93.2% vs 92.8% | 2 / 3 题不同 |

结论：
- **静态 per_channel + smooth-K 基本无损**（−0.3 个点，翻转数和 bf16 自身的数值噪声同量级）。
- **scale 必须取最大值**：99.9% / 99.99% 分位截断都掉到 78% 左右（ΔNLL +0.10–0.13）。被截掉的是 attention sink 之类的少数
  极端 token，它们的 K 偏离恰恰是 attention 要用的信号。课程建议用分位数，实测在 Qwen2.5 的 KV 上不成立（和阶段二的往返误差结论一致）。
- **校准集可以跨域**：文本校准用在数学题上 91.58%，ΔNLL ≈ 0；校准只需要覆盖 K/V 的数值范围，不需要和任务同分布。
- smooth-K 让 per_head 也能用了（静态 per_head + smooth 92.27%；阶段五动态 per_head 不加 smooth 是 89.2–91.4%）。
- 不加 smooth-K 的静态版掉 2.7 个点，和阶段五的结论一致（decode 的 int8 QK 需要 smooth-K）。

遇到的问题：
1. **ΔNLL 在静态 backend 下只测到了 prefill 路径**：teacher-forcing 全是 prefill（多 token 请求），而 backend 对多 token 请求走
   反量化（实现 A），所以"不加 smooth-K"这一项准确率掉 2.7 点、ΔNLL 却是 +0.0000。要测 decode 的 int8 算术，
   得让所有行都走 MODE 2（`KVQ_INT8_MAX_Q` 设大），见下一步的补测。
2. 个别配置 ΔNLL 略小于 0（−0.0003 / −0.0006）：bf16 自己的 greedy 输出不一定是任意扰动模型下概率最大的序列，
   量化噪声让某些 token 的分布变尖时 NLL 会小幅下降；这个量级应视为"没有退化"，不解读成"变好"。

## Step 8a kernel 基准（阶段七/八，A100）

`scripts/bench_kernels.py`，Qwen2.5-7B 形状（28 q head / 4 kv head / D 128 / block 16），decode（每请求 1 个 query），
`triton.testing.do_bench`（L2 冲刷、CUDA event 计时），所有 Triton kernel 用同一套 split-KV 策略。完整表在
`docs/tables/kernels_a100.md`，摘录（µs）：

| batch × ctx | FA2 bf16 | vLLM Triton bf16 | vLLM int8_per_token_head | 我们 Triton bf16 | 我们 Triton int8 | **我们 CUDA int8** | CUDA / FA2 |
|---|---|---|---|---|---|---|---|
| 1 × 4k | 38.2 | 40.4 | 41.8 | 24.7 | 25.3 | 25.2 | 1.52× |
| 1 × 32k | 89.6 | 213.2 | 235.4 | 99.6 | 107.3 | 66.6 | 1.35× |
| 8 × 32k | 368.9 | 360.1 | 528.0 | 351.0 | 260.9 | 216.8 | 1.70× |
| 32 × 32k | 1322 | 1356 | 1551 | 1302 | 855 | **703** | **1.88×** |
| 64 × 16k | 1757 | 1374 | 1312 | 1288 | 943 | **853** | **2.06×** |
| 128 × 32k | 5584 | 5822 | 4697 | 5035 | 3696 | **2709** | **2.06×** |
| 256 × 16k | 5034 | 5152 | 4644 | 5240 | 3528 | **2545** | **1.98×** |

- 有效 KV 带宽（必须读的字节数 / 时间）：CUDA int8 1.36–1.45 TB/s，Triton int8 1.15–1.26 TB/s，我们的 bf16 kernel 1.6–1.7 TB/s，
  FA2 1.2–1.7 TB/s（A100 峰值约 2.0 TB/s）。int8 kernel 每字节要做的指令更多（反量化、P 量化），带宽利用率比 bf16 低，
  但读的字节减半，所以净收益 1.4–2.1 倍；**上下文越长、batch 越大收益越大**（课程的判断成立）。
- 小 batch 短上下文（1 × 1k）是延迟瓶颈，int8 没有收益，CUDA 版还略慢（固定开销：q 量化序言 + 合并）。
- **vLLM 原生 int8_per_token_head 在多数点比 bf16 还慢**（32 × 32k：1551 vs FA2 1322）：它读出 int8 后转成 bf16 再用 bf16
  Tensor Core，还要额外读每个 (token, head) 的 fp32 scale 并在 S/P 上逐列乘，访存省下的被这些开销吃掉了。
- 写 cache：融合的量化+写入 8192 token 28 µs（Triton）/ 33 µs（CUDA），**比 vLLM 原生 bf16 写入（42 µs）还快**（只写一半字节）；
  不融合的两遍 PyTorch 实现 310 µs；vLLM 的 per_token_head 写入 70 µs（要现算每个 token 的 scale）。
- q 量化：单独 kernel 8–13 µs；融合进 attention 序言后 attention 耗时不变（159 vs 159 µs），所以融合版是对的选择。

## Step 8b H100 上和 FP8（FA3）对比

同一个基准在 H100 上跑（`docs/tables/kernels_h100.md`），加上 vLLM 自带的 FA3 FP8 KV（每 kv head 一个 descale）作为
"有 FP8 的卡上的标准做法"：

| batch × ctx | FA2 bf16 | **FA3 FP8** | 我们 Triton int8 | **我们 CUDA int8** | CUDA / FA3-FP8 | CUDA / FA2 |
|---|---|---|---|---|---|---|
| 1 × 32k | 59.5 | 39.2 | 94.1 | 45.4 | 0.86× | 1.31× |
| 8 × 32k | 216.7 | 130.5 | 128.8 | 140.5 | 0.93× | 1.54× |
| 32 × 32k | 747.9 | 432.2 | 462.6 | 425.5 | 1.02× | 1.76× |
| 64 × 16k | 790.3 | 436.2 | 656.7 | 417.3 | 1.05× | 1.89× |
| 128 × 32k | 3081 | 1643 | 2447 | 1761 | 0.93× | 1.75× |
| 256 × 16k | 3099 | 1660 | 1977 | 1601 | 1.04× | 1.94× |

- 课程说"int8 kernel 和 FP8 实现理论上应接近"：**成立**，我们的 CUDA int8 和 FA3 FP8 在 ±15% 以内。
  两者读的字节数一样（1 字节/元素），差别在实现：FA3 用 Hopper 的 wgmma + TMA 异步拷贝，我们的 kernel 是为 A100 写的
  `mma.sync` + cp.async，在 H100 上没用到这些新硬件，长上下文小 batch（1 × 32k）落后 14%。
- 在 H100 上 Triton int8 明显落后于 CUDA（128 × 32k：2447 vs 1761 µs），Triton 版的 int8 dot 布局和流水在 Hopper 上效率不高。
- vLLM 原生 int8_per_token_head 在 H100 上同样比 bf16 慢。

## Step 6c 补测：静态方案 decode 路径（int8 算术）的 ΔNLL

把所有行都放到 MODE 2（`KVQ_INT8_MAX_Q=1000000`，prefill 也走 int8 QK/PV），teacher-forcing 才覆盖 decode 的数值
（`results/06-static-nllB/`）：

| 静态配置 | ΔNLL gsm8k | ΔNLL aime24（2k 级长回答） |
|---|---|---|
| per_channel max，不加 smooth-K | +0.0023 | **+0.0112** |
| per_channel max + smooth-K | +0.0004 | +0.0019 |
| 同上 + q 每 32 通道 scale | −0.0000 | +0.0015 |
| per_head max + smooth-K | −0.0002 | +0.0006 |
| wikitext 校准的 per_channel + smooth-K | −0.0006 | +0.0000 |

和生成准确率一致：不加 smooth-K 的 int8 QK 明显退化，而且**长文本上退化大 5 倍**（误差随上下文累积）；
加 smooth-K 后长文本 ΔNLL ≤ 0.002。

## Step 8c 最终版 ncu（A100，batch 64 × ctx 4096，不切分）

`results/08-ncu-final/`（第一次重跑时 ncu 报 "driver resource unavailable"，宿主机上 DCGM 占着性能计数器，换个容器重跑就好了）：

| kernel | 耗时 µs | DRAM 吞吐 | SM 吞吐 | 寄存器/线程 | 实际占用率 | 每条指令等待周期 |
|---|---|---|---|---|---|---|
| FA2 bf16 | 431.7 | 61.4% | 26.0% | 178 | 12.1% | 8.6 |
| 我们 Triton bf16 | 308.7 | **85.8%** | 19.9% | 156 | 15.0% | 11.0 |
| 我们 Triton int8 | 245.8 | 54.2% | 32.6% | 156 | 15.0% | 6.75 |
| **我们 CUDA int8（v3）** | **224.5** | 59.4% | 23.6% | 168 | 11.6% | **5.49** |

- bf16 kernel 已经接近带宽上限（86%），int8 kernel 只有 54–59%：读的字节减半，但每字节要做的事（int8 mma、反量化、P 量化）多了，
  开始受指令发射和延迟限制。CUDA v3 的每条指令等待从 v1 的 9.4 周期降到 5.5。
- 剩下的提升空间：占用率（76 KB smem → 每个 SM 只能放 2 个 block，8 个 warp）；int32→float 的 I2F（可以用 2^23 魔数加法代替）；
  Hopper 上换 wgmma + TMA。这几项没有继续做。

## Step 9e 扩展 3.5(e)：和 W8A8 / W4 权重量化叠加

模型：RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8（INT8 权重 + INT8 动态激活，A100 上走 CUTLASS int8 GEMM）、
Qwen/Qwen2.5-7B-Instruct-AWQ（W4A16）。每个模型都用自己重新校准 KV scale，同时对照"直接沿用 bf16 模型的 scale"
（`results/11-wquant/`，GSM8K 1319 题，CUDA graph 开启）：

| 模型 | bf16 KV | int8 KV（本模型校准） | int8 KV（沿用 bf16 模型的 scale） |
|---|---|---|---|
| bf16 原模型 | 92.19% | 91.89% | – |
| W8A8 INT8 | 92.34% | 92.19% | 91.89% |
| W4A16 AWQ | 90.98% | 90.90% | 91.43% |

- **能兼容，原因**：KV 量化作用在 K/V 投影的**输出**上。W8A8 的 int8 GEMM 在 epilogue 里就反量化成 bf16，
  W4A16 本来就输出 bf16，到 RoPE 和写 cache 时数据流已经是 bf16，KV 量化看到的只是一个分布略有变化的 bf16 张量，
  和权重 / 激活怎么量化没有耦合。两种模型叠加 int8 KV 都不额外掉点。
- **scale 要不要重新校准**：逐通道看，W8A8 / AWQ 模型的 k_scale 和 bf16 模型最多差 48% / 44%，v_scale 差 70% / 55%
  （k_mu 的相对差很大是因为有些通道均值接近 0，相对差没有意义）。但沿用 bf16 的 scale 准确率仍在噪声内：
  max 型 per-channel scale 对小幅分布漂移不敏感。工程上还是建议对量化后的模型重新校准（成本只有几分钟）。
- **W4A8**：vLLM 0.32 在 GPU 上没有真正的 W4A8-int 路径。`CompressedTensorsW4A8Int` 构造 kernel 时
  `act_type=params_dtype`（bf16），实际跑的是 W4A16 的混合精度 kernel，int8 激活只有 CPU kernel；
  W4A8-FP8 需要 Hopper，而且 Qwen2.5-7B 没有公开的 W4A8 checkpoint。按上面的原因，W4A8 的激活量化同样发生在 GEMM 输入侧，
  K/V 投影输出仍是 bf16，和 int8 KV 不冲突，所以用 W4A16 作为 4bit 权重的代表来验证。

遇到的问题：RedHatAI 的 w4a16 checkpoint 用的是 `actorder='group'`，vLLM 0.32 直接拒绝加载
（"group/dynamic activation ordering is no longer supported"），换成 Qwen 官方的 AWQ。

## Step 9c 扩展 3.5(c)：投机解码

**怎么隔离 draft 的 KV**：vLLM 0.32 的 `speculative_config` 有 `kv_cache_dtype`（draft 单独的 KV dtype，
不设就继承 target 的）和 `attention_backend`（draft 的 backend 默认**独立自动选择**，不继承 target）。所以：
- target int8 + draft bf16：`speculative_config.kv_cache_dtype="auto"`，draft 自动选 FA2；
  如果不设，draft 会继承 `int8_static` 却拿到 FA2，直接报 dtype 不支持。
- target int8 + draft int8：`speculative_config.attention_backend="CUSTOM"`；draft 的层名带 `draft_model.` 前缀，
  backend 的 builder 从 `KVQ_DRAFT_SCALES`（draft 模型单独校准）取它的 scale。
- 两种 dtype 并存时，vLLM 的 hybrid KV cache manager 会按各自的页大小管理（target 1 字节/元素，draft 2 字节/元素）。

**实验**（`results/10-specdec/`，GSM8K 500 题）：vLLM 的 draft_model 要求 target 和 draft 词表大小相同，
Qwen2.5-7B 是 152064、0.5B 是 151936，所以 draft_model 实验用 Qwen2.5-1.5B（target）+ 0.5B（draft），k=4；
另在 7B 上用 ngram（无 draft 模型，只有 target 的 verify 批次读 int8 cache）：

| 配置 | GSM8K | 接受率 | 平均接受长度（≤5） | 各位置接受数 |
|---|---|---|---|---|
| 1.5B 无投机，bf16 KV | 73.8% | – | – | – |
| 1.5B 无投机，int8 KV | 75.0% | – | – | – |
| draft：target bf16 / draft bf16 | 74.8% | 87.0% | 4.48 | 31425 / 29714 / 28214 / 26877 |
| draft：target int8 / draft bf16 | 74.0% | 86.7% | 4.47 | 31665 / 29873 / 28328 / 26957 |
| draft：target int8 / draft int8 | 75.2% | 86.7% | 4.47 | 31367 / 29632 / 28075 / 26763 |
| 同上 + verify 也走 int8 算术 | 74.2% | 86.8% | 4.47 | 31419 / 29661 / 28134 / 26779 |
| 7B ngram，bf16 KV | 92.0% | 11.1% | 1.44 | |
| 7B ngram，int8 KV | 92.0% | 11.1% | 1.45 | |
| 7B ngram，int8 KV + verify int8 算术 | 92.0% | 11.1% | 1.45 | |

结论：
- **接受率几乎不受影响**（87.0% → 86.7%），draft 的 KV 量化与否、verify 走反量化还是 int8 算术都一样。
  原因：接受与否取决于 draft 和 target 的 argmax 是否一致，KV 量化带来的 logits 扰动远小于两个模型之间本来的差距。
- draft 的 KV 可以一起量化：draft 自己的 cache 也省一半，接受率不变；draft 模型小、cache 也小，收益主要还在 target。
- 投机验证批次（每请求 k+1 个 query）默认走反量化路径（多 token 请求），`KVQ_INT8_MAX_Q=8` 可以让它也走 int8 算术，精度不变。

遇到的问题：
1. 7B + 0.5B 直接报 "Target and draft model should have the same vocabulary size"，换成 1.5B + 0.5B。
2. 离线 `LLM` 默认 `disable_log_stats=True`，`get_metrics()` 报 "Stat logging disabled"，拿不到接受率；第一轮白跑，打开后重跑。
3. 用随机 token 做 prompt 的 TPOT 基准不能衡量投机解码：随机前缀让模型续写乱码，draft 几乎猜不中；大 batch 下
   （GSM8K 500 题一起跑）投机解码本来就比不开慢（16.5k → 7–10k tok/s，计算已饱和）。加速比要在 batch 1、真实 prompt 下测，见下一步。

## Step 9a 扩展 3.5(a)：分布式（TP）下的校准和 scale 加载

2×L40S（这个实验看正确性，不看性能；sm_89 同样有 int8 Tensor Core），Qwen2.5-7B TP=2，4 个 kv head → 每个 rank 2 个
（`results/13-tp/`，GSM8K 1319 题）：

| 配置 | GSM8K | KV 容量（token） |
|---|---|---|
| TP=2 bf16 | 91.51% | 1,111,088 |
| TP=2 int8，用单卡校准的 scale、按 rank 切 head | 92.42% | 2,198,784 |
| TP=2 int8，TP=2 下分布式校准的 scale | 92.72% | 2,203,152 |

- **加载**：scale 文件存全局的 `[num_kv_heads, D]`，builder 里按 `get_tensor_model_parallel_rank()` 切出本 rank 的行
  （`kvq/scales.py:shard`）；kv head 数少于 TP 数时 vLLM 会复制 head，取 `rank * H // tp` 那一行。
- **分布式校准**：每个 rank 只看到自己的 kv head，`collective_rpc` 把各 rank 的统计量收回来按 rank 顺序在 head 维拼接
  （`scales.merge_tp`），复制的 head 只留一份；DP 副本之间对 max/min 取最大最小、sum 和 token 数相加即可（本实验没开 DP）。
- 两份 scale 逐通道最多差 64%（k_scale）/ 48%（v_scale），但准确率一样。差异来自校准数据本身不同：TP 改变累加顺序，
  模型 greedy 生成的校准 token 也跟着变，而 max 对极端 token 敏感。不是拼接顺序错误（那样离群 head 会差 10–20 倍）。

## Step 9f 进阶：MLA（DeepSeek-V2-Lite）的 int8 KV

MLA 每个 token 只缓存一行 latent：`c = kv_c_normed`（512 维）+ `k_pe`（64 维 RoPE 部分）。吸收 W_UK/W_UV 后的 decode 是
`s_t = (q_lat·c_t + q_pe·kpe_t)·scale`，`o_lat = Σ p_t c_t`，再乘 W_UV。**同一个 c 既当 K（对 512 个通道缩并）又当 V（对 token 缩并）**，
所以和 GQA 的分析不同（`kvq/mla.py` 顶部注释）：
- QK 角色：per-channel 的 s_c 在缩并维上 → 折进 q_lat（和 GQA 的 K 一样）；
- PV 角色：s_c 与 token 无关 → 乘在 o_lat 上；
- **smooth 的 μ 在 PV 角色里必须加回来**：`Σ p_t (c_t − μ) = o_lat − μ`（Σp=1）。softmax 平移不变只对 QK 成立。
  `tests/test_mla_ref.py` 专门验证了"不加回 μ 输出就错"；
- k_pe 带 RoPE（随位置变化），可以单独一组 scale，也可以留 bf16（64/576 通道）。

做法（`scripts/ext_mla.sh`，A100，`TRITON_MLA` backend，`results/12-mla/`）：
1. 采集 27 层的 c / k_pe 统计和 3 层 decode dump（`KVQ_MLA=capture`，包住 `MLAAttentionImpl.do_kv_cache_update`
   和 `TritonMLAImpl.forward_mqa`；MLA impl 不知道层名，按第一次调用的顺序编号）；
2. 分布：c 的通道最大/中位数比中位 4.6、最大 15.4（Qwen 的 K 最高 74），per_channel 往返误差 0.89%，
   **smooth 几乎没用（0.88%）**：latent 经过 `kv_a_layernorm`，没有 Qwen2 那种 qkv bias 带来的常数偏移；k_pe per_channel 0.40%；
3. decode 算子误差（对 vLLM Triton MLA kernel 的捕获输出，参考实现本身 0.18%）：

| 层 | A（反量化） | A + smooth | A + smooth + k_pe int8 | B（int8 算术） | B + smooth | B + smooth + k_pe int8 |
|---|---|---|---|---|---|---|
| 0 | 1.03% | 1.64% | 1.60% | 2.29% | 3.43% | 3.34% |
| 13 | 1.04% | 1.08% | 1.41% | 1.79% | 1.71% | 1.96% |
| 26 | 1.47% | 1.44% | 1.53% | 1.99% | 2.05% | 2.08% |

4. 端到端：在写 cache 前把 c（和可选的 k_pe）按静态 scale 量化再反量化（= int8 latent cache 实际存的东西），GSM8K 1319 题：

| 配置 | GSM8K | 翻转（对/错） |
|---|---|---|
| bf16 | 72.33% | – |
| c int8 per_channel，k_pe bf16 | 71.34% | +67 / −80 |
| c int8 per_channel + smooth（μ 加回），k_pe bf16 | 71.04% | +67 / −84 |
| c 和 k_pe 都 int8，smooth | 72.02% | +69 / −73 |

结论：MLA 的 latent 比 Qwen2.5 的 K 好量化得多，per_channel 不加 smooth 就够；差异在 ±1 个点（约 1 个标准误，
这个模型翻转数本来就多），可以全部 576 维都存 int8（显存 1152 → 576 B/token）。

遇到的问题：
1. **MLA 的 ΔNLL 测不到量化**（都在 1e-5 级）：MLA 的 prefill（`forward_mha`）对当前 chunk 直接用新算的 `kv_c_normed`，
   不读 cache，而 teacher-forcing 的文本一个 chunk 就处理完，伪量化只改了 cache 里的那份。decode（`forward_mqa`）才读 cache，
   所以 MLA 以生成准确率为准。GQA 那边 FA backend 的 prefill 也读 cache，ΔNLL 是有效的。
2. 第一版 dump 触发条件是"batch 里最短序列 ≥1024"，GSM8K 回答短，永远等不到；改成最长序列 ≥600。

### 3.5(c) 补测：batch 1 的加速比（`results/10-spec-latency/`，24 道 GSM8K 逐条跑，engine 时间戳算每请求 TPOT）

| 配置 | TPOT | 相对无投机 |
|---|---|---|
| 7B 无投机 bf16 | 10.18 ms | – |
| 7B 无投机 int8（CUDA kernel） | 10.01 ms | – |
| 7B ngram bf16 | 8.06 ms | 1.26× |
| 7B ngram int8 | 8.14 ms | 1.23× |
| 1.5B 无投机 bf16 / int8 | 3.87 / 3.52 ms | – |
| 1.5B + 0.5B draft（bf16/bf16、int8/bf16、int8/int8） | 13.9–18.8 ms | 变慢 |

- **int8 KV 不改变投机解码的加速比**（ngram 1.26× vs 1.23×）。batch 1、短上下文时一步的时间主要是读 15 GB 权重，KV 只占一小部分，
  所以 int8 本身只快 2%（1.5B 上 9%）。
- 1.5B + 0.5B 的 draft_model 在 batch 1 反而慢 3–5 倍：draft 只比 target 小 3 倍，每步跑 4 次 draft + 1 次 verify，
  这么小的模型上调度和 kernel 启动开销占主导。这是这组模型搭配的问题，与 KV 量化无关（三种 KV 组合同样慢）。
- 这个容器中途被 Modal 抢占（preemption）后整段重跑，两次的 TPOT 在 1–3% 内一致（draft 那组 14–18 ms 波动较大）。

## Step 8d 端到端：精度（CUDA graph 打开）、容量、吞吐、TPOT（阶段八 + 扩展 3.5(b)）

Qwen2.5-7B，A100，vLLM 默认的 CUDA graph 模式（FULL_AND_PIECEWISE）打开（`results/09-e2e/`）。

**精度**（GSM8K 1319 题）：

| 配置 | GSM8K | 翻转 | ΔNLL gsm8k / aime24 |
|---|---|---|---|
| bf16 FA2 | 92.19% | – | 0 / 0 |
| 我们 int8_static + Triton，CUDA graph | 92.12% | +12 / −13 | （见 Step 6c）|
| 我们 int8_static + **CUDA kernel**，CUDA graph | **92.19%** | +12 / −12 | |
| vLLM 原生 int8_per_token_head | 91.36% | +11 / −22 | +0.0017 / +0.0026 |

**KV 容量**（同样 0.9 显存利用率）：bf16 1,034,368 token → int8_static 2,068,896–2,071,520 token（2.0×）；
vLLM int8_per_token_head 2,006,208（scale 内联在页里，略少）。

**容量受限的吞吐**（512 个请求 × 6144 上下文 + 1024 输出，`max_num_seqs=512`）：

| 配置 | KV 允许的最大并发 | 输出吞吐 |
|---|---|---|
| bf16 FA2 | 143 | 1115 tok/s |
| 我们 int8 + Triton | 287 | 1361 tok/s（+22%） |
| **我们 int8 + CUDA kernel** | 287 | **1431 tok/s（+28%）** |
| vLLM int8_per_token_head | 278 | 676 tok/s（−39%） |

并发翻倍，再加上 decode attention 更快，吞吐提升 22–28%。vLLM 的 int8_per_token_head 虽然并发也翻倍，但它的 kernel
（尤其是 prefill：ctx 16k × 32 的 prefill 146 s，FA2 39 s）太慢，吞吐反而掉 39%。

**TPOT（差值法）**（`docs/tables/stage8_e2e_diff_method.md`）：`(带 128 个输出的总时间 − 只 prefill 的时间) / 127`，
摘录（ms）：

| ctx × batch | FA2 bf16 | 我们 Triton bf16 | 我们 Triton int8 | 我们 CUDA int8 | vLLM int8 原生 | 我们 int8 eager |
|---|---|---|---|---|---|---|
| 1k × 1 | 10.36 | 9.84 | 9.88 | 9.92 | 10.11 | 22.62 |
| 4k × 64 | 29.05 | 21.55 | 20.47 | 19.26 | 40.18 | 21.76 |
| 4k × 128 | 55.89 | 34.08 | 31.65 | 28.46 | 59.77 | 32.68 |
| 16k × 32 | 50.55 | 31.04 | 27.45 | 25.77 | 62.05 | 29.11 |

- int8 的收益在大 batch、长上下文才出现（4k × 128：同一份 kernel bf16 34.1 → int8 31.7（Triton）/ 28.5（CUDA））；
  batch 1 时每步主要是读 15 GB 权重，KV 那点访存省不出差别。
- **3.5(b) CUDA graph**：int8 backend 完全兼容（CUDA graph 下精度 92.12%/92.19%，和 eager 一样）。关掉 graph 后
  batch 1 的 TPOT 从 9.9 ms 涨到 22.6 ms：attention 之外的几百个小 kernel 的启动开销。兼容需要做的事：
  静态 scale 在 metadata builder 里分配（早于 capture），之后只能原地改；kernel 里不能有 host 同步，所有分支只依赖
  capture 时就固定的形状（decode 批大小、是否切分）；padding slot 写成 −1 由写 cache 的 kernel 跳过；
  CUDA 扩展在模型构建时就 JIT 编译好；split-KV 的缓冲区复用 builder 预分配的那一份。
- 差值法的问题：chunked prefill 下，先 prefill 完的请求会和别人的 prefill chunk 混在同一批里 decode，这些步也被算进"decode"，
  prefill 越慢的配置被高估越多（FA2 一列偏高）。所以又用 engine 的每请求时间戳补测了一遍，见下一步。

## Step 9d 扩展 3.5(d)：PD 分离 vs 混部

**怎么做的**：vLLM 的 PD 分离靠 KV connector 把 P 实例算好的 KV 页搬给 D 实例。主力是 NixlConnector（UCX/RDMA），
但 **Modal 的 gVisor 沙箱里没有 `/sys/class/net`**，UCX 的 TCP 传输枚举不到网卡，共享内存传输又不支持 NIXL 要求的
peer failure 处理，engine 起不来（`results/` 里的调试日志：`scandir(/sys/class/net) failed` → `NIXL_ERR_BACKEND`）。
所以改用 vLLM 自带的 ExampleConnector：P 实例（kv_producer，只做 prefill）把每个请求每层的 KV 页按 prompt 哈希写成 safetensors，
D 实例（kv_consumer）找到后直接注入自己的分页 cache、跳过 prompt 的 prefill。两个**独立的 engine**先后在同一张卡上跑
（`scripts/pd_offline.py`），500 道 GSM8K 一次性提交（高并发），TTFT/TPOT 用 engine 时间戳逐请求统计（`results/14-pd/`）：

| | KV 每 token 字节 | 存储总量（5.3 万 prompt token） | D 端 GSM8K | D 端 TTFT | D 端 TPOT | D 端吞吐 |
|---|---|---|---|---|---|---|
| PD 分离 bf16 | 52,978 | 2.81 GB | 92.6% | 7.33 s | 34.4 ms | 5266 tok/s |
| PD 分离 int8 | **26,499** | **1.40 GB** | 92.6% | 8.90 s | **31.6 ms** | 5228 tok/s |
| 混部 bf16 | – | – | 92.6% | 5.36 s | 32.2 ms | 6187 tok/s |
| 混部 int8 | – | – | 93.0% | 6.55 s | **28.6 ms** | 6124 tok/s |

结论：
- **int8 KV 在 P/D 之间搬运完全正确**：D 端精度和 bf16、和混部都一样（92.6%）。int8 页按字节原样存取，静态 scale 两边各自
  从同一个文件加载，**scale 不需要随 KV 一起传**（这是静态 scale 相对 per-token 动态 scale 的一个好处：后者的 scale 是 KV 的一部分）。
- **传输量正好减半**：52,978 → 26,499 B/token（28 层 × 4 head × 256 × 2/1 字节，加 safetensors 头）。
- D 端（纯 decode）TPOT 降 8%，混部降 11%（500 并发、短上下文，KV 读取只占一部分）。
- TTFT 变高（7.3 → 8.9 s，混部 5.4 → 6.5 s）有两个原因：(1) 500 个请求同时提交，TTFT 主要是排队时间；
  (2) **我们的 prefill 走自己写的反量化 Triton kernel，比 FA2 的 prefill 慢**（Step 8d 里 16k × 16 的 prefill 23 s vs 19 s）。
  第二点是当前实现真实的代价：prefill 若改成"读 int8 → 反量化成 bf16 临时页 → 调 FA2"，或者继续优化 MODE 1 kernel，可以消掉。
- 这个环境测不了真实网络传输下"字节减半 → TTFT 降多少"。课程说"传输时间摊到 TPOT、不计入 TTFT"取决于 connector：
  NIXL 这类 pull 模式是 D 收到请求后再拉 KV，拉取时间计入 D 端 TTFT。

遇到的问题（按时间）：
1. NixlConnector 在 gVisor 里起不来（见上）。
2. 第一版在线方案（P/D 两个 server + 我写的 FastAPI 代理）：`tail -50 文件1 文件2` 在 GNU tail 里不合法，报错没打出来；
   而且 P/D 起不来时函数直接 return 没杀进程，后面几组会抢端口。改成 `tail -n` + 统一 `cleanup`。
3. 单发请求代理是通的，但并发 64 时客户端收到 `Connection reset by peer`，bench 成功数为 0。换成两个离线 engine 的做法。
4. ExampleConnector 把一个请求的 KV 按"计算它的那一步"写文件，chunked prefill 把长 prompt 切成几步时只留下最后一段
   （D 端 `value tensor [64,4,256] cannot be broadcast to [80,4,256]`）。P/D 两边都关掉 chunked prefill、单步 token 预算设 16384 后正常。

### 8d 补测 1：每请求 TPOT（engine 的首/末 token 时间戳，`results/09-e2e-tpot-req/`）

| ctx × batch | FA2 bf16 | 我们 Triton bf16 | 我们 Triton int8 | 我们 CUDA int8 | vLLM int8 原生 |
|---|---|---|---|---|---|
| 1k × 1 | 10.40 | 9.92 | 9.97 | 10.04 | 10.22 |
| 1k × 64 | 27.56 | 28.04 | 29.31 | 26.30 | 34.12 |
| 4k × 64 | 84.72 | 82.31 | 83.67 | 82.89 | 145.63 |
| 4k × 128 | 165.95 | 157.65 | 159.64 | 157.02 | 273.81 |
| 16k × 16 | 90.08 | 99.79 | 107.27 | 106.60 | 295.90 |

这个口径下 int8 看不出收益，16k × 16 甚至比 FA2 慢。原因：所有请求同时到达，chunked prefill 下一个请求从首 token 到末 token
的大部分步里都夹着别人的 8192 token prefill chunk，每步的时间主要是 prefill；而我们的 prefill（反量化 kernel）比 FA2 慢。
也就是说，**"请求同时涌入"的场景下 TPOT 由 prefill 速度决定**，int8 decode kernel 的收益要在 prefill 结束之后才体现。
差值法和这个口径都没把 decode 单独拎出来，所以再用两个输出长度做差测纯 decode（下一节）。

### 8d 补测 2：纯 decode TPOT（两个输出长度做差，`results/09-e2e-tpot-pure/`、`results/09-e2e-tpot-pure-rep/`）

同一批请求（随机 token、prefix cache 关）分别生成 L = 128 和 2L = 256 个 token（`bench_e2e.py --pure`）。两次的 prefill
完全一样，多出来的 L 步都在所有 prefill 结束之后，所以 `(T(2L) − T(L)) / L` 就是纯 decode 一步的时间。A100，CUDA graph 开，ms：

| ctx × batch | FA2 bf16 | 我们 Triton bf16 | 我们 Triton int8 | 我们 CUDA int8 | vLLM int8 原生 |
|---|---|---|---|---|---|
| 1k × 128（单次，作废） | 20.77 | 18.56 | 15.53 | 17.97 | 28.03 |
| 1k × 128（重复 3 次中位数） | 21.20 | 19.98 | 19.14 | **18.33** | – |
| 4k × 64 | 27.09 | 23.71 | 21.39 | **20.57** | 43.58 |
| 4k × 128 | 36.88 | 32.87 | 28.37 | **24.07** | 58.09 |
| 4k × 128（重复 3 次中位数） | 37.69 | 33.78 | 30.52 | **26.27** | – |
| 16k × 16 | 20.72 | 19.92 | 16.88 | **15.85** | 35.47 |
| 16k × 32 | 30.37 | 29.92 | 23.80 | **21.56** | 57.10 |

- 这个口径下 int8 的收益完整体现：CUDA int8 比 FA2 快 13–35%（4k × 128：36.9 → 24.1 ms），同一份 Triton 代码
  bf16 → int8 快 4–20%，KV 总量越大收益越大。和 kernel 测试的量级对得上：4k × 128 的 kernel 差
  （735 − 373）µs × 28 层 ≈ 10 ms，TPOT 差 12.8 ms。
- vLLM 原生 int8_per_token_head 每步比 FA2 慢 58–88%（后四个点；1k × 128 只有单次值，不算）。kernel 测试里 128 × 4k 它反而比 FA2 快（605 vs 735 µs），
  端到端每步却多 21 ms（每层 0.75 ms，比它整个 kernel 还长），所以慢在 attention kernel 之外。它是对照组，没追。
- 单次测量的 1k × 128 一行作废：它是每个配置测的第一个点，Triton 的首次 JIT 编译落进了 T(L)，TPOT 被算小。
  主任务里五个配置是在同一个容器里按顺序跑的，编译缓存留在磁盘上，所以偏差大小因配置而异：我们的 bf16 最先跑，
  要编 bf16 kernel，偏 1.4 ms；Triton int8 要编 int8 kernel，偏 3.6 ms；CUDA int8 的 prefill kernel 已经在缓存里，
  只偏 0.4 ms，和 FA2 一样。`--repeat` 加了每点预热，换一台 A100 把 1k × 128 和 4k × 128 各重复 3 次：
  重复之间差 0.1–0.9 ms（4k × 128 的两个 int8 配置例外：Triton 29.3–30.6、CUDA 24.0–26.6）；两台机器的同一个点差 2–9%。
- 三种口径的教训：差值法混进了 prefill chunk；每请求时间戳在请求同时涌入时由 prefill 主导；只有纯 decode 能看 decode kernel 的收益。

## Step 10 去掉 Triton，只用 CUDA（`results/15-*` 到 `results/19-*`）

起因：CUDA 版在所有点都比 Triton 版快，两套 kernel 没必要并存。Triton 版留在 tag `triton-backend`。

**删之前要先接住 Triton 负责的两类请求**（CUDA kernel 原来只做 q_len = 1）：

1. 短 query（q_len ≤ 8：投机验证、短 chunk）扩进 CUDA INT8 kernel：m16 的行从"一个 token 的 G 个 q head"推广成
   (token, head) 对 R = t·G + h，超过 16 行沿 grid.z 切 m-tile；每行因果上限 L − q_len + 1 + t。
   这样均匀的验证批也只走 INT8 kernel，backend 的 CUDA graph 支持从继承来的 `ALWAYS` 改成 `UNIFORM_BATCH`，
   验证批仍在 full graph 里（冒烟里 ngram 投机也捕获了 FULL graph）。
2. 长 query（prefill）：把这些请求的 int8 页反量化成 bf16 临时页（`[page, block, H, D]`），交给 vLLM 的 FA2。
   vLLM 0.32 的 V2 runner 不把 decode 排到批前面，所以不切批：FA2 先跑整批，INT8 kernel 再覆盖短请求的行。
3. 参考实现 `paged_static_attention` 加 `short_q`：q_len ≤ short_q 的请求按实现 B 算。CUDA 测试：短 query 对参考
   rel_l2 ≤ 2.3e-4，混合批长请求 8e-4（bf16 的 FA2），短请求逐位一致。

**冒烟过了，1319 题一跑就 illegal memory access**，最后查出三个问题：

| 现象 | 原因 | 修法 |
|---|---|---|
| `KVQ_DEBUG=1` 的 CPU 检查报"要反量化的源页号越界"，而 forward 时页表本身是对的 | 临时页的源页号在 metadata builder 里从页表 gather，读到的不是这一步的页表 | builder 只在 CPU 上记 (请求, 页序号)，反量化 kernel 在 forward 时自己查页表；临时页表也在第一层 forward 才拷到 GPU |
| 检查全过，`CUDA_LAUNCH_BLOCKING=1` 指向 FA2 | FA2 自己的 bug：`seqused_k = 0` 的行把 softmax_lse 写出界（离线重放：只要没有 0 行就不崩；根因见下） | 短请求给它自己 q_len 个 key（读临时第 0 页，结果随后被覆盖） |
| `KVQ_SHORT_Q=1`（验证批走 FA2）掉到 29%；默认配置的 int8 draft 接受率 81.2%（应为 ~86.7%） | `seq_lens_cpu_upper_bound` 在 async 投机解码下是乐观上界，被我当成了 FA2 的 key 数；draft 模型处理 prompt 时也用到了它 | CPU 长度只决定拷几页，key 数在 forward 时用 GPU 上的精确长度算。修后接受率 86.6%，sq1 75.0% |

查法：每批 kernel 前的 CPU 端越界检查（`KVQ_DEBUG=1`）→ 同步启动定位 kernel → 把出错那次 FA2 调用的 metadata 存下来
（`KVQ_DEBUG_DUMP`）→ `scripts/replay_fa_call.py` 用随机张量离线重放，一次改一个因素。中间猜过"FA2 按整个 KV tile 查页表、
临时页表太窄"，重放证明不是；加宽页表没坏处，保留了。另外，我们扩展里的 launcher 没有在 kernel 后检查错误，
同步模式下越界也会算到下一个 CUDA 调用头上，这次靠转储时的 `.cpu()` 同步排除了反量化 kernel。

**FA2 越界的根因**（`scripts/repro_fa2_empty_rows.py`，`results/22-fa2-lse-oob*`，A100）：`mha_varlen_fwd` 传了
`block_table` 就走 split-KV kernel（num_splits = 1），`softmax_lse` 按 `(h, total_q)` 分配（`unpadded_lse = true`）。
没有 key block 可算的行提前退出，把 +inf 写到 `(bidb·h + bidh)·max_seqlen_q + m_block·64`，这是 padded
`(b, h, max_seqlen_q)` 的偏移，没看 `unpadded_lse`；同一个 kernel 的正常收尾用的是 `bidh·total_q + q_offset`
（vLLM 的 flash-attention fork，`flash_fwd_kernel.h` 第 552 行 vs 第 1045 行）。dump 那批 LSE 只有 721,504 字节，bidb 大约 ≥ 34 的空行全写出界，最远超出 4.7 MB。

- memcheck（`PYTORCH_NO_CUDA_MEMORY_CACHING=1`）：165 个 4 字节越界写，全在 `flash_fwd_splitkv_kernel`，block 都是预测的空行，
  `地址 − 4 × 偏移` 是同一个基址，正是那块 721,504 字节的分配。每行至少 1 个 key：0 个错误。
- 单测大小的批（`CASES["mixed"]`）只超出 146 KB，留在 PyTorch 缓存池里，不崩，memcheck 也看不到；关掉缓存才报。
  重建的 dump 批直接跑也没崩（后面的地址正好映射着）。一个 8k prefill + 511 个空 decode 的批超出 ~470 MB，直接跑就 illegal memory access。
- 没出界的也写错了地方，落在别的行的 LSE 槽上：`return_softmax_lse=True` 时 27 个预测槽里 19–20 个变成 +inf（看 block 执行先后），
  其余槽不变。
- 上游 Dao-AILab flash-attn 2.8.3.post2 同一行代码、同样的结果（`results/22-fa2-lse-oob-upstream*`）。这是已知 bug：
  上游 issue [Dao-AILab/flash-attention#2842](https://github.com/Dao-AILab/flash-attention/issues/2842)（修复 PR #2573、#2851、#3093），
  vLLM 的 fork 有修复 PR [vllm-project/flash-attention#139](https://github.com/vllm-project/flash-attention/pull/139)，截至 2026-10-30 都没合。

**还丢了一个能力**：Triton 对 head_dim 通用，CUDA 版写死 128，Qwen2.5-0.5B（head_dim 64）当 int8 draft 起不来。
kernel 按 D 模板化（`Geom<D>`：k-step 数 D/32、PV n-tile 数 D/8、每线程通道数 D/8、cp.async 每行 D/16 个 lane、smem 大小），
支持 64 / 128，D=128 的代码路径不变（纯 decode TPOT 4k × 128 仍比 FA2 低 31%）。

**结果**（A100，CUDA graph 开；对照列见 WALKTHROUGH 第 3.8 节）：

| | FA2 bf16 | 只用 CUDA |
|---|---|---|
| GSM8K 1319 题 | 92.19% | 92.12%（+14 / −15） |
| ΔNLL gsm8k / aime24（默认 / 全走 INT8 kernel） | 0 | +0.0001 / +0.0013，+0.0005 / +0.0020 |
| prefill 4k × 64、4k × 128、16k × 16、16k × 32（s） | 15.82、31.94、19.55、39.09 | 15.84、31.96、19.56、39.08 |
| 纯 decode TPOT 4k × 128（ms） | 33.4 | 23.2 |
| 容量受限吞吐（tok/s） | 1112 | 1574（+42%，删 Triton 前 +28%） |
| 投机解码 GSM8K 500 / 接受率（1.5B + 0.5B draft） | 74.8% / 87.0% | 76.2% / 86.6%（验证批走 FA2：75.0% / 86.6%） |
| ngram 7B GSM8K 500 | 92.0% | 92.2% |
| PD D 端 / 混部 TTFT（s） | 6.41 / 5.03 | 5.77 / 4.62 |

- prefill 和 FA2 持平：以前 Triton 反量化 kernel 慢 10–32%，这是 int8 的 TTFT 偏高、吞吐只 +28% 的主要原因。
- 单测 120 个全过（新增：短 query、混合批、仿服务的 256 请求批、CPU 长度偏大、head_dim 64、CUDA graph 下的 q_len 4 验证批）。
- batch 1 投机解码延迟（ms，bf16 / int8）：7B ngram 8.18 / 8.65（删 Triton 前 int8 8.14），1.5B + 0.5B draft 14.03 / 11.69
  （删 Triton 前 int8 18.78）。ngram 的回退来自 kernel：batch 1、上下文 512 时 q_len 5 的 INT8 kernel 33 µs，FA2 25 µs
  （3 个 m-tile 各读一遍 KV + split 合并）；上下文 ≥ 1k 时 INT8 持平或更快（`scripts/bench_short_q.py`）。
  `results/19-cuda-only-perf/latency_sd_int8_int8.json` 是长度修复之前测的，以 `results/19-cuda-only-spec-latency` 为准。

## Step 11 投机验证批的 INT8 kernel（`results/20-*`、`results/21-*`）

起因：只用 CUDA 后，7B ngram 投机解码在 batch 1 下比 bf16 慢 4–6%（Step 10）。

1. **先拆时间**（`scripts/profile_short_q.py`，torch profiler 按 kernel 汇总，batch 1、上下文 512，µs）：

   | | attention kernel | split-KV 合并 kernel | FA2 |
   |---|---|---|---|
   | q_len 1 | 10.2 | 4.3 | 17.6（+ 拷贝） |
   | q_len 5 | 13.9 | 11.9 | 21.5 |

   多出来的时间大部分在合并 kernel：它一个 block 串行处理 5 个 token，每个 token 串行读 16 段，每次读都等上一次。
2. **合并 kernel 改写**：每个 (请求, q head, token) 一个 block；一个 warp 把各段的 max 变成权重放进 smem，D 个线程再用
   互不依赖（展开）的读取累加各段部分和。新旧 kernel 同机 A/B（`scripts/bench_short_q_ab.py`，µs）：

   | | FA2 | 改前 | 改后 |
   |---|---|---|---|
   | batch 1 × 512，q_len 5 | 23.3 | 31.6 | **24.0** |
   | batch 1 × 1k，q_len 5 | 34.7 | 33.2 | **25.2** |
   | batch 1 × 4k，q_len 5 | 109.8 | 37.4 | **29.6** |
   | batch 1 × 512，q_len 1 | 29.2 | 23.8 | **21.4** |
   | batch 8 / 32（不走 split） | – | 不变 | 不变 |

3. **"一个 CTA 处理多个 m-tile、共用 K/V"**（原计划，commit 7c89bde）：4 个 warp 排成 TPC 个 tile × 4/TPC 个 chunk 通道，
   同通道的 warp 一起把 chunk 读进共用的 ring（named barrier 同步），各算各的 tile。单测（TPC = 1/2/4、CUDA graph 回放）
   全过，但在所有点都更慢（`results/20-bench-short-q-ab`）：batch 1 × 512 的 q_len 5 从 24.1 到 34.6 µs，32 × 4k 从 224 到
   383 µs。上下文短时 kernel 受延迟限制：split 16 段后每个 CTA 只有 1 个 chunk，合并 tile 让序言和收尾在一个 CTA 里串行
   做 3–4 遍；省下的重复 K/V 读取本来就命中 L2；长上下文时只剩一个 chunk 通道，访存并行度掉到 1/4。代码删了（8459b09）。
4. **端到端**：`bench_spec_latency.py` 加了接受数和"每步时间"（TPOT 还受接受数影响，int8 的输出和 bf16 不同）。单次测量在
   两台机器上给出了相反的方向（+2.8%、−1.6%），所以在一个容器里 bf16 / 新 int8 / 旧 int8（`KVQ_CUDA_SRC` 指向改前的 .cu）
   交替跑 3 轮（`results/21-ngram-ab`）：

   | 每步时间（ms） | 第 1 轮 | 第 2 轮 | 第 3 轮 | 均值 |
   |---|---|---|---|---|
   | bf16 FA2 | 11.53 | 11.63 | 11.55 | 11.57 |
   | int8，改后 | 11.68 | 11.59 | 11.59 | **11.62（+0.45%）** |
   | int8，改前 | 11.89 | 12.10 | 11.95 | 11.98（+3.6%） |

   每步时间回到和 bf16 持平。TPOT 8.23 / 8.37 / 8.53 ms，剩下的 1.6% 来自 int8 在这 24 道题上每步接受的 token 少 1.4%。
