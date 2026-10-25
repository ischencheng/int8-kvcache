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
