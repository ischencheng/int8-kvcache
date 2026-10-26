# INT8 KV cache 项目讲解

Qwen2.5-7B-Instruct / vLLM 0.32.0 / A100-80GB（没有 FP8 的卡）上，把 decode 的 KV cache 改成静态 per-channel INT8，
从分析、PyTorch 参考、动态量化、静态量化、Triton 和 CUDA kernel 一路做到端到端，课程列出的扩展项全部做了。
每一步一个 commit（`git log --oneline`），过程记录在 [LOG.md](LOG.md)，原始数据在 `results/`。

## 0. 一页结论

| 项 | 结果 |
|---|---|
| 方案 | K、V 都用静态 **per-channel** scale（`[kv_head, head_dim]`，校准取 max），K 额外做 **smooth-K**（存 k−μ，softmax 平移不变，不用加回）；decode 用 INT8 Tensor Core（s_k 折进 q、q 每 token-head 动态量化、P 按 tile 动态量化），prefill 反量化后算 |
| 精度（GSM8K 1319 题，bf16 92.19%） | 静态 int8 + CUDA kernel（CUDA graph 开）**92.19%**；Triton 版 92.12%；teacher-forcing ΔNLL +0.0004（长回答 +0.0019）。vLLM 原生 int8_per_token_head 91.36%（ΔNLL +0.0017） |
| 显存 | KV 容量 1.03M → **2.07M token**（2.0×） |
| decode attention kernel（A100） | CUDA int8 比 FA2 bf16 快 **1.35–2.06×**（batch 1×32k 到 256×16k），有效带宽 1.36–1.45 TB/s |
| 对比 FP8（H100） | CUDA int8 是 FA3 FP8 的 **0.86–1.05×**（课程"应接近 FP8"成立） |
| 端到端 | 容量受限吞吐 1115 → **1431 tok/s（+28%）**；纯 decode TPOT 比 FA2 低 **13–35%**（4k × 128：36.9 → 24.1 ms；另一台机器重复 3 次 37.7 → 26.3 ms） |
| 扩展 | TP 分布式校准等价；CUDA graph 兼容（batch 1 TPOT 9.9 vs 22.6 ms eager）；投机解码接受率 87.0% → 86.7%、加速比不变；PD 分离下 int8 页正确传递、传输量减半；W8A8 / W4A16 叠加不额外掉点；MLA latent 用 per-channel 不需要 smooth |

## 1. 整体结构：改了 vLLM 的哪里

```
vLLM 0.32.0 进程（API server / engine core / 每个 TP worker）
 └─ 启动时加载 general plugin  kvq/vllm_plugin.py:register()
      ├─ register_backend(CUSTOM → kvq.vllm_int8.backend.Int8StaticBackend)   ← 阶段六
      ├─ KVQ_CAPTURE=1 : kvq/capture.py 包住 FlashAttention 的写 cache / forward  ← 阶段二
      ├─ KVQ_ATTN=...  : kvq/override.py 把 FA2 的 attention 换成 PyTorch 版     ← 阶段三/五/六二分
      └─ KVQ_MLA=...   : kvq/mla.py 包住 MLA 的写 cache / decode                 ← MLA 扩展

patches/vllm-0.32.0-int8-static-kv.patch（构建镜像时打到已安装的 vLLM 上）
 ├─ config/cache.py        CacheDType += "int8_static"      （pydantic Literal 校验，只能改源码）
 ├─ utils/torch_utils.py   "int8_static" → torch.int8       （KV cache 按 1 字节分配，block 数翻倍）
 └─ v1/kv_cache_interface  KVQuantMode.INT8_STATIC          （is_quantized_kv_cache() 认它）
```

一个 token 的 KV 在 INT8 backend 里的路径：

```
prefill / decode 新 token:  qkv_proj → RoPE → k, v (bf16)
   └─ do_kv_cache_update → quant_reshape_and_cache  (k-μ)/s_k, v/s_v → rint → int8 → 按 slot_mapping 写页
attention:
   ├─ 多 token 请求（prefill、chunked prefill、投机验证）: MODE 1  int8 页 → ×s → bf16 → bf16 MMA
   └─ 纯 decode 批:                                        MODE 2  q·s_k → 量化 q → int8 MMA(QK)
                                                                   → P 按 tile 量化 → int8 MMA(PV) → ×s_v
```

运行方式：`kv_cache_dtype="int8_static", attention_config={"backend": "CUSTOM"}`，再用环境变量
`KVQ_SCALES`（校准文件）、`KVQ_CFG`（数值配置名，见 `kvq/int8_ref.py:CONFIGS`）、`KVQ_KERNEL`（torch / triton / cuda）。
用插件 + 小 patch 而不是 fork vLLM：改动面只有 3 个文件 12 行，其余都在我们的包里，换 vLLM 版本时只需要重做这个 patch。

## 2. 代码导览（按课程的阶段）

### 2.1 运行环境：`modal_app.py`、`patches/`

- 镜像 = 官方 `vllm/vllm-openai:v0.32.0` + 打 patch + 一个"插件桩"：
  `patches/install_plugin_stub.sh` 在 site-packages 里写一个只有 `entry_points.txt` 的 dist-info，
  再用 `.pth` 把挂载进来的 `/root/repo` 放进 `sys.path`。这样本地改代码不用重建镜像，vLLM 的子进程也能通过
  entry point 找到 `kvq.vllm_plugin:register`。
- 每次运行把 `$OUT` 打包传回 `results/<run-id>/`，命令失败或超时也会传回日志。

### 2.2 阶段二 采集：`kvq/capture.py`、`scripts/capture_kv.py`、`scripts/analyze_kv.py`

- 插桩点选 `FlashAttentionImpl.do_kv_cache_update`：这是 K/V 写进 cache 的唯一入口，prefill 和 decode 都经过它，
  拿到的正是要存的东西（RoPE 之后的 K）。注意 key/value 带 padding、`slot_mapping` 不带，先按 `slot_mapping` 截断再用 `>=0` 过滤。
- 统计量：每 (kv_head, channel) 的 |x| 最大值、有符号 max/min、sum（后面 smooth-K 要用均值和 max|k-μ|），
  外加抽样 16k 个 token 的原始行（算分位数、画图）。
- decode dump 只存用到的页并重编号 block_table，存成 safetensors（不用 pickle）。
- `analyze_kv.py` 对每层算 5 种粒度的往返误差，以及"softmax 看得见的误差"（K 减去 token 均值之后）。

### 2.3 阶段三 金标准：`kvq/ref_attn.py`、`tests/test_ref_attn.py`

- `run_paged(attend, ...)` 只管分页：gather 页 → padding → mask（右下角对齐的 causal）；decode 请求一起向量化，
  prefill 请求按 512 行分块。`attend(qb, k, v, mask, valid)` 只看稠密张量。
- `attend_bf16` 用分组 einsum 做 GQA（q 视为 `[Hkv, G, D]`），不复制 K/V。
- 三级验证：dump 逐位复现 → 金标准对 dump / 随机分页 → 整网替换跑 GSM8K。

### 2.4 阶段四 两类 int8 实现：`kvq/quant.py`、`kvq/int8_ref.py`

- `quant.py`：对称量化到 [-127, 127]，5 种粒度一个 `_REDUCE_DIMS` 表搞定；`absmax(..., quantile=)` 支持分位数。
- `int8_ref.py`：
  - `quantize_kv`：动态（每个请求自己的 K/V，先按 valid mask）或静态（校准文件）scale；smooth-K 先减 μ。
  - `attend_dequant`（实现 A）：反量化成 bf16 再走金标准。
  - `attend_int8`（实现 B）：per_channel 时把 s_k 折进 q（`ks.shape[-1] > 1` 判断），否则 s_k 乘在分数上；
    q 量化支持 none / tensor / token_head / group32；P 量化支持 none / static(1/127) / tile；整数点积用 float64 精确计算。
  - `CONFIGS`：所有实验用到的命名配置，测试、报告、vLLM 覆盖都按名字引用。
  - `paged_static_attention`：静态 scale 版，decode 行用 B，多 token 行用 A；`cache_is_int8=False` 时就是二分工具。

### 2.5 阶段五 动态量化：`kvq/override.py`、`scripts/run_eval.py`、`scripts/eval_nll.py`

- `override.py` 只替换 `FlashAttentionImpl.forward` 里的 attention 计算，vLLM 仍然写 bf16 cache；
  `KVQ_ATTN=<配置名>` 时在读 cache 时现场量化（动态）。
- 评测两种指标：GSM8K 全量 1319 题准确率（逐题对比翻转数），以及 teacher-forcing 的 ΔNLL：
  用 bf16 模型自己的 greedy 回答当固定文本，算各配置下回答部分的平均 NLL。后者是连续量，噪声小得多。

### 2.6 阶段六 静态量化：`kvq/scales.py`、`scripts/calibrate.py`、`kvq/vllm_int8/backend.py`

- `calibrate.py` 在 FA2 上开采集跑校准集（GSM8K train / wikitext），一次产出 6 种 scale 文件；TP>1 时
  每个 rank 只看到自己的 kv head，`scales.merge_tp` 按 rank 顺序拼接（kv head 少于 rank 时去重复制的 head）。
- `backend.py`：
  - `Int8StaticBackend(TritonAttentionBackend)`：`get_name()` 返回 `"CUSTOM"`，`supported_kv_cache_dtypes` 加 `int8_static`。
  - `Int8StaticBuilder.__init__`：把每层的静态 scale（按 TP rank 切片）交给 `layer.impl.set_static()`。
    它在模型加载之后、CUDA graph capture 之前执行，并且拿得到层名。投机解码的 draft 层（`draft_model.` 前缀）
    从 `KVQ_DRAFT_SCALES` 取。
  - `Int8StaticImpl.do_kv_cache_update / forward`：torch 路径（阶段六）/ triton 路径 / cuda 路径（阶段七）三选一；
    `kv_cache_dtype=auto` 时跑 bf16（triton 路径就是同一份 kernel 的 bf16 模式，用作公平基线）。
    cuda 路径：写 cache 和纯 decode 批走 CUDA kernel，多 token 批（prefill / 混合批）仍走 Triton 的反量化 kernel。

### 2.7 阶段七 kernel：`kvq/triton_kernels/`（Triton）和 `kvq/cuda/int8_kv.cu`（CUDA）

**Triton 版**

- `quant_cache.py`：一个 program 一个 token × 全部 kv head；`div_rn` + `rint` 保证与 PyTorch 逐位一致。
- `quant_q.py`：独立的 q 折叠 + 量化 kernel（attention kernel 序言里也能融合做，两种都测了速度）。
- `int8_attention.py`：
  - `_find_seq`：每个请求的 q block 从 `cu_q[i] // BLOCK_Q + i` 开始编号（上界打包，避免在 host 上算真实 block 数），
    program 二分查找自己属于哪个请求。
  - 行 = token × head（GQA 的 G 个 head 共享一个 K/V tile）；KV 按 64 token 一个 tile，按 block_table 逐 token 取物理块。
  - MODE 2 的数学和 `attend_int8` 一一对应：`_quant_rows` 量化 q；`tl.dot(int8, int8, out_dtype=int32)`；
    online softmax 用 exp2（softmax_scale 预乘 log2e）；P 量化 `rint(P / (rowmax/127))`；分母 l 用量化后的 P 求和。
  - split-KV：纯 decode 且请求数 ≤ 128/Hkv 时 16 段，`_reduce_kernel` 按 `exp2(m_s - max)` 加权合并。

**CUDA 版**（`kvq/cuda_kernels.py` 用 `torch.utils.cpp_extension.load` JIT 编译，`KVQ_KERNEL=cuda`）
- `quant_cache_kernel`：一个 block 一个 token，每线程 8 个通道，16 B 读 / 8 B 写，`__fdiv_rn` + `__float2int_rn`。
- `quant_q_kernel`：一个 warp 一个 (token, q head)，shuffle 求 amax。
- `int8_decode_attn_kernel`（v3）—— 建议按这个顺序读：
  1. grid = (请求, kv head, split)，4 个 warp；m16 的 16 行 = 这个 kv head 的 G 个 q head。
  2. `load_pages` / `issue`：一个 32-token chunk 最多跨 2 页，只查 2 次 block_table（提前一轮取），
     `cp.async` 把 K/V 原样搬进每个 warp 自己的 2 级 smem 环（行距 144 B，B fragment 读取无 bank 冲突）。
  3. 序言：8 个线程一行，s_k 折进 q、行 amax、量化，写 smem，每个 warp 读出 4 个 k-step 的 A fragment 常驻寄存器。
  4. QK：`mma.sync.m16n8k32.s8`，K 的 B fragment **按 `slot_of_token(8j+g)` 置换后的行号**去读。
  5. softmax + P 量化：同一 quad 的 4 个线程持有同一行，`__shfl_xor` 求行最大值；P 按 32-token chunk 量化成 int8，
     直接 `pack4` 进 PV 的 A fragment（置换保证了 P 不用跨线程搬）。
  6. PV：V 保持原顺序，每线程 4 条 32 位 LDS 读 4 个 token × 4 个通道，8 条 `__byte_perm` 转置成 4 个 n-tile 的 B fragment；
     输出通道因此是置换的（`chan_of`）。
  7. 合并：每个 warp 写自己的 smem 片（行距 129 个 float），收尾时 4 份相加，乘 s_v，写 bf16；切分时写部分和，
     `split_reduce_kernel` 合并。**不用 smem 原子加**（sm_80 上是 CAS 循环）。

### 2.8 扩展

- CUDA graph：`tests/test_cudagraph.py`（capture/replay、scale 必须原地更新、padding slot）。
- 投机解码：`scripts/ext_specdec.sh`（draft_model，draft KV 用 `speculative_config.kv_cache_dtype` 隔离）。
- TP 校准：`scripts/ext_tp.sh` + `scripts/compare_scales.py`。
- PD 分离：`scripts/ext_pd.sh` + `scripts/pd_offline.py`（两个独立 engine + ExampleConnector）；
  `scripts/pd_proxy.py` / `scripts/eval_api.py` 是在线方案（NixlConnector 在 gVisor 里起不来，见第 4 节）。
- W8A8 / W4A16：`scripts/ext_wquant.sh`。
- MLA：`kvq/mla.py` + `scripts/mla_capture.py` + `scripts/mla_analyze.py`。

## 3. 各阶段的结果和关键决策

### 3.1 阶段二：为什么是 per-channel + smooth-K

- K 有很强的**通道**离群：每个 head 的 52–63、116–127 通道（neox RoPE 的低频维）在几乎所有 token 上都是同一个大值
  （第 27 层 head 1 约 −400），来源是 Qwen2 的 qkv bias + 低频维几乎不旋转。V 没有通道结构。
- 往返误差：K per-channel 0.61%（per-head 1.90%，per-token-head 1.29%）；V per-channel 1.06%（静态里最好）。
- **关键认识**：K 里跨 token 不变的分量对 softmax 不可见（`softmax(q·(k+μ)) = softmax(q·k + q·μ)`）。只看可见部分 `k−μ`，
  per-token-head 平均 5.0%（第 27 层 47%），per-channel 1.53%，**per-channel + smooth-K 0.90%**。
- scale 取 max 而不是分位数（阶段二往返误差 + 阶段六端到端都证实）。

### 3.2 阶段四：两类实现与误差拆分（真实 decode dump 上 attention 输出误差）

| | A（只量化 KV） | B（真 int8 计算） |
|---|---|---|
| per_head | 12.27% | 12.15% |
| per_token_head | 11.10% | – |
| per_channel | 2.47% | **8.29%** |
| per_channel + smooth-K | **1.06%** | 2.21%（q 每 32 通道一个 scale：1.71%） |

- 公因式判据：scale 依赖被缩并的维度就提不出去。QK 缩并 head_dim → per-channel 的 s_k 折进 q；PV 缩并 token → s_v 乘在输出上。
- B 不加 smooth-K 差 3 倍：s_k 折进 q 后 q 的通道范围被离群通道撑大；smooth-K 去掉常数偏移后恢复。
- P 不能用固定 1/127（注意力分散时小概率全舍成 0），要按 tile 取最大值。q 不能 per-tensor（37%）。

### 3.3 阶段五 / 六：端到端精度（GSM8K 1319 题 + ΔNLL）

| 配置 | GSM8K | ΔNLL gsm8k / aime24 回答 |
|---|---|---|
| bf16 | 92.19% | 0 / 0 |
| 动态 A per_channel + smooth | 91.58% | +0.0001 / +0.0003 |
| 动态 B per_channel（无 smooth） | **63.08%** | +0.0221 / +0.0161 |
| 动态 B per_channel + smooth + group32 q | 92.04% | +0.0002 / +0.0003 |
| 静态 per_channel max + smooth（主方案） | 91.89% | +0.0004 / +0.0019（decode 路径） |
| 静态 99.9% 分位截断 | **78.24%** | +0.11 / +0.13 |
| 静态 文本校准 → 数学评测 | 91.58% | −0.0006 / +0.0000 |
| 静态 per_head + smooth | 92.27% | −0.0002 / +0.0006 |

- 准确率每个配置都有 ±30–60 题的翻转噪声；排序看 ΔNLL（per_token_head 准确率 92.72% 但 ΔNLL 和掉 3 点的 per_head 一样差）。
- 框架改动的正确性：真 int8 存储和"bf16 cache + 读时按同一套 scale 量化"（课程的二分工具）50 题输出逐字一致。

### 3.4 阶段七：kernel

- 三个 kernel 都有 Triton 版和 CUDA 版；写 cache、q 量化与 PyTorch 逐位一致，attention 与 PyTorch 实现 B 差 1e-5。
- 融合写 cache（8192 token）28 µs，比 vLLM 原生 bf16 写入 42 µs 还快；q 量化融合进 attention 序言，额外开销为 0。
- CUDA decode kernel 三轮 ncu 优化（b32×32k）：1514 → 1173 → **743 µs**（Triton int8 892，FA2 1402）。

### 3.5 A100 / H100 kernel 对比（µs）

| batch × ctx | FA2 bf16 | 我们 Triton int8 | 我们 CUDA int8 | H100: FA3 FP8 | H100: 我们 CUDA int8 |
|---|---|---|---|---|---|
| 1 × 32k | 89.6 | 107.3 | 66.6 | 39.2 | 45.4 |
| 32 × 32k | 1322 | 855 | 703 | 432 | 426 |
| 128 × 32k | 5584 | 3696 | 2709 | 1643 | 1761 |

### 3.6 阶段八：端到端

- 容量：1.03M → 2.07M token。
- 容量受限吞吐（512 × 6k+1k）：FA2 1115、Triton int8 1361、**CUDA int8 1431 tok/s**、vLLM int8_per_token_head 676。
- 纯 decode TPOT（ms）：同一批请求分别生成 L = 128 和 2L = 256 个 token，两次的 prefill 完全一样，
  多出来的 L 步都在所有 prefill 结束之后，`(T(2L) − T(L)) / L` 就是纯 decode 一步的时间。

  | ctx × batch | FA2 bf16 | 我们 Triton bf16 | 我们 Triton int8 | 我们 CUDA int8 | vLLM int8 原生 |
  |---|---|---|---|---|---|
  | 1k × 128 † | 21.20 | 19.98 | 19.14 | **18.33** | 28.03 |
  | 4k × 64 | 27.09 | 23.71 | 21.39 | **20.57** | 43.58 |
  | 4k × 128 | 36.88 | 32.87 | 28.37 | **24.07** | 58.09 |
  | 16k × 16 | 20.72 | 19.92 | 16.88 | **15.85** | 35.47 |
  | 16k × 32 | 30.37 | 29.92 | 23.80 | **21.56** | 57.10 |

  † 每个配置测的第一个点。单次测量时 Triton 的首次 JIT 编译落进了 T(L)，TPOT 被算小（Triton int8 算成 15.5 ms），
  这一行换成另一台 A100 上"每点先预热、重复 3 次取中位数"的结果；vLLM 原生一列仍是单次值。

  CUDA int8 比 FA2 快 13–35%，同一份 Triton 代码 bf16 → int8 快 4–20%，KV 总量（ctx × batch）越大收益越大。
  重复测量那台机器上 4k × 128 是 FA2 37.7 / bf16 33.8 / Triton int8 30.5 / CUDA int8 26.3 ms（CUDA 三次 24.0–26.6），
  CUDA 比 FA2 快 30%，单次表里是 35%，机器之间差几个百分点。
  vLLM 原生 int8_per_token_head 每步比 FA2 慢 58–88%（后四个点），远超 kernel 测试里的差距（128 × 4k 它的 kernel 甚至比 FA2 快），
  慢在 attention kernel 之外，没追（见第 6 节）。
- TPOT 先后用了三种口径，只有纯 decode 这一种把 decode 单独拎了出来（第 4 节第 17 条）：差值法混进了 prefill chunk；
  engine 的每请求时间戳在请求同时涌入时由 prefill 主导，int8 反而看不出收益。
- CUDA graph：batch 1 时 eager 22.6 ms vs graph 9.9 ms。

### 3.7 扩展

| 扩展 | 结论 |
|---|---|
| 3.5(a) 校准参数 | max 不要分位数；文本校准可用于数学任务；TP=2 下单卡 scale 按 rank 切 92.42%、分布式校准 92.72% |
| 3.5(b) CUDA graph | 完全兼容（精度不变）。要点：scale 早于 capture 分配且只能原地改；kernel 无 host 同步、分支只依赖 capture 时固定的形状；padding slot −1；扩展提前编译 |
| 3.5(c) 投机解码 | draft 的 KV 用 `speculative_config.kv_cache_dtype/attention_backend` 隔离；接受率 87.0% → 86.7%（draft 量化与否一样）；7B ngram 加速比 1.26× → 1.23× |
| 3.5(d) PD | int8 页正确传递（D 端 92.6% = bf16 = 混部）；传输量 53.0 → 26.5 KB/token；D 端 TPOT −8%；静态 scale 不需要随 KV 传 |
| 3.5(e) W8A8 / W4A16 | W8A8 92.34 → 92.19%，AWQ 90.98 → 90.90%；KV 量化作用在 bf16 的 K/V 投影输出上，与权重/激活量化解耦；vLLM 0.32 在 GPU 上没有真 W4A8-int 路径 |
| MLA | latent 同时是 K 和 V：QK 折进 q_lat，PV 乘在 o_lat 上，smooth 的 μ 在 PV 角色必须加回；latent 通道温和，per-channel 不加 smooth 就够，GSM8K 72.33 → 71.3–72.0% |

## 4. 遇到的问题（按类型）

**数值 / 算法**
1. "往返误差"对 K 有误导：K 的离群通道几乎是常数，对 softmax 不可见；换成"可见部分"的误差后才看出 per-token-head 很差、smooth-K 很有用。
2. 实现 B 不加 smooth-K 端到端崩到 63%：per-channel s_k 折进 q 后，q 的量化被离群通道的大 scale 主导。
3. P 用固定 scale 1/127 会把注意力分散时的小概率全舍成 0；改成每个 tile 按行最大值量化。
4. q 用 per-tensor（课程正文写法）误差 37%：一个 batch 里最大的 q 决定所有人的 scale。
5. 课程建议 99.9% 分位截断，实测掉到 78%：被截掉的是 attention sink 之类的极端 token。
6. 准确率噪声大，±1 点是噪声；引入 teacher-forcing ΔNLL 才能稳定排序。
7. 静态 backend 下 ΔNLL 只测到 prefill（反量化）路径；要让所有行走 int8 路径才测到 decode 的数值。MLA 的 ΔNLL 完全测不到量化（prefill 用新算的 latent）。
8. 用 dump 的整页算 scale：最后一页未写满的 slot 里有旧数据，第 27 层误差 14.5%；只用有效 token 后 3.3%。动态 scale 也要先按 valid 掩掉 padding 行。

**vLLM 框架**
9. KV 布局和课程文档都不一样：0.32 每层是 `[blocks, kv_heads, block, 2*D]`（K|V 拼在最后一维），物理布局 LBNHC。kernel 一律按 stride 访问。
10. `CacheDType` 是 pydantic Literal，插件改不了，只能打源码 patch；自定义 backend 的 `get_name()` 必须返回 `"CUSTOM"`。
11. 静态 scale 交给每层的时机：impl 构造时拿不到层名，forward 里懒加载会撞上 graph capture；放在 metadata builder 的 `__init__`。
12. 我们模块的 logger 不打印（vLLM 只给 `vllm.*` 配了 handler）。
13. Python 钩子只在 eager 下生效（full CUDA graph 回放不回 Python）。
14. 离线 `LLM` 默认关统计，`get_metrics()` 拿不到投机解码接受率。
15. draft_model 要求词表一致（7B 152064 vs 0.5B 151936）。
16. RedHatAI w4a16 的 `actorder='group'` 被 0.32 拒绝，换官方 AWQ。
17. TPOT 的口径：chunked prefill 下"总时间 − prefill 时间"混进了 prefill chunk；engine 每请求时间戳在请求同时涌入时由 prefill 主导；
    最后用两个输出长度做差测纯 decode。每个配置第一个点的 T(L) 含 Triton JIT 编译，同一容器里编译缓存还会被下一个配置复用，
    偏差大小因配置而异（0.4–3.6 ms），要每个点先预热。
18. ExampleConnector 遇到 chunked prefill 只保存最后一段 KV。

**Kernel**
19. Triton 的 `/` 是近似除法，量化要 `tl.div_rn`；取整用 `rint`。int8 `tl.dot` 要求 K 维 ≥ 32。
20. CUDA graph 记录张量地址，scale 换对象不生效，只能原地改（有单测演示）。
21. CUDA v1：访存延迟暴露、V 逐字节转置、局部数组进 local memory → v2 cp.async + 置换挪到 K + `__byte_perm`。
22. CUDA v2：切分模式下 smem `atomicAdd(float)` 是 CAS 循环，占 75% 的 stall → v3 每 warp 独立 smem 片再相加；
    运行时 block_size 的除法取模 → 移位、每 chunk 两次页表查询。
23. `ATen/cuda/CUDAContext.h` 拉 cuSPARSE 头文件（镜像里没有）；镜像预设 7 个架构，JIT 要改成只编当前卡。

**环境 / 工程**
24. vLLM 官方镜像只有 `python3`，Modal 要找 `python`；`run_commands` 不能传换行开头的脚本。
25. PyTorch 参考实现整网跑时 OOM：prefill 按行分块、GQA 用分组 einsum 不复制 K/V。
26. 8 个 PyTorch 动态配置串行超过 3 小时时限，改成先生成参考答案存 Volume 再 3 台并行。
27. ncu 偶发 "driver resource unavailable"（宿主机 DCGM 占计数器），换容器重跑。
28. gVisor 没有 `/sys/class/net`，UCX / NIXL 起不来；自写代理在高并发下 connection reset；最后用两个离线 engine。
29. Modal 中途抢占容器，任务整段重跑（顺便验证了结果可重复）。
30. zsh 下一个无匹配的 glob 让整条 `rm` 没执行，重复结果被提交了，下一个 commit 删掉。

## 5. 复盘要点（面试可以这样讲）

- **为什么 per-channel**：先看分布，K 的离群在通道维，而且跨 token 不变。per-token 的 scale 会被离群通道吃掉，per-channel 正好对齐；
  静态方案要求 scale 与 token 无关，per-token 本来也做不了静态。
- **per-channel 的 int8 QK 怎么做**：scale 在点积维上变，提不出去，把 s_k 乘进 q 再量化 q；PV 的 s_v 沿通道、和 token 无关，乘在输出上。
- **smooth-K 是什么、为什么免费**：存 k−μ，q·μ 对同一 query 的所有 key 是常数，softmax 不变，attention 里什么都不用做；
  在 Qwen2.5 上它是 int8 QK 能用的前提。MLA 的 latent 还要当 V，μ 就必须在输出上加回。
- **怎么证明正确**：PyTorch 金标准三级验证 → 实现 A/B 拆误差 → 动态量化端到端 → 静态量化和"二分工具"逐字一致 → kernel 对 PyTorch 1e-5。
- **性能从哪来**：decode attention 访存减半 + int8 MMA；上下文越长、batch 越大越明显；容量翻倍带来更高并发（吞吐 +28%）。
- **CUDA kernel 的难点**：P 的寄存器布局（QK 的 C fragment ≠ PV 的 A fragment），用 token 置换解决；int8 B 操作数没有 ldmatrix.trans，
  用 `__byte_perm` 做 4×4 字节转置；用 ncu 一轮轮找瓶颈（访存延迟 → smem 原子 → 整数指令）。

## 6. 局限和没做完的

- **prefill 比 FA2 慢**：int8 backend 的 prefill 走自己的反量化 Triton kernel（16k × 16：23 s vs 19 s），TTFT 因此高约 20%。
  可以改成反量化到 bf16 临时页后调 FA2，或继续优化 MODE 1。
- CUDA kernel 只做了 decode、head_dim 128；Hopper 上没用 wgmma/TMA；76 KB smem 限制了占用率；group32 的 q scale 只在 Triton 版里有。
- PD 分离只能用存储型 connector（gVisor 限制），测不到真实网络传输下 TTFT 的变化。
- vLLM 原生 int8_per_token_head 端到端 decode 比它的 kernel 测试慢得多（4k × 128 每步比 FA2 多 21 ms，kernel 本身才 0.6 ms/层），
  原因没查；它只是对照组。
- AIME24 对 Qwen2.5-7B-Instruct 太难（13.3%），没法用准确率比较；改用它在 AIME 上的长回答做 ΔNLL。GPQA 是 gated 数据集，没用。
- 这是课程项目、不是对外贡献，所以没走仓库里贡献流程要求的 RLCR / Codex review。

## 7. 复现

```bash
# 单测（参考实现 / int8 实现 / Triton / CUDA / CUDA graph）
modal run modal_app.py --cmd "python3 -m pytest -q tests" --run-id tests --gpu A100-80GB
# 各阶段的完整脚本
modal run modal_app.py --cmd "bash scripts/stage5_dynamic.sh A_channel_smooth B_channel_smooth" --run-id s5 --gpu A100-80GB
modal run modal_app.py --cmd "bash scripts/stage6_static.sh" --run-id s6 --gpu A100-80GB
modal run modal_app.py --cmd "cd scripts && python3 bench_kernels.py" --run-id bench --gpu A100-80GB
modal run modal_app.py --cmd "bash scripts/stage8_e2e.sh" --run-id s8 --gpu A100-80GB
```
