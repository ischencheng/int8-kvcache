# INT8 KV cache 项目讲解

<!-- RESULTS_SUMMARY -->

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
`KVQ_SCALES`（校准文件）、`KVQ_CFG`（数值配置名，见 `kvq/int8_ref.py:CONFIGS`）、`KVQ_KERNEL`（torch / triton）。
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
  - `Int8StaticImpl.do_kv_cache_update / forward`：torch 路径（阶段六）和 triton 路径（阶段七）二选一；
    `kv_cache_dtype=auto` 时跑 bf16（triton 路径就是同一份 kernel 的 bf16 模式，用作公平基线）。

### 2.7 阶段七 kernel：`kvq/triton_kernels/`

- `quant_cache.py`：一个 program 一个 token × 全部 kv head；`div_rn` + `rint` 保证与 PyTorch 逐位一致。
- `quant_q.py`：独立的 q 折叠 + 量化 kernel（attention kernel 序言里也能融合做，两种都测了速度）。
- `int8_attention.py`：
  - `_find_seq`：每个请求的 q block 从 `cu_q[i] // BLOCK_Q + i` 开始编号（上界打包，避免在 host 上算真实 block 数），
    program 二分查找自己属于哪个请求。
  - 行 = token × head（GQA 的 G 个 head 共享一个 K/V tile）；KV 按 64 token 一个 tile，按 block_table 逐 token 取物理块。
  - MODE 2 的数学和 `attend_int8` 一一对应：`_quant_rows` 量化 q；`tl.dot(int8, int8, out_dtype=int32)`；
    online softmax 用 exp2（softmax_scale 预乘 log2e）；P 量化 `rint(P / (rowmax/127))`；分母 l 用量化后的 P 求和。
  - split-KV：纯 decode 且请求数 ≤ 128/Hkv 时 16 段，`_reduce_kernel` 按 `exp2(m_s - max)` 加权合并。

### 2.8 扩展

- CUDA graph：`tests/test_cudagraph.py`（capture/replay、scale 必须原地更新、padding slot）。
- 投机解码：`scripts/ext_specdec.sh`（draft_model，draft KV 用 `speculative_config.kv_cache_dtype` 隔离）。
- TP 校准：`scripts/ext_tp.sh` + `scripts/compare_scales.py`。
- PD 分离：`scripts/ext_pd.sh` + `scripts/pd_proxy.py`（NixlConnector）+ `scripts/eval_api.py`。
- W8A8 / W4A16：`scripts/ext_wquant.sh`。
- MLA：`kvq/mla.py` + `scripts/mla_capture.py` + `scripts/mla_analyze.py`。

<!-- RESULTS_AND_PROBLEMS -->
