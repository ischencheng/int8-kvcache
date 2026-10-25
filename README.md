# INT8 KV cache for vLLM 0.32 (course project 4)

Static per-channel INT8 KV cache for decode attention on a GPU without FP8 (A100), built
the way the course describes: analyse the KV distribution, write a PyTorch golden
reference, verify INT8 numerics in PyTorch (dynamic quantization), move to static scales
inside vLLM, then write the Triton kernels (fused quantize + cache write, q quantization,
INT8 tensor-core paged attention) and measure accuracy and speed end to end, plus every
extension item (calibration under TP, CUDA graphs, speculative decoding, PD
disaggregation, W8A8/W4A16, MLA).

Results, the reasoning behind each step and the problems hit along the way are in
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md); the step-by-step log is [docs/LOG.md](docs/LOG.md).

## Layout

| path | what |
|---|---|
| `modal_app.py` | runs any command from this repo on a Modal GPU on top of `vllm/vllm-openai:v0.32.0` |
| `patches/vllm-0.32.0-int8-static-kv.patch` | the only vLLM source change: registers `kv_cache_dtype="int8_static"` |
| `kvq/vllm_plugin.py` | vLLM general plugin: registers the CUSTOM backend, installs capture/override hooks |
| `kvq/capture.py` | stage 2: records what vLLM writes into the KV cache (stats, samples, decode dumps) |
| `kvq/quant.py` | symmetric INT8 at per_tensor / per_head / per_channel / per_token / per_token_head |
| `kvq/ref_attn.py` | stage 3: bf16 golden paged attention (`run_paged` does paging, `attend` does math) |
| `kvq/int8_ref.py` | stage 4: implementation A (quant-dequant) and B (INT8 arithmetic) in PyTorch |
| `kvq/override.py` | stage 5: swap vLLM's attention math for the PyTorch versions (dynamic / static bisect) |
| `kvq/scales.py` | static scales: from calibration stats, file format, TP sharding |
| `kvq/vllm_int8/backend.py` | stage 6: the INT8 attention backend (`AttentionBackendEnum.CUSTOM`) |
| `kvq/triton_kernels/` | stage 7: `quant_cache.py`, `quant_q.py`, `int8_attention.py` |
| `kvq/mla.py` | MLA (DeepSeek-V2-Lite) latent-cache analysis, fake-quant hook and references |
| `scripts/` | capture, analysis, calibration, evaluation, benchmarks, extension runs |
| `tests/` | golden-reference, INT8-reference, kernel and CUDA-graph tests |
| `results/<step>/` | raw outputs of every GPU run (JSON, logs, plots) |

## Running

```bash
# any script on a GPU (results land in results/<run-id>/)
modal run modal_app.py --cmd "python3 -m pytest -q tests" --run-id tests --gpu A100-80GB

# static INT8 KV in vLLM
KVQ_SCALES=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors \
KVQ_CFG=B_channel_smooth KVQ_KERNEL=triton \
python3 -c "from vllm import LLM; LLM('Qwen/Qwen2.5-7B-Instruct', kv_cache_dtype='int8_static',
            attention_config={'backend': 'CUSTOM'})"
```
