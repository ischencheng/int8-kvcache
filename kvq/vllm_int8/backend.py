"""Static INT8 KV cache backend for vLLM 0.32.0 (registered as AttentionBackendEnum.CUSTOM).

Run with  kv_cache_dtype="int8_static", attention_config={"backend": "CUSTOM"}  and
  KVQ_SCALES  calibrated scales (safetensors, see kvq/scales.py)
  KVQ_CFG     numerics of the INT8 path, a name from kvq.int8_ref.CONFIGS (default B_channel_smooth)
  KVQ_KERNEL  "cuda" (default, kvq/cuda_kernels.py) or "torch" (the stage-6 PyTorch path)

What changes compared with vLLM's Triton backend, whose metadata it reuses (the stage-6 list):
  (a) the cache is allocated as int8 (patch: CacheDType "int8_static" -> torch.int8), so the
      same memory holds twice the blocks;
  (b)/(e) do_kv_cache_update quantizes the new K/V with the static scales before writing,
      for prefill and decode alike (K first has the smooth-K mean removed);
  (c) long queries (prefill, chunked prefill) read the int8 cache dequantized: the CUDA path
      copies their pages into a bf16 paged cache and runs vLLM's FlashAttention on it;
  (d) short queries run INT8 arithmetic: q is quantized dynamically, K/V scales are static.
      "Short" is up to cuda_kernels.SHORT_Q query tokens on the CUDA path (decode and
      spec-decode verification), one token on the torch path.
"""
import os
from typing import ClassVar

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)

from kvq import cuda_kernels, quant, scales
from kvq.int8_ref import CONFIGS, paged_static_attention

logger = init_logger("vllm.kvq")  # under "vllm" so vLLM's log handler prints it
_SCALES_CACHE = {}


def _kernel():
    return os.environ.get("KVQ_KERNEL", "cuda")


def _load_scales(path):
    if path not in _SCALES_CACHE:
        _SCALES_CACHE[path] = scales.load(path)
    return _SCALES_CACHE[path]


class Int8StaticBackend(TritonAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list] = ["int8_static"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return Int8StaticImpl

    @staticmethod
    def get_builder_cls():
        return Int8StaticBuilder

    @classmethod
    def customize_spec(cls, spec):
        return spec  # scales live in the backend, not inside the KV pages


class Int8StaticBuilder(TritonAttentionMetadataBuilder):
    """Metadata as for the Triton backend (lengths, block table, split-KV scratch) plus, for a
    batch with long queries, the page table of the temporary bf16 cache (cuda_kernels.plan_long),
    computed once per step for all layers. Built once per KV-cache group after the model is
    loaded and before CUDA graphs are captured, which makes it the place to hand every layer its
    static scales (persistent tensors that a captured graph can keep pointing at)."""

    # A draft-decode step reusing an earlier metadata object would also reuse its long-query plan.
    supports_draft_decode_metadata_update = False

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if _kernel() != "cuda":
            return super().get_cudagraph_support(vllm_config, kv_cache_spec)
        # Full graphs only for uniform batches the INT8 kernel covers alone: the long-query path
        # sizes its buffers from host-side lengths. Mixed batches run piecewise (attention eager).
        spec = vllm_config.speculative_config
        q_len = 1 + (spec.num_speculative_tokens if spec is not None else 0)
        if q_len <= cuda_kernels.SHORT_Q:
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        all_scales = _load_scales(os.environ["KVQ_SCALES"])
        ctx = vllm_config.compilation_config.static_forward_context
        rank, size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        for name in layer_names:
            src, key = all_scales, name
            if name not in all_scales and os.environ.get("KVQ_DRAFT_SCALES"):
                # Spec-decode draft model (prefix "draft_model."): its own calibration file.
                src, key = _load_scales(os.environ["KVQ_DRAFT_SCALES"]), name[name.index("model.layers"):]
            ctx[name].impl.set_static(scales.for_layer(src, key, rank, size, device))
        logger.info("kvq: static INT8 scales for %d layers from %s (tp %d/%d, cfg %s, kernel %s)",
                    len(layer_names), os.environ["KVQ_SCALES"], rank, size,
                    os.environ.get("KVQ_CFG", "B_channel_smooth"), _kernel())

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        md.kvq_long = None
        c = common_attn_metadata
        if _kernel() == "cuda" and c.max_query_len > cuda_kernels.SHORT_Q:
            # Exact on long-query rows, which are the only ones the plan reads.
            seq_cpu = c.seq_lens_cpu_upper_bound if c.seq_lens_cpu_upper_bound is not None else c.seq_lens.cpu()
            md.kvq_long = cuda_kernels.plan_long(c.query_start_loc_cpu, seq_cpu, self.block_size)
        return md


class Int8StaticImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kv_cache_dtype == "int8_static", self.kv_cache_dtype
        self.cfg = CONFIGS[os.environ.get("KVQ_CFG", "B_channel_smooth")]
        self.kernel = _kernel()
        self.static = None
        if self.kernel == "cuda":
            c = self.cfg
            assert c.impl == "B" and c.q_gran == "token_head" and c.p_quant == "tile" and c.p_norm == "quant", (
                f"the CUDA kernel implements B with per-(token, head) q and per-tile P, not {c}")
            cuda_kernels.ext()  # JIT-compile at model build time, long before any CUDA graph capture
        else:
            assert self.kernel == "torch", self.kernel

    def fused_output_quant_supported(self, quant_key):
        return False  # our kernels write bf16 output only

    def set_static(self, static):
        assert static["k_scale"].shape[0] == self.num_kv_heads, (static["k_scale"].shape, self.num_kv_heads)
        assert ("k_mu" in static) == self.cfg.smooth_k, "smooth-K config and scales file disagree"
        static = {k: v.float().contiguous() for k, v in static.items()}
        if "k_mu" not in static:
            static["k_mu"] = torch.zeros_like(static["k_scale"].expand(-1, self.head_size)).contiguous()
        self.static = static
        self._torch_attn = paged_static_attention(self.cfg, static, cache_is_int8=True)

    def _caches(self, kv_cache):
        # Logical [num_blocks, H, block, 2*D] -> two [num_blocks, block, H, D] views (K | V halves).
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        key_cache, value_cache = self._caches(kv_cache)
        if self.kernel == "cuda":
            return cuda_kernels.quant_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                                                        self.static)
        # Stage-6 PyTorch path: quantize, then scatter with the slot mapping.
        n = slot_mapping.shape[0]
        slots = slot_mapping.long()
        valid = slots >= 0
        s = self.static
        kq = quant.quantize(key[:n].float() - s["k_mu"], s["k_scale"])
        vq = quant.quantize(value[:n].float(), s["v_scale"])
        bs = key_cache.shape[1]
        blk, off = slots[valid] // bs, slots[valid] % bs
        key_cache[blk, off] = kq[valid]
        value_cache[blk, off] = vq[valid]

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)  # profiling run
        md = attn_metadata
        n = md.num_actual_tokens
        num_reqs = md.query_start_loc.shape[0] - 1
        key_cache, value_cache = self._caches(kv_cache)
        if self.kernel == "cuda":
            segm = None
            if num_reqs <= md.seq_threshold_3D and n <= md.softmax_segm_max.shape[0]:
                segm = (md.softmax_segm_output, md.softmax_segm_max, md.softmax_segm_expsum)
            cuda_kernels.paged_attention(query[:n], key_cache, value_cache, output[:n], md.query_start_loc,
                                         md.seq_lens, md.block_table, md.max_query_len, self.scale, self.static,
                                         getattr(md, "kvq_long", None), segm)
            return output
        output[:n] = self._torch_attn(query[:n], key_cache, value_cache, md.query_start_loc,
                                      md.seq_lens[:num_reqs], md.block_table[:num_reqs], self.scale)
        return output
