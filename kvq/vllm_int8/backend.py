"""Static INT8 KV cache backend for vLLM 0.32.0 (registered as AttentionBackendEnum.CUSTOM).

Run with  kv_cache_dtype="int8_static", attention_config={"backend": "CUSTOM"}  and
  KVQ_SCALES  calibrated scales (safetensors, see kvq/scales.py)
  KVQ_CFG     numerics of the decode path, a name from kvq.int8_ref.CONFIGS (default B_channel_smooth)
  KVQ_KERNEL  "torch" (stage-6 PyTorch path) or "triton" (stage-7 kernels)

What changes compared with the bf16 Triton backend it extends (the stage-6 list):
  (a) the cache is allocated as int8 (patch: CacheDType "int8_static" -> torch.int8), so the
      same memory holds twice the blocks;
  (b)/(e) do_kv_cache_update quantizes the new K/V with the static scales before writing,
      for prefill and decode alike (K first has the smooth-K mean removed);
  (c) multi-token requests (prefill, chunked prefill, spec-decode verification) read the int8
      cache and dequantize it;
  (d) decode runs INT8 arithmetic: q is quantized dynamically, K/V scales are static.
With kv_cache_dtype "auto" the backend runs the bf16 path (torch: reference, triton: our kernel
in bf16 mode), which gives a same-code baseline for the performance comparison.
"""
import os
from typing import ClassVar

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)

from kvq import quant, scales
from kvq.int8_ref import CONFIGS, paged_static_attention

logger = init_logger("vllm.kvq")  # under "vllm" so vLLM's log handler prints it
_SCALES_CACHE = {}


def _load_scales(path):
    if path not in _SCALES_CACHE:
        _SCALES_CACHE[path] = scales.load(path)
    return _SCALES_CACHE[path]


class Int8StaticBackend(TritonAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list] = ["int8_static", "auto", "bfloat16", "float16"]

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
    """Metadata as for the Triton backend. Built once per KV-cache group after the model is
    loaded and before CUDA graphs are captured, which makes it the place to hand every layer
    its static scales (persistent tensors that a captured graph can keep pointing at)."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.cache_config.cache_dtype != "int8_static":
            return
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
                    os.environ.get("KVQ_CFG", "B_channel_smooth"), os.environ.get("KVQ_KERNEL", "torch"))


class Int8StaticImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.int8 = self.kv_cache_dtype == "int8_static"
        self.cfg = CONFIGS[os.environ.get("KVQ_CFG", "B_channel_smooth")]
        self.kernel = os.environ.get("KVQ_KERNEL", "torch")
        self.static = None

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
        if not self.int8:
            if self.kernel == "torch":
                return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        key_cache, value_cache = self._caches(kv_cache)
        if self.kernel == "triton":
            from kvq.triton_kernels.quant_cache import quant_reshape_and_cache

            return quant_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                                           self.static if self.int8 else None)
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
        key_cache, value_cache = self._caches(kv_cache)
        if self.kernel == "triton":
            from kvq.triton_kernels.int8_attention import int8_paged_attention

            int8_paged_attention(query[:n], key_cache, value_cache, output[:n], md, self.scale,
                                 self.static if self.int8 else None, self.cfg)
            return output
        if not self.int8:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                                   output_scale, output_block_scale)
        num_reqs = md.query_start_loc.shape[0] - 1
        output[:n] = self._torch_attn(query[:n], key_cache, value_cache, md.query_start_loc,
                                      md.seq_lens[:num_reqs], md.block_table[:num_reqs], self.scale)
        return output
