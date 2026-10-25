"""Swap the attention math inside vLLM's FlashAttention backend for PyTorch code.

Installed by the plugin when KVQ_ATTN is set. vLLM still writes the bf16 KV cache
with its own kernel; we only replace the paged-attention call:
  KVQ_ATTN=ref           the stage-3 golden reference (level-3 integration check)
  KVQ_ATTN=<cfg>         stage 5: dynamic INT8 quantization, a name from kvq.int8_ref.CONFIGS
  KVQ_ATTN=static:<cfg>  stage-6 bisect tool: quantize the bf16 cache at read time with the
                         static scales in KVQ_SCALES, i.e. the INT8 backend's math without any
                         of its framework changes
Host syncs inside these functions make them eager-only (enforce_eager=True).
"""
import os


def resolve(mode):
    if mode == "ref":
        from kvq.ref_attn import paged_attention_ref

        return lambda layer: paged_attention_ref
    if mode.startswith("static:"):
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        from kvq import scales
        from kvq.int8_ref import CONFIGS, paged_static_attention

        cfg = CONFIGS[mode.split(":", 1)[1]]
        all_scales = scales.load(os.environ["KVQ_SCALES"])
        fns = {}

        def per_layer(layer):
            if layer.layer_name not in fns:
                st = scales.for_layer(all_scales, layer.layer_name, get_tensor_model_parallel_rank(),
                                      get_tensor_model_parallel_world_size(), "cuda")
                fns[layer.layer_name] = paged_static_attention(cfg, st, cache_is_int8=False)
            return fns[layer.layer_name]

        return per_layer
    from kvq.int8_ref import MODES

    return lambda layer: MODES[mode]


def install():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    fn_for = resolve(os.environ["KVQ_ATTN"])
    orig_forward = FlashAttentionImpl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if attn_metadata is None:
            return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        md = attn_metadata
        n = md.num_actual_tokens
        num_reqs = md.query_start_loc.shape[0] - 1
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        output[:n] = fn_for(layer)(query[:n], key_cache, value_cache, md.query_start_loc, md.seq_lens[:num_reqs],
                                   md.block_table[:num_reqs], self.scale)
        return output

    FlashAttentionImpl.forward = forward
