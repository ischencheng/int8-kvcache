"""Swap the attention math inside vLLM's FlashAttention backend for PyTorch code.

Installed by the plugin when KVQ_ATTN is set. vLLM still writes the bf16 KV cache
with its own kernel; we only replace the paged-attention call:
  KVQ_ATTN=ref     the stage-3 golden reference (level-3 integration check)
  KVQ_ATTN=<mode>  a stage-5 dynamic-quantization mode from kvq.int8_ref.MODES
Host syncs inside these functions make them eager-only (enforce_eager=True).
"""
import os


def resolve(mode):
    if mode == "ref":
        from kvq.ref_attn import paged_attention_ref

        return paged_attention_ref
    from kvq.int8_ref import MODES

    return MODES[mode]


def install():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    fn = resolve(os.environ["KVQ_ATTN"])
    orig_forward = FlashAttentionImpl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if attn_metadata is None:
            return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        md = attn_metadata
        n = md.num_actual_tokens
        num_reqs = md.query_start_loc.shape[0] - 1
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        output[:n] = fn(query[:n], key_cache, value_cache, md.query_start_loc, md.seq_lens[:num_reqs],
                        md.block_table[:num_reqs], self.scale)
        return output

    FlashAttentionImpl.forward = forward
