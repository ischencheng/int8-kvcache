"""Python side of the CUDA kernels in kvq/cuda/int8_kv.cu (JIT-compiled on first use).

Same call shapes as the Triton versions, so the backend can switch with KVQ_KERNEL=cuda.
The CUDA attention kernel covers decode (one query token per request); multi-token
batches keep using the Triton dequantize kernel.
"""
import os

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        major, minor = torch.cuda.get_device_capability()
        # The vLLM image presets 7 architectures for its own builds; compile only for this GPU.
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        src = os.path.join(os.path.dirname(__file__), "cuda", "int8_kv.cu")
        _EXT = load(name="kvq_int8_kv", sources=[src], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def quant_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, static):
    ext().quant_cache(key, value, key_cache, value_cache, slot_mapping, static["k_scale"], static["v_scale"],
                      static["k_mu"])


def quant_q(q, k_scale, group=None):
    return tuple(ext().quant_q(q, k_scale, group or q.shape[-1]))


def decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, static,
                     segm=None):
    so, sm, ss = segm if segm is not None else (None, None, None)
    ext().int8_decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, static["k_scale"],
                                static["v_scale"], softmax_scale, so, sm, ss)
    return out
