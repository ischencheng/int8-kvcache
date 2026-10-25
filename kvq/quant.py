"""Symmetric INT8 quantization at the granularities discussed in the course.

All functions take x shaped [tokens, heads, head_dim] (a K or V slice) and return
a scale that broadcasts against x:

    granularity      scale shape    static?  (scale independent of the token)
    per_tensor       [1, 1, 1]      yes
    per_head         [1, H, 1]      yes
    per_channel      [1, H, D]      yes      (course definition: one scale per head x head_dim)
    per_token        [T, 1, 1]      no       (one scale per token vector, all heads)
    per_token_head   [T, H, 1]      no       (what vLLM 0.32 int8_per_token_head does)

We use the symmetric range [-127, 127] so that q = 0 means 0 and the grid is
symmetric; -128 is never produced.
"""
import torch

QMAX = 127.0
GRANULARITIES = ("per_tensor", "per_head", "per_channel", "per_token", "per_token_head")
STATIC = ("per_tensor", "per_head", "per_channel")
_REDUCE_DIMS = {
    "per_tensor": (0, 1, 2),
    "per_head": (0, 2),
    "per_channel": (0,),
    "per_token": (1, 2),
    "per_token_head": (2,),
}


def absmax(x, granularity, quantile=None):
    """|x| max (or quantile over tokens, static granularities only) kept-dim."""
    dims = _REDUCE_DIMS[granularity]
    a = x.abs().float()
    if quantile is None or quantile >= 1.0:
        return a.amax(dim=dims, keepdim=True)
    assert granularity in STATIC, "a quantile scale only makes sense for a static scale"
    # Move reduced dims to the front and take the k-th value along them.
    keep = [d for d in range(3) if d not in dims]
    a = a.permute(*dims, *keep).reshape(-1, *[x.shape[d] for d in keep])
    k = max(1, min(a.shape[0], int(round(quantile * a.shape[0]))))
    v = a.kthvalue(k, dim=0).values
    shape = [1 if d in dims else x.shape[d] for d in range(3)]
    return v.reshape(shape)


def scale_from_absmax(amax):
    return (amax / QMAX).clamp(min=1e-8)


def quantize(x, scale):
    return torch.clamp(torch.round(x.float() / scale), -QMAX, QMAX).to(torch.int8)


def dequantize(q, scale, dtype=torch.float32):
    return (q.float() * scale).to(dtype)


def quant_dequant(x, granularity, quantile=None, scale=None):
    if scale is None:
        scale = scale_from_absmax(absmax(x, granularity, quantile))
    return dequantize(quantize(x, scale), scale, x.dtype), scale


def rel_error(x, y):
    x = x.float()
    return ((x - y.float()).norm() / x.norm().clamp(min=1e-12)).item()
