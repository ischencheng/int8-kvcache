"""Advanced item: INT8 KV for MLA (DeepSeek-V2-Lite), analysis and an end-to-end check.

An MLA layer caches one latent row per token: c = kv_c_normed (kv_lora_rank = 512) and
k_pe (64 RoPE dims). With the up-projections absorbed, decode is
    scores_t = (q_lat . c_t + q_pe . kpe_t) * scale      q_lat = q_nope @ W_UK  [512]
    o_lat    = sum_t p_t c_t                             then o = o_lat @ W_UV
so the same c is both "K" (contracted over its 512 channels) and "V" (contracted over
tokens). For a per-channel static scale s_c[512]:
  * QK role: s_c lies on the contracted dim -> fold into q_lat (as for GQA K).
  * PV role: s_c is constant over tokens -> multiply o_lat by s_c afterwards.
  * smooth (c - mu): softmax ignores q_lat . mu, but the PV role does not:
    sum_t p_t (c_t - mu) = o_lat - mu, so mu must be added back to o_lat. One extra
    512-vector add per output row, or fold mu @ W_UV into a bias.
k_pe carries RoPE (position dependent); it can stay bf16 (64 of 576 channels).

Hooks (installed by the plugin with KVQ_MLA=capture|fakequant):
  capture    per-layer stats of c / k_pe and one decode dump (q_lat, q_pe, used pages, output)
  fakequant  quantize-dequantize c (and k_pe with KVQ_MLA_ROPE=1) with static scales right
             before vLLM writes its bf16 cache: exactly what an INT8 latent cache would store.
"""
import os

import torch

from kvq import quant

STATE = {"order": {}, "stats": {}, "samples": {}, "dumps": {}}
L_RANK = 512
MAX_SAMPLES = int(os.environ.get("KVQ_CAPTURE_MAX_TOKENS", "8192"))
DUMP_AT = int(os.environ.get("KVQ_DUMP_AT_SEQLEN", "1024"))
DUMP_LAYERS = {int(x) for x in os.environ.get("KVQ_DUMP_LAYERS", "0,13,26").split(",") if x}


def _layer(impl):
    """MLA impls do not know their layer name; number them by first call (= layer order)."""
    return STATE["order"].setdefault(id(impl), len(STATE["order"]))


def _record(i, c, kpe):
    for side, x in (("c", c), ("kpe", kpe)):
        xf = x.float()
        new = {"absmax": xf.abs().amax(0), "max": xf.amax(0), "min": xf.amin(0), "sum": xf.double().sum(0)}
        st = STATE["stats"].setdefault(i, {})
        if side not in st:
            st[side] = {**new, "tokens": 0}
        else:
            s = st[side]
            torch.maximum(s["absmax"], new["absmax"], out=s["absmax"])
            torch.maximum(s["max"], new["max"], out=s["max"])
            torch.minimum(s["min"], new["min"], out=s["min"])
            s["sum"] += new["sum"]
        st[side]["tokens"] += x.shape[0]
    smp = STATE["samples"].setdefault(i, {"c": [], "kpe": [], "n": 0})
    if smp["n"] < MAX_SAMPLES:
        smp["c"].append(c.to(torch.float16).cpu())
        smp["kpe"].append(kpe.to(torch.float16).cpu())
        smp["n"] += c.shape[0]


def load_static(path):
    from safetensors.torch import load_file

    out = {}
    for k, t in load_file(path, device="cuda").items():
        i, field = k.split("|")
        out.setdefault(int(i), {})[field] = t
    return out


def install():
    from vllm.v1.attention.backend import MLAAttentionImpl
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

    mode = os.environ["KVQ_MLA"]
    orig_update = MLAAttentionImpl.do_kv_cache_update
    static = load_static(os.environ["KVQ_MLA_SCALES"]) if mode == "fakequant" else None
    rope_too = os.environ.get("KVQ_MLA_ROPE") == "1"

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        n = slot_mapping.flatten().shape[0]
        i = _layer(self)
        if mode == "capture" and kv_cache.numel():
            valid = slot_mapping.flatten() >= 0
            _record(i, kv_c_normed[:n][valid], k_pe[:n].reshape(n, -1)[valid])
        if mode == "fakequant" and kv_cache.numel():
            st = static[i]
            c = kv_c_normed.float() - st["c_mu"]
            kv_c_normed = (quant.dequantize(quant.quantize(c, st["c_scale"]), st["c_scale"]) + st["c_mu"]).to(
                kv_c_normed.dtype)
            if rope_too:
                shp = k_pe.shape
                kp = k_pe.reshape(shp[0], -1).float()
                k_pe = quant.dequantize(quant.quantize(kp, st["kpe_scale"]), st["kpe_scale"]).to(k_pe.dtype).view(shp)
        return orig_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)

    MLAAttentionImpl.do_kv_cache_update = do_kv_cache_update

    if mode == "capture":
        orig_mqa = TritonMLAImpl.forward_mqa

        def forward_mqa(self, q, kv_cache, attn_metadata, layer):
            o, lse = orig_mqa(self, q, kv_cache, attn_metadata, layer)
            i = _layer(self)
            dec = attn_metadata.decode
            if (i in DUMP_LAYERS and i not in STATE["dumps"] and isinstance(q, tuple)
                    and int(dec.seq_lens.min()) >= DUMP_AT and q[0].shape[0] == dec.seq_lens.shape[0]):
                bs = kv_cache.shape[1]
                pages = (dec.seq_lens + bs - 1) // bs
                used = torch.cat([dec.block_table[b, : pages[b]] for b in range(pages.shape[0])])
                remap = torch.zeros(int(dec.block_table.max()) + 1, dtype=torch.int32, device=used.device)
                remap[used] = torch.arange(used.numel(), dtype=torch.int32, device=used.device)
                bt = torch.zeros_like(dec.block_table)
                for b in range(pages.shape[0]):
                    bt[b, : pages[b]] = remap[dec.block_table[b, : pages[b]]]
                STATE["dumps"][i] = {"q_lat": q[0].cpu(), "q_pe": q[1].cpu(), "cache": kv_cache[used].cpu(),
                                     "block_table": bt.cpu(), "seq_lens": dec.seq_lens.cpu(), "o": o.cpu(),
                                     "scale": torch.tensor(self.scale)}
            return o, lse

        TritonMLAImpl.forward_mqa = forward_mqa


def collect(worker=None):
    out = {"stats": {}, "samples": {}, "dumps": STATE["dumps"]}
    for i, st in STATE["stats"].items():
        out["stats"][i] = {side: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
                           for side, d in st.items()}
    for i, s in STATE["samples"].items():
        out["samples"][i] = {"c": torch.cat(s["c"]), "kpe": torch.cat(s["kpe"])}
    return out


# ----------------------------------------------------------------------------------------
# Reference MLA decode on a paged latent cache, bf16 and INT8 (implementations A / B).
# ----------------------------------------------------------------------------------------

def gather(cache, block_table, seq_len, b):
    bs = cache.shape[1]
    pages = (seq_len + bs - 1) // bs
    return cache[block_table[b, :pages].long()].flatten(0, 1)[:seq_len]   # [T, 576]


def decode_ref(q_lat, q_pe, cache, block_table, seq_lens, scale, cfg=None, static=None):
    """q_lat [B, N, 512], q_pe [B, N, 64]; returns o_lat [B, N, 512] (fp32).

    cfg None: bf16 reference. cfg dict: {"impl": "A"|"B", "smooth": bool, "rope_int8": bool}
    with static scales c_scale [512], c_mu [512], kpe_scale [64] from calibration.
    """
    outs = []
    for b in range(q_lat.shape[0]):
        kv = gather(cache, block_table, int(seq_lens[b]), b).float()
        c, kpe = kv[:, :L_RANK], kv[:, L_RANK:]
        ql, qp = q_lat[b].float(), q_pe[b].float()                       # [N, 512], [N, 64]
        if cfg is None:
            p = torch.softmax((ql @ c.T + qp @ kpe.T) * scale, -1)
            outs.append(p @ c)
            continue
        mu = static["c_mu"] if cfg["smooth"] else torch.zeros_like(static["c_mu"])
        sc = static["c_scale"]
        c_i8 = quant.quantize(c - mu, sc)
        kpe_q = quant.dequantize(quant.quantize(kpe, static["kpe_scale"]), static["kpe_scale"]) \
            if cfg["rope_int8"] else kpe
        if cfg["impl"] == "A":
            cd = quant.dequantize(c_i8, sc) + mu
            p = torch.softmax((ql @ cd.T + qp @ kpe_q.T) * scale, -1)
            outs.append(p @ cd)
            continue
        # B: QK on INT8 with s_c folded into q_lat (mu drops out of softmax), rope part in fp32.
        qf = ql * sc
        qs = quant.scale_from_absmax(qf.abs().amax(-1, keepdim=True))
        s_lat = (quant.quantize(qf, qs).double() @ c_i8.double().T).float() * qs
        s = (s_lat + qp @ kpe_q.T) * scale
        e = torch.exp(s - s.amax(-1, keepdim=True))
        # P per 64-key tile, INT8 PV, then s_c on the 512 output channels and + mu (PV role).
        T = e.shape[-1]
        t = torch.nn.functional.pad(e, (0, (-T) % 64)).unflatten(-1, (-1, 64))
        ps = (t.amax(-1, keepdim=True) / quant.QMAX).clamp(min=1e-30)
        e_hat = (torch.round(t / ps) * ps).flatten(-2)[..., :T]
        o = (e_hat.double() @ c_i8.double()).float() / e_hat.sum(-1, keepdim=True)
        outs.append(o * sc + mu)
    return torch.stack(outs)
