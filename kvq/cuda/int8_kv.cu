// CUDA kernels for the static INT8 KV cache (stage 7, CUDA version).
//
//   quant_cache        fused "quantize + reshape_and_cache" for one step's new K/V
//   quant_q            fold the per-channel K scale into q and quantize q (standalone variant)
//   int8_decode_attn   paged attention for short queries (decode, spec-decode verification: at most
//                      qmax query tokens per request) on the INT8 cache with INT8 tensor cores
//                      (mma.sync.m16n8k32.s8), optional split-KV + reduce
//   dequant_pages      longer queries (prefill): dequantize their pages into a bf16 paged cache
//                      that FlashAttention reads
//
// Numerics are those of kvq/int8_ref.py:attend_int8 (implementation B):
//   q' = q * s_k[kvh, :] ; q_i8 = rint(q' / s_q) with one s_q per (token, q head)
//   s  = (q_i8 . k_i8) * s_q * softmax_scale            (int32 accumulate)
//   P  = exp2(s - m) quantized per (row, 32-token chunk): p_i8 = rint(P / (rowmax / 127))
//   o  = sum_chunks (p_i8 . v_i8) * rowmax / 127, divided by the sum of the quantized P, times s_v[d]
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: that pulls cuSPARSE/cuBLAS headers
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int D = 128;        // head_dim this kernel is written for
constexpr int CHUNK = 32;     // tokens a warp processes per iteration (= k of the PV mma)
constexpr int NWARPS = 4;
constexpr int KROW = D + 16;  // smem K row stride in bytes: 144 -> the B-fragment loads are conflict free
constexpr int QROW = D + 16;

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int8_t sat_i8(float x) {
  int v = __float2int_rn(x);  // round half to even, like torch.round
  return static_cast<int8_t>(max(-127, min(127, v)));
}

// Where token tau (0..31 inside a chunk) sits among the k slots of the PV A fragment.
// The QK accumulator gives thread (g, t) tokens 8j + 2t + e (n-tile j, element e); we let those
// fill its A slots in order, so P never moves between registers. V^T in smem is stored in the
// same permuted token order, which makes each B fragment a plain 32-bit load.
__device__ __forceinline__ int slot_of_token(int tau) {
  int j = tau >> 3, r = tau & 7, t = r >> 1, e = r & 1;
  return (j >= 2 ? 16 : 0) + 4 * t + 2 * (j & 1) + e;
}

// ------------------------------------------------------------------------------------------
// Fused quantize + reshape_and_cache. One block per token, 8 channels per thread.
// ------------------------------------------------------------------------------------------
template <bool PC_K, bool PC_V>
__global__ void quant_cache_kernel(const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
                                   int8_t* __restrict__ kc, int8_t* __restrict__ vc,
                                   const int64_t* __restrict__ slots, const float* __restrict__ ks,
                                   const float* __restrict__ vs, const float* __restrict__ kmu, int64_t k_st,
                                   int64_t k_sh, int64_t v_st, int64_t v_sh, int64_t kc_sb, int64_t kc_sn,
                                   int64_t kc_sh, int64_t vc_sb, int64_t vc_sn, int64_t vc_sh, int H,
                                   int block_size) {
  const int64_t slot = slots[blockIdx.x];
  if (slot < 0) return;  // padding token
  const int64_t blk = slot / block_size, off = slot % block_size;
  for (int i = threadIdx.x; i < H * (D / 8); i += blockDim.x) {
    const int h = i / (D / 8), d0 = (i % (D / 8)) * 8;
    const uint4 kr = *reinterpret_cast<const uint4*>(key + blockIdx.x * k_st + h * k_sh + d0);
    const uint4 vr = *reinterpret_cast<const uint4*>(value + blockIdx.x * v_st + h * v_sh + d0);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kr);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vr);
    alignas(8) int8_t kq[8], vq[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int c = h * D + d0 + e;
      const float kscale = PC_K ? ks[c] : ks[h];
      const float vscale = PC_V ? vs[c] : vs[h];
      kq[e] = sat_i8(__fdiv_rn(__bfloat162float(kb[e]) - kmu[c], kscale));
      vq[e] = sat_i8(__fdiv_rn(__bfloat162float(vb[e]), vscale));
    }
    *reinterpret_cast<uint2*>(kc + blk * kc_sb + off * kc_sn + h * kc_sh + d0) = *reinterpret_cast<uint2*>(kq);
    *reinterpret_cast<uint2*>(vc + blk * vc_sb + off * vc_sn + h * vc_sh + d0) = *reinterpret_cast<uint2*>(vq);
  }
}

// ------------------------------------------------------------------------------------------
// q fold + quantize. One warp per (token, q head), 4 channels per lane; GROUP = 128 or 32 channels.
// ------------------------------------------------------------------------------------------
template <int GROUP, bool PC_K>
__global__ void quant_q_kernel(const __nv_bfloat16* __restrict__ q, int8_t* __restrict__ q_i8,
                               float* __restrict__ q_scale, const float* __restrict__ ks, int64_t q_st,
                               int64_t q_sh, int G) {
  const int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, Hq = gridDim.y;
  const int d0 = lane * 4;
  float x[4], amax = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    x[e] = __bfloat162float(q[t * q_st + h * q_sh + d0 + e]) * (PC_K ? ks[(h / G) * D + d0 + e] : 1.f);
    amax = fmaxf(amax, fabsf(x[e]));
  }
#pragma unroll
  for (int o = 1; o < GROUP / 4; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  const float s = fmaxf(amax / 127.f, 1e-8f);
  alignas(4) int8_t out[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) out[e] = sat_i8(__fdiv_rn(x[e], s));
  *reinterpret_cast<uint32_t*>(q_i8 + (static_cast<int64_t>(t) * Hq + h) * D + d0) =
      *reinterpret_cast<uint32_t*>(out);
  if (lane % (GROUP / 4) == 0) q_scale[(static_cast<int64_t>(t) * Hq + h) * (D / GROUP) + d0 / GROUP] = s;
}

// ------------------------------------------------------------------------------------------
// INT8 attention for short queries. Grid (num_seqs, num_kv_heads, num_splits x m_tiles), 4 warps.
// The q_len x G (token, q head) pairs of this kv head are rows R = t * G + h, 16 per m tile; for decode
// (q_len 1) the single tile holds the G q heads. Row R sees keys n < L - q_len + 1 + t (causal).
// Requests with more than qmax query tokens are left to the dequantize + FlashAttention path.
//
// Data movement (v2, after ncu showed exposed load latency and an instruction-heavy V transpose):
//  * each warp streams its own 32-token chunks with cp.async into a 2-stage smem ring, so the next
//    chunk's K/V are in flight while the current one is computed;
//  * no V^T buffer: the token permutation that lets P feed the PV A fragment straight from the QK
//    accumulator is applied when reading K rows (free), V keeps natural token order, and each thread
//    builds its PV B fragments with 4 x LDS.32 + a 4x4 byte transpose (__byte_perm). The PV output
//    columns come out channel-permuted (chan_of) and are put back in place in the combine step.
// ------------------------------------------------------------------------------------------
constexpr int STAGE_BYTES = CHUNK * KROW;  // one K or V chunk in smem (rows padded to 144 B)
constexpr int ACC_ROW = D + 1;             // combine buffer row stride in floats

__device__ __forceinline__ float row_sum(const float* s_acc, int row, int d) {
  float x = 0.f;
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) x += s_acc[(w * 16 + row) * ACC_ROW + d];
  return x;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int n = valid ? 16 : 0;  // src-size 0 -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// PV n-tile j, column n -> V channel: thread column n reads 4 consecutive channels for tiles 4q..4q+3.
__device__ __forceinline__ int chan_of(int j, int n) { return 32 * (j >> 2) + 4 * n + (j & 3); }

__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
  return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | (static_cast<uint32_t>(d & 0xff) << 24);
}

template <bool PC_K, bool PC_V>
__global__ void __launch_bounds__(NWARPS * 32)
    int8_decode_attn_kernel(const __nv_bfloat16* __restrict__ q, const int8_t* __restrict__ kc,
                            const int8_t* __restrict__ vc, __nv_bfloat16* __restrict__ out,
                            float* __restrict__ segm_out, float* __restrict__ segm_max,
                            float* __restrict__ segm_sum, const int* __restrict__ cu_q,
                            const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                            const float* __restrict__ ks, const float* __restrict__ vs, float sm_scale_log2,
                            int64_t q_st, int64_t q_sh, int64_t o_st, int64_t o_sh, int64_t k_sb, int64_t k_sn,
                            int64_t k_sh, int64_t v_sb, int64_t v_sn, int64_t v_sh, int bt_stride, int block_size,
                            int lbs, int G, int64_t seg_st, int64_t seg_sh, int num_splits, int qmax) {
  extern __shared__ __align__(16) char smem_raw[];
  int8_t* s_q = reinterpret_cast<int8_t*>(smem_raw);                 // 16 x QROW
  float* s_qscale = reinterpret_cast<float*>(smem_raw + 16 * QROW);  // 16
  float* s_m = s_qscale + 16;                                        // NWARPS x 16
  float* s_l = s_m + NWARPS * 16;                                    // NWARPS x 16
  char* ring = smem_raw + 16 * QROW + (16 + 2 * NWARPS * 16) * 4;    // per warp: 2 stages x (K, V)
  float* s_acc = reinterpret_cast<float*>(ring);                     // NWARPS x 16 x ACC_ROW, after the loop

  const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z % num_splits, mt = blockIdx.z / num_splits;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tok = cu_q[seq], qlen = cu_q[seq + 1] - tok;
  // Padding requests have 0 tokens; long queries go to FlashAttention; tiles past the last row idle.
  if (qlen == 0 || qlen > qmax || mt * 16 >= qlen * G) return;
  const int L = seq_lens[seq];
  const int nchunks = (L + CHUNK - 1) / CHUNK;
  const int per_split = (nchunks + num_splits - 1) / num_splits;
  const int c_begin = split * per_split, c_end = min(nchunks, c_begin + per_split);
  const int* bt = block_table + static_cast<int64_t>(seq) * bt_stride;
  int8_t* wring = reinterpret_cast<int8_t*>(ring) + warp * 4 * STAGE_BYTES;

  // A 32-token chunk spans at most two pages (block_size is a power of two >= 16): look the page ids
  // up once per chunk, one round ahead, instead of a divide/modulo per token.
  auto load_pages = [&](int cc, int& pa0, int& pa1) {
    if (cc < c_end) {
      const int n0 = cc * CHUNK;
      pa0 = bt[n0 >> lbs];
      pa1 = (n0 + 16 < L) ? bt[(n0 + 16) >> lbs] : pa0;
    }
  };
  // cp.async copies of chunk c into stage st (8 lanes per token row, 4 rows per step, coalesced).
  auto issue = [&](int c, int st, int pa0, int pa1) {
    int8_t* sk = wring + (2 * st) * STAGE_BYTES;
    int8_t* sv = sk + STAGE_BYTES;
    const int8_t* kb0 = kc + static_cast<int64_t>(pa0) * k_sb + kvh * k_sh + (lane & 7) * 16;
    const int8_t* kb1 = kc + static_cast<int64_t>(pa1) * k_sb + kvh * k_sh + (lane & 7) * 16;
    const int8_t* vb0 = vc + static_cast<int64_t>(pa0) * v_sb + kvh * v_sh + (lane & 7) * 16;
    const int8_t* vb1 = vc + static_cast<int64_t>(pa1) * v_sb + kvh * v_sh + (lane & 7) * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int tau = (lane >> 3) + 4 * i;
      const int n = c * CHUNK + tau;
      const int off = n & (block_size - 1);
      const bool ok = n < L;
      const bool second = tau >= 16;
      cp_async16(sk + tau * KROW + (lane & 7) * 16, (second ? kb1 : kb0) + off * k_sn, ok);
      cp_async16(sv + tau * KROW + (lane & 7) * 16, (second ? vb1 : vb0) + off * v_sn, ok);
    }
    cp_async_commit();
  };
  int c = c_begin + warp;
  int pc0 = 0, pc1 = 0, pn0 = 0, pn1 = 0;
  load_pages(c, pc0, pc1);
  if (c < c_end) issue(c, 0, pc0, pc1);  // the first chunk is already in flight during the q prologue
  load_pages(c + NWARPS, pn0, pn1);

  // ---- prologue: fold s_k into q, quantize per row (8 threads per row, 16 channels each) ----
  {
    const int row = threadIdx.x >> 3, part = threadIdx.x & 7, d0 = part * 16;
    const int R = mt * 16 + row, rt = R / G, rh = R % G;
    float x[16], amax = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      x[e] = 0.f;
      if (rt < qlen) {
        x[e] = __bfloat162float(q[(tok + rt) * q_st + (kvh * G + rh) * q_sh + d0 + e]);
        if (PC_K) x[e] *= ks[kvh * D + d0 + e];
      }
      amax = fmaxf(amax, fabsf(x[e]));
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float s = fmaxf(amax / 127.f, 1e-8f);
    uint32_t w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      w[k] = pack4(sat_i8(__fdiv_rn(x[4 * k], s)), sat_i8(__fdiv_rn(x[4 * k + 1], s)),
                   sat_i8(__fdiv_rn(x[4 * k + 2], s)), sat_i8(__fdiv_rn(x[4 * k + 3], s)));
    *reinterpret_cast<uint4*>(&s_q[row * QROW + d0]) = make_uint4(w[0], w[1], w[2], w[3]);
    if (part == 0) s_qscale[row] = s * (PC_K ? 1.f : ks[kvh]) * sm_scale_log2;
  }
  __syncthreads();
  uint32_t a[4][4];  // q A fragments for the 4 k-steps (32 channels each)
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    a[s][0] = *reinterpret_cast<const uint32_t*>(&s_q[g * QROW + 32 * s + 4 * t]);
    a[s][1] = *reinterpret_cast<const uint32_t*>(&s_q[(g + 8) * QROW + 32 * s + 4 * t]);
    a[s][2] = *reinterpret_cast<const uint32_t*>(&s_q[g * QROW + 32 * s + 16 + 4 * t]);
    a[s][3] = *reinterpret_cast<const uint32_t*>(&s_q[(g + 8) * QROW + 32 * s + 16 + 4 * t]);
  }
  const float rs0 = s_qscale[g], rs1 = s_qscale[g + 8];
  // Causal key limit of this thread's two rows (rows past the last token keep L; they are not written).
  auto row_lim = [&](int r) {
    const int rt = (mt * 16 + r) / G;
    return rt < qlen ? L - qlen + 1 + rt : L;
  };
  const int lim0 = row_lim(g), lim1 = row_lim(g + 8);
  int krow[4];  // K row read for QK n-tile j: the P-friendly token permutation
#pragma unroll
  for (int j = 0; j < 4; ++j) krow[j] = slot_of_token(8 * j + g);

  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  for (int st = 0; c < c_end; c += NWARPS, st ^= 1) {
    if (c + NWARPS < c_end) {
      issue(c + NWARPS, st ^ 1, pn0, pn1);
      load_pages(c + 2 * NWARPS, pn0, pn1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncwarp();
    const int8_t* sk = wring + (2 * st) * STAGE_BYTES;
    const int8_t* sv = sk + STAGE_BYTES;

    // QK: 4 n-tiles of 8 (permuted) tokens x 4 k-steps of 32 channels.
    int sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0;
      const int8_t* kr = &sk[krow[j] * KROW + 4 * t];
#pragma unroll
      for (int s = 0; s < 4; ++s)
        mma_s8(sacc[j], a[s], *reinterpret_cast<const uint32_t*>(kr + 32 * s),
               *reinterpret_cast<const uint32_t*>(kr + 32 * s + 16));
    }
    // Accumulator (j, e) is the score of token (j >= 2 ? 16 : 0) + 4t + 2(j & 1) + e.
    float s0[8], s1[8], mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int n = c * CHUNK + (j >= 2 ? 16 : 0) + 4 * t + 2 * (j & 1) + e;
        s0[2 * j + e] = n < lim0 ? sacc[j][e] * rs0 : -INFINITY;
        s1[2 * j + e] = n < lim1 ? sacc[j][2 + e] * rs1 : -INFINITY;
        mx0 = fmaxf(mx0, s0[2 * j + e]);
        mx1 = fmaxf(mx1, s1[2 * j + e]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    float pm0 = 0.f, pm1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s0[i] = exp2f(s0[i] - mn0);  // now P
      s1[i] = exp2f(s1[i] - mn1);
      pm0 = fmaxf(pm0, s0[i]);
      pm1 = fmaxf(pm1, s1[i]);
    }
    pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 1));
    pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 2));
    pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 1));
    pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 2));
    const float ps0 = fmaxf(pm0 / 127.f, 1e-30f), ps1 = fmaxf(pm1 / 127.f, 1e-30f);
    const float ri0 = 1.f / ps0, ri1 = 1.f / ps1;
    int q0[8], q1[8], qs0 = 0, qs1 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q0[i] = min(__float2int_rn(s0[i] * ri0), 127);
      q1[i] = min(__float2int_rn(s1[i] * ri1), 127);
      qs0 += q0[i];
      qs1 += q1[i];
    }
    qs0 += __shfl_xor_sync(0xffffffff, qs0, 1);
    qs0 += __shfl_xor_sync(0xffffffff, qs0, 2);
    qs1 += __shfl_xor_sync(0xffffffff, qs1, 1);
    qs1 += __shfl_xor_sync(0xffffffff, qs1, 2);
    l0 = l0 * al0 + qs0 * ps0;
    l1 = l1 * al1 + qs1 * ps1;
    // A fragment slots 4t..4t+3 / 16+4t..16+4t+3 hold tokens 4t.. / 16+4t.. (natural order).
    const uint32_t pa[4] = {pack4(q0[0], q0[1], q0[2], q0[3]), pack4(q1[0], q1[1], q1[2], q1[3]),
                            pack4(q0[4], q0[5], q0[6], q0[7]), pack4(q1[4], q1[5], q1[6], q1[7])};
    // PV: for each group of 4 n-tiles, read V[4 tokens][4 channels] twice (tokens 4t.. and 16+4t..)
    // as 32-bit words and transpose the 4x4 bytes into the B fragments of those 4 tiles.
#pragma unroll
    for (int qg = 0; qg < 4; ++qg) {
      uint32_t b0[4], b1[4];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int8_t* vp = &sv[(16 * half + 4 * t) * KROW + 32 * qg + 4 * g];
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(vp);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(vp + KROW);
        const uint32_t w2 = *reinterpret_cast<const uint32_t*>(vp + 2 * KROW);
        const uint32_t w3 = *reinterpret_cast<const uint32_t*>(vp + 3 * KROW);
        const uint32_t lo01 = __byte_perm(w0, w1, 0x5140), hi01 = __byte_perm(w0, w1, 0x7362);
        const uint32_t lo23 = __byte_perm(w2, w3, 0x5140), hi23 = __byte_perm(w2, w3, 0x7362);
        uint32_t* b = half ? b1 : b0;
        b[0] = __byte_perm(lo01, lo23, 0x5410);
        b[1] = __byte_perm(lo01, lo23, 0x7632);
        b[2] = __byte_perm(hi01, hi23, 0x5410);
        b[3] = __byte_perm(hi01, hi23, 0x7632);
      }
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int j = 4 * qg + cc;
        int pacc[4] = {0, 0, 0, 0};
        mma_s8(pacc, pa, b0[cc], b1[cc]);
        acc[j][0] = acc[j][0] * al0 + pacc[0] * ps0;
        acc[j][1] = acc[j][1] * al0 + pacc[1] * ps0;
        acc[j][2] = acc[j][2] * al1 + pacc[2] * ps1;
        acc[j][3] = acc[j][3] * al1 + pacc[3] * ps1;
      }
    }
    m0 = mn0;
    m1 = mn1;
    __syncwarp();  // this stage is refilled by the next iteration's issue
  }

  // ---- combine the 4 warps: weights exp2(m_w - M) ----
  if (t == 0) {
    s_m[warp * 16 + g] = m0;
    s_m[warp * 16 + g + 8] = m1;
    s_l[warp * 16 + g] = l0;
    s_l[warp * 16 + g + 8] = l1;
  }
  __syncthreads();  // also ends every use of the K/V ring, which s_acc reuses
  float M0 = -1e30f, M1 = -1e30f;
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) {
    M0 = fmaxf(M0, s_m[w * 16 + g]);
    M1 = fmaxf(M1, s_m[w * 16 + g + 8]);
  }
  const float w0 = exp2f(m0 - M0), w1 = exp2f(m1 - M1);
  // Each warp writes its rescaled rows to its own slice (row stride ACC_ROW = 129 floats keeps the
  // 32 lanes on distinct banks); the epilogue sums the 4 slices. No shared-memory atomics: on sm_80
  // a float atomicAdd to smem is a CAS loop, which ncu showed dominating the split-KV launches.
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int ch = chan_of(j, 2 * t + e);
      s_acc[(warp * 16 + g) * ACC_ROW + ch] = acc[j][e] * w0;
      s_acc[(warp * 16 + g + 8) * ACC_ROW + ch] = acc[j][2 + e] * w1;
    }
  __syncthreads();

  // ---- epilogue: 8 threads per row, 16 channels each; s_v applied here ----
  const int row = threadIdx.x >> 3, d0 = (threadIdx.x & 7) * 16;
  const int R = mt * 16 + row, rt = R / G;
  if (rt >= qlen) return;
  const int qt = tok + rt;  // this row's query token
  float Mr = -1e30f, Lr = 0.f;
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) Mr = fmaxf(Mr, s_m[w * 16 + row]);
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) Lr += s_l[w * 16 + row] * exp2f(s_m[w * 16 + row] - Mr);
  const int head = kvh * G + R % G;
  if (num_splits == 1) {
    const float inv = 1.f / fmaxf(Lr, 1e-30f);
    uint32_t o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int d = d0 + 2 * e;
      const float v0 = row_sum(s_acc, row, d) * inv * (PC_V ? vs[kvh * D + d] : vs[kvh]);
      const float v1 = row_sum(s_acc, row, d + 1) * inv * (PC_V ? vs[kvh * D + d + 1] : vs[kvh]);
      const __nv_bfloat162 h = __floats2bfloat162_rn(v0, v1);
      o[e] = *reinterpret_cast<const uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(out + qt * o_st + head * o_sh + d0) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(out + qt * o_st + head * o_sh + d0 + 8) = make_uint4(o[4], o[5], o[6], o[7]);
  } else {
    const int64_t sidx = qt * seg_st + head * seg_sh + split;
    if ((threadIdx.x & 7) == 0) {
      segm_max[sidx] = Mr;
      segm_sum[sidx] = Lr;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e)
      segm_out[sidx * D + d0 + e] = row_sum(s_acc, row, d0 + e) * (PC_V ? vs[kvh * D + d0 + e] : vs[kvh]);
  }
}

constexpr size_t DECODE_SMEM = 16 * QROW + (16 + 2 * NWARPS * 16) * 4 + NWARPS * 4 * STAGE_BYTES;
static_assert(NWARPS * 4 * STAGE_BYTES >= NWARPS * 16 * ACC_ROW * 4, "s_acc must fit in the K/V ring");

// Merge split-KV partials: o = sum_s 2^(m_s - M) acc_s / sum_s 2^(m_s - M) l_s.
__global__ void split_reduce_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ segm_out,
                                    const float* __restrict__ segm_max, const float* __restrict__ segm_sum,
                                    const int* __restrict__ cu_q, int64_t o_st, int64_t o_sh, int64_t seg_st,
                                    int64_t seg_sh, int S, int qmax) {
  const int seq = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int tok = cu_q[seq], qlen = cu_q[seq + 1] - tok;
  if (qlen > qmax) return;
  for (int qt = tok; qt < tok + qlen; ++qt) {
    const int64_t base = qt * seg_st + h * seg_sh;
    float M = -1e30f;
    for (int s = 0; s < S; ++s) M = fmaxf(M, segm_max[base + s]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < S; ++s) {
      const float w = exp2f(segm_max[base + s] - M);
      num += w * segm_out[(base + s) * D + d];
      den += w * segm_sum[base + s];
    }
    out[qt * o_st + h * o_sh + d] = __float2bfloat16(num / fmaxf(den, 1e-30f));
  }
}

// ------------------------------------------------------------------------------------------
// Long queries: copy their pages out of the INT8 cache as bf16 ([pages, block, H, D], the layout
// FlashAttention's paged KV reads). One block per output page, 8 channels per thread. K keeps the
// smooth-K offset (k - mu): softmax does not see the constant q.mu, as in the INT8 kernel.
// ------------------------------------------------------------------------------------------
template <bool PC_K, bool PC_V>
__global__ void dequant_pages_kernel(const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
                                     __nv_bfloat16* __restrict__ ko, __nv_bfloat16* __restrict__ vo,
                                     const int* __restrict__ src_pages, const float* __restrict__ ks,
                                     const float* __restrict__ vs, int64_t k_sb, int64_t k_sn, int64_t k_sh,
                                     int64_t v_sb, int64_t v_sn, int64_t v_sh, int H, int block_size) {
  const int64_t src = src_pages[blockIdx.x];
  const int64_t dst = static_cast<int64_t>(blockIdx.x) * block_size * H * D;
  for (int i = threadIdx.x; i < block_size * H * (D / 8); i += blockDim.x) {
    const int d0 = (i % (D / 8)) * 8, h = (i / (D / 8)) % H, n = i / (H * (D / 8));
    const uint2 kr = *reinterpret_cast<const uint2*>(kc + src * k_sb + n * k_sn + h * k_sh + d0);
    const uint2 vr = *reinterpret_cast<const uint2*>(vc + src * v_sb + n * v_sn + h * v_sh + d0);
    const int8_t* kb = reinterpret_cast<const int8_t*>(&kr);
    const int8_t* vb = reinterpret_cast<const int8_t*>(&vr);
    uint32_t ow[4], vw[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int c = h * D + d0 + 2 * e;
      const __nv_bfloat162 kh = __floats2bfloat162_rn(kb[2 * e] * (PC_K ? ks[c] : ks[h]),
                                                      kb[2 * e + 1] * (PC_K ? ks[c + 1] : ks[h]));
      const __nv_bfloat162 vh = __floats2bfloat162_rn(vb[2 * e] * (PC_V ? vs[c] : vs[h]),
                                                      vb[2 * e + 1] * (PC_V ? vs[c + 1] : vs[h]));
      ow[e] = *reinterpret_cast<const uint32_t*>(&kh);
      vw[e] = *reinterpret_cast<const uint32_t*>(&vh);
    }
    const int64_t o = dst + (static_cast<int64_t>(n) * H + h) * D + d0;
    *reinterpret_cast<uint4*>(ko + o) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    *reinterpret_cast<uint4*>(vo + o) = make_uint4(vw[0], vw[1], vw[2], vw[3]);
  }
}

}  // namespace

void quant_cache(torch::Tensor key, torch::Tensor value, torch::Tensor kc, torch::Tensor vc, torch::Tensor slots,
                 torch::Tensor ks, torch::Tensor vs, torch::Tensor kmu) {
  const int n = slots.size(0), H = key.size(1);
  TORCH_CHECK(key.size(2) == D && key.stride(2) == 1 && kc.stride(3) == 1, "head_dim 128, contiguous last dim");
  if (n == 0) return;
  auto stream = c10::cuda::getCurrentCUDAStream();
  const bool pk = ks.size(-1) > 1, pv = vs.size(-1) > 1;
  auto launch = [&](auto kernel) {
    kernel<<<n, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(), slots.data_ptr<int64_t>(), ks.data_ptr<float>(),
        vs.data_ptr<float>(), kmu.data_ptr<float>(), key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2), H, kc.size(1));
  };
  if (pk && pv) launch(quant_cache_kernel<true, true>);
  else if (pk) launch(quant_cache_kernel<true, false>);
  else if (pv) launch(quant_cache_kernel<false, true>);
  else launch(quant_cache_kernel<false, false>);
}

std::vector<torch::Tensor> quant_q(torch::Tensor q, torch::Tensor ks, int64_t group) {
  const int T = q.size(0), Hq = q.size(1);
  TORCH_CHECK(q.size(2) == D && (group == 128 || group == 32));
  auto q_i8 = torch::empty({T, Hq, D}, q.options().dtype(torch::kInt8));
  auto q_s = torch::empty({T, Hq, D / group}, q.options().dtype(torch::kFloat32));
  if (T == 0) return {q_i8, q_s};
  const int G = Hq / ks.size(0);
  const bool pk = ks.size(-1) > 1;
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(T, Hq);
#define KVQ_LAUNCH_Q(GR, PK)                                                                                    \
  quant_q_kernel<GR, PK><<<grid, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),         \
                                                  q_i8.data_ptr<int8_t>(), q_s.data_ptr<float>(),               \
                                                  ks.data_ptr<float>(), q.stride(0), q.stride(1), G)
  if (group == 128) {
    if (pk) KVQ_LAUNCH_Q(128, true); else KVQ_LAUNCH_Q(128, false);
  } else {
    if (pk) KVQ_LAUNCH_Q(32, true); else KVQ_LAUNCH_Q(32, false);
  }
#undef KVQ_LAUNCH_Q
  return {q_i8, q_s};
}

void int8_decode_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor out,
                           torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor block_table, torch::Tensor ks,
                           torch::Tensor vs, double softmax_scale, c10::optional<torch::Tensor> segm_out,
                           c10::optional<torch::Tensor> segm_max, c10::optional<torch::Tensor> segm_sum,
                           int64_t max_query_len, int64_t qmax) {
  const int num_seqs = cu_q.size(0) - 1, Hq = q.size(1), Hkv = kc.size(2), G = Hq / Hkv;
  TORCH_CHECK(q.size(2) == D && kc.stride(3) == 1 && q.stride(2) == 1, "unsupported shape/layout");
  const int bs = kc.size(1);
  TORCH_CHECK(bs >= 16 && (bs & (bs - 1)) == 0, "block_size must be a power of two >= 16");
  int lbs = 0;
  while ((1 << lbs) < bs) ++lbs;
  if (num_seqs == 0) return;
  const bool split = segm_max.has_value();
  const int S = split ? segm_max->size(2) : 1;
  const float sm_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  auto stream = c10::cuda::getCurrentCUDAStream();
  // m tiles for the longest short query: max_query_len is fixed for a captured graph (uniform batch).
  const int mtiles = (G * static_cast<int>(std::min(max_query_len, qmax)) + 15) / 16;
  dim3 grid(num_seqs, Hkv, S * mtiles);
  const int64_t seg_st = split ? segm_max->stride(0) : 0, seg_sh = split ? segm_max->stride(1) : 0;
  float* so = split ? segm_out->data_ptr<float>() : nullptr;
  float* sx = split ? segm_max->data_ptr<float>() : nullptr;
  float* ss = split ? segm_sum->data_ptr<float>() : nullptr;
  const bool pk = ks.size(-1) > 1, pv = vs.size(-1) > 1;
  auto launch = [&](auto kernel) {
    // ~76 KB of dynamic smem (2-stage K/V ring per warp): above the 48 KB default, so opt in.
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DECODE_SMEM));
    kernel<<<grid, NWARPS * 32, DECODE_SMEM, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), so, sx, ss, cu_q.data_ptr<int>(), seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), sm_log2, q.stride(0), q.stride(1),
        out.stride(0), out.stride(1), kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1),
        vc.stride(2), block_table.stride(0), bs, lbs, G, seg_st, seg_sh, S, static_cast<int>(qmax));
  };
  if (pk && pv) launch(int8_decode_attn_kernel<true, true>);
  else if (pk) launch(int8_decode_attn_kernel<true, false>);
  else if (pv) launch(int8_decode_attn_kernel<false, true>);
  else launch(int8_decode_attn_kernel<false, false>);
  if (split) {
    split_reduce_kernel<<<dim3(num_seqs, Hq), D, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), so, sx, ss, cu_q.data_ptr<int>(), out.stride(0),
        out.stride(1), seg_st, seg_sh, S, static_cast<int>(qmax));
  }
}

void dequant_pages(torch::Tensor kc, torch::Tensor vc, torch::Tensor ko, torch::Tensor vo, torch::Tensor src_pages,
                   torch::Tensor ks, torch::Tensor vs) {
  const int n = src_pages.size(0), bs = kc.size(1), H = kc.size(2);
  TORCH_CHECK(kc.size(3) == D && kc.stride(3) == 1 && ko.is_contiguous() && vo.is_contiguous(), "layout");
  if (n == 0) return;
  const bool pk = ks.size(-1) > 1, pv = vs.size(-1) > 1;
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    kernel<<<n, 256, 0, stream>>>(kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(),
                                  reinterpret_cast<__nv_bfloat16*>(ko.data_ptr()),
                                  reinterpret_cast<__nv_bfloat16*>(vo.data_ptr()), src_pages.data_ptr<int>(),
                                  ks.data_ptr<float>(), vs.data_ptr<float>(), kc.stride(0), kc.stride(1), kc.stride(2),
                                  vc.stride(0), vc.stride(1), vc.stride(2), H, bs);
  };
  if (pk && pv) launch(dequant_pages_kernel<true, true>);
  else if (pk) launch(dequant_pages_kernel<true, false>);
  else if (pv) launch(dequant_pages_kernel<false, true>);
  else launch(dequant_pages_kernel<false, false>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_cache", &quant_cache, "fused quantize + reshape_and_cache (int8)");
  m.def("quant_q", &quant_q, "fold k scale into q and quantize");
  m.def("int8_decode_attention", &int8_decode_attention, "INT8 tensor-core paged attention for short queries");
  m.def("dequant_pages", &dequant_pages, "dequantize pages of the INT8 cache into a bf16 paged cache");
}
