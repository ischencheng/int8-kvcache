// CUDA kernels for the static INT8 KV cache (stage 7, CUDA version).
//
//   quant_cache        fused "quantize + reshape_and_cache" for one step's new K/V
//   quant_q            fold the per-channel K scale into q and quantize q (standalone variant)
//   int8_decode_attn   decode paged attention on the INT8 cache with INT8 tensor cores
//                      (mma.sync.m16n8k32.s8), optional split-KV + reduce
//
// Numerics are those of kvq/int8_ref.py:attend_int8 (implementation B):
//   q' = q * s_k[kvh, :] ; q_i8 = rint(q' / s_q) with one s_q per (token, q head)
//   s  = (q_i8 . k_i8) * s_q * softmax_scale            (int32 accumulate)
//   P  = exp2(s - m) quantized per (row, 32-token chunk): p_i8 = rint(P / (rowmax / 127))
//   o  = sum_chunks (p_i8 . v_i8) * rowmax / 127, divided by the sum of the quantized P, times s_v[d]
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: that pulls cuSPARSE/cuBLAS headers
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int D = 128;        // head_dim this kernel is written for
constexpr int CHUNK = 32;     // tokens a warp processes per iteration (= k of the PV mma)
constexpr int NWARPS = 4;
constexpr int KROW = D + 16;  // smem K row stride in bytes: 144 -> the B-fragment loads are conflict free
constexpr int VROW = CHUNK + 4;  // smem V^T row stride in bytes (36)
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
// INT8 decode attention. Grid (num_seqs, num_kv_heads, num_splits), 4 warps.
// Rows of the m16 tile = the G q heads of this kv head (G <= 16; rows >= G are zero and unused).
// ------------------------------------------------------------------------------------------
struct SmemLayout {
  int8_t q[16 * QROW];
  float q_scale[16];
  float m[NWARPS][16];
  float l[NWARPS][16];
  union {
    struct {
      int8_t k[CHUNK * KROW];
      int8_t vt[D * VROW];
    } w[NWARPS];
    float acc[16][D];  // reused for the cross-warp combine after the main loop
  } u;
};

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
                            int G, int64_t seg_st, int64_t seg_sh) {
  __shared__ SmemLayout sm;
  const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z, num_splits = gridDim.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tok = cu_q[seq];
  if (cu_q[seq + 1] - tok != 1) return;  // decode rows only (padding requests have 0 tokens)
  const int L = seq_lens[seq];

  // ---- prologue: fold s_k into q, quantize per row (8 threads per row, 16 channels each) ----
  {
    const int row = threadIdx.x >> 3, part = threadIdx.x & 7, d0 = part * 16;
    float x[16], amax = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      x[e] = 0.f;
      if (row < G) {
        x[e] = __bfloat162float(q[tok * q_st + (kvh * G + row) * q_sh + d0 + e]);
        if (PC_K) x[e] *= ks[kvh * D + d0 + e];
      }
      amax = fmaxf(amax, fabsf(x[e]));
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float s = fmaxf(amax / 127.f, 1e-8f);
    alignas(16) int8_t qq[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) qq[e] = sat_i8(__fdiv_rn(x[e], s));
    *reinterpret_cast<uint4*>(&sm.q[row * QROW + d0]) = *reinterpret_cast<uint4*>(qq);
    if (part == 0) sm.q_scale[row] = s * (PC_K ? 1.f : ks[kvh]) * sm_scale_log2;
  }
  __syncthreads();
  uint32_t a[4][4];  // q A fragments for the 4 k-steps (32 channels each)
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    a[s][0] = *reinterpret_cast<const uint32_t*>(&sm.q[g * QROW + 32 * s + 4 * t]);
    a[s][1] = *reinterpret_cast<const uint32_t*>(&sm.q[(g + 8) * QROW + 32 * s + 4 * t]);
    a[s][2] = *reinterpret_cast<const uint32_t*>(&sm.q[g * QROW + 32 * s + 16 + 4 * t]);
    a[s][3] = *reinterpret_cast<const uint32_t*>(&sm.q[(g + 8) * QROW + 32 * s + 16 + 4 * t]);
  }
  const float rs0 = sm.q_scale[g], rs1 = sm.q_scale[g + 8];

  // ---- this split's chunks; warp w takes chunks w, w + 4, ... ----
  const int nchunks = (L + CHUNK - 1) / CHUNK;
  const int per_split = (nchunks + num_splits - 1) / num_splits;
  const int c_begin = split * per_split, c_end = min(nchunks, c_begin + per_split);
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  int8_t* sk = sm.u.w[warp].k;
  int8_t* svt = sm.u.w[warp].vt;
  const int* bt = block_table + static_cast<int64_t>(seq) * bt_stride;

  for (int c = c_begin + warp; c < c_end; c += NWARPS) {
    // Load the chunk: 8 lanes per token row (16 B each), 4 tokens per instruction -> coalesced.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int tau = (lane >> 3) + 4 * i, part = lane & 7;
      const int n = c * CHUNK + tau;
      uint4 kr = make_uint4(0, 0, 0, 0), vr = make_uint4(0, 0, 0, 0);
      if (n < L) {
        const int64_t blk = bt[n / block_size];
        const int off = n % block_size;
        kr = *reinterpret_cast<const uint4*>(kc + blk * k_sb + off * k_sn + kvh * k_sh + part * 16);
        vr = *reinterpret_cast<const uint4*>(vc + blk * v_sb + off * v_sn + kvh * v_sh + part * 16);
      }
      *reinterpret_cast<uint4*>(&sk[tau * KROW + part * 16]) = kr;
      const int8_t* vb = reinterpret_cast<const int8_t*>(&vr);
      const int col = slot_of_token(tau);
#pragma unroll
      for (int e = 0; e < 16; ++e) svt[(part * 16 + e) * VROW + col] = vb[e];
    }
    __syncwarp();

    // QK: 4 n-tiles of 8 tokens x 4 k-steps of 32 channels.
    int sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0;
      const int8_t* krow = &sk[(8 * j + g) * KROW + 4 * t];
#pragma unroll
      for (int s = 0; s < 4; ++s)
        mma_s8(sacc[j], a[s], *reinterpret_cast<const uint32_t*>(krow + 32 * s),
               *reinterpret_cast<const uint32_t*>(krow + 32 * s + 16));
    }
    // Scores of the 8 tokens this thread holds, rows g and g + 8 (log2 domain).
    float s0[8], s1[8], mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const bool ok = c * CHUNK + 8 * j + 2 * t + e < L;
        s0[2 * j + e] = ok ? sacc[j][e] * rs0 : -INFINITY;
        s1[2 * j + e] = ok ? sacc[j][2 + e] * rs1 : -INFINITY;
        mx0 = fmaxf(mx0, s0[2 * j + e]);
        mx1 = fmaxf(mx1, s1[2 * j + e]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    float p0[8], p1[8], pm0 = 0.f, pm1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      p0[i] = exp2f(s0[i] - mn0);
      p1[i] = exp2f(s1[i] - mn1);
      pm0 = fmaxf(pm0, p0[i]);
      pm1 = fmaxf(pm1, p1[i]);
    }
    pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 1));
    pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 2));
    pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 1));
    pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 2));
    const float ps0 = fmaxf(pm0 / 127.f, 1e-30f), ps1 = fmaxf(pm1 / 127.f, 1e-30f);
    // P -> int8 straight into the PV A fragment (slot order = token order of the QK accumulator).
    alignas(16) int8_t q0[8], q1[8];
    float qs0 = 0.f, qs1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int v0 = __float2int_rn(p0[i] / ps0), v1 = __float2int_rn(p1[i] / ps1);
      q0[i] = static_cast<int8_t>(min(v0, 127));
      q1[i] = static_cast<int8_t>(min(v1, 127));
      qs0 += q0[i];
      qs1 += q1[i];
    }
    qs0 += __shfl_xor_sync(0xffffffff, qs0, 1);
    qs0 += __shfl_xor_sync(0xffffffff, qs0, 2);
    qs1 += __shfl_xor_sync(0xffffffff, qs1, 1);
    qs1 += __shfl_xor_sync(0xffffffff, qs1, 2);
    l0 = l0 * al0 + qs0 * ps0;
    l1 = l1 * al1 + qs1 * ps1;
    uint32_t pa[4];
    pa[0] = *reinterpret_cast<const uint32_t*>(&q0[0]);  // row g,     slots 4t..4t+3   (tiles 0, 1)
    pa[1] = *reinterpret_cast<const uint32_t*>(&q1[0]);  // row g + 8, slots 4t..4t+3
    pa[2] = *reinterpret_cast<const uint32_t*>(&q0[4]);  // row g,     slots 16+4t..    (tiles 2, 3)
    pa[3] = *reinterpret_cast<const uint32_t*>(&q1[4]);  // row g + 8, slots 16+4t..
    // PV: 16 n-tiles of 8 channels, one k-step of 32 (permuted) tokens.
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      int pacc[4] = {0, 0, 0, 0};
      const int8_t* vrow = &svt[(8 * j + g) * VROW + 4 * t];
      mma_s8(pacc, pa, *reinterpret_cast<const uint32_t*>(vrow), *reinterpret_cast<const uint32_t*>(vrow + 16));
      acc[j][0] = acc[j][0] * al0 + pacc[0] * ps0;
      acc[j][1] = acc[j][1] * al0 + pacc[1] * ps0;
      acc[j][2] = acc[j][2] * al1 + pacc[2] * ps1;
      acc[j][3] = acc[j][3] * al1 + pacc[3] * ps1;
    }
    m0 = mn0;
    m1 = mn1;
    __syncwarp();
  }

  // ---- combine the 4 warps: weights exp2(m_w - M) ----
  if (t == 0) {
    sm.m[warp][g] = m0;
    sm.m[warp][g + 8] = m1;
    sm.l[warp][g] = l0;
    sm.l[warp][g + 8] = l1;
  }
  __syncthreads();  // also ends every use of the per-warp K/V buffers
  float M0 = -1e30f, M1 = -1e30f;
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) {
    M0 = fmaxf(M0, sm.m[w][g]);
    M1 = fmaxf(M1, sm.m[w][g + 8]);
  }
  const float w0 = exp2f(m0 - M0), w1 = exp2f(m1 - M1);
  for (int i = threadIdx.x; i < 16 * D; i += blockDim.x) (&sm.u.acc[0][0])[i] = 0.f;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    atomicAdd(&sm.u.acc[g][8 * j + 2 * t], acc[j][0] * w0);
    atomicAdd(&sm.u.acc[g][8 * j + 2 * t + 1], acc[j][1] * w0);
    atomicAdd(&sm.u.acc[g + 8][8 * j + 2 * t], acc[j][2] * w1);
    atomicAdd(&sm.u.acc[g + 8][8 * j + 2 * t + 1], acc[j][3] * w1);
  }
  __syncthreads();

  // ---- epilogue: 8 threads per row, 16 channels each; s_v applied here ----
  const int row = threadIdx.x >> 3, d0 = (threadIdx.x & 7) * 16;
  if (row >= G) return;
  float Mr = -1e30f, Lr = 0.f;
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) Mr = fmaxf(Mr, sm.m[w][row]);
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) Lr += sm.l[w][row] * exp2f(sm.m[w][row] - Mr);
  const int head = kvh * G + row;
  if (num_splits == 1) {
    const float inv = 1.f / fmaxf(Lr, 1e-30f);
    alignas(16) __nv_bfloat16 o[16];
#pragma unroll
    for (int e = 0; e < 16; ++e)
      o[e] = __float2bfloat16(sm.u.acc[row][d0 + e] * inv * (PC_V ? vs[kvh * D + d0 + e] : vs[kvh]));
    *reinterpret_cast<uint4*>(out + tok * o_st + head * o_sh + d0) = *reinterpret_cast<uint4*>(&o[0]);
    *reinterpret_cast<uint4*>(out + tok * o_st + head * o_sh + d0 + 8) = *reinterpret_cast<uint4*>(&o[8]);
  } else {
    const int64_t sidx = tok * seg_st + head * seg_sh + split;
    if ((threadIdx.x & 7) == 0) {
      segm_max[sidx] = Mr;
      segm_sum[sidx] = Lr;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e)
      segm_out[sidx * D + d0 + e] = sm.u.acc[row][d0 + e] * (PC_V ? vs[kvh * D + d0 + e] : vs[kvh]);
  }
}

// Merge split-KV partials: o = sum_s 2^(m_s - M) acc_s / sum_s 2^(m_s - M) l_s.
__global__ void split_reduce_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ segm_out,
                                    const float* __restrict__ segm_max, const float* __restrict__ segm_sum,
                                    const int* __restrict__ cu_q, int64_t o_st, int64_t o_sh, int64_t seg_st,
                                    int64_t seg_sh, int S) {
  const int seq = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int tok = cu_q[seq];
  if (cu_q[seq + 1] - tok != 1) return;
  const int64_t base = tok * seg_st + h * seg_sh;
  float M = -1e30f;
  for (int s = 0; s < S; ++s) M = fmaxf(M, segm_max[base + s]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < S; ++s) {
    const float w = exp2f(segm_max[base + s] - M);
    num += w * segm_out[(base + s) * D + d];
    den += w * segm_sum[base + s];
  }
  out[tok * o_st + h * o_sh + d] = __float2bfloat16(num / fmaxf(den, 1e-30f));
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
                           c10::optional<torch::Tensor> segm_max, c10::optional<torch::Tensor> segm_sum) {
  const int num_seqs = cu_q.size(0) - 1, Hq = q.size(1), Hkv = kc.size(2), G = Hq / Hkv;
  TORCH_CHECK(q.size(2) == D && G <= 16 && kc.stride(3) == 1 && q.stride(2) == 1, "unsupported shape/layout");
  TORCH_CHECK(kc.size(1) % 16 == 0, "block_size must be a multiple of 16");
  if (num_seqs == 0) return;
  const bool split = segm_max.has_value();
  const int S = split ? segm_max->size(2) : 1;
  const float sm_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(num_seqs, Hkv, S);
  const int64_t seg_st = split ? segm_max->stride(0) : 0, seg_sh = split ? segm_max->stride(1) : 0;
  float* so = split ? segm_out->data_ptr<float>() : nullptr;
  float* sx = split ? segm_max->data_ptr<float>() : nullptr;
  float* ss = split ? segm_sum->data_ptr<float>() : nullptr;
  const bool pk = ks.size(-1) > 1, pv = vs.size(-1) > 1;
  auto launch = [&](auto kernel) {
    kernel<<<grid, NWARPS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), so, sx, ss, cu_q.data_ptr<int>(), seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), sm_log2, q.stride(0), q.stride(1),
        out.stride(0), out.stride(1), kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1),
        vc.stride(2), block_table.stride(0), kc.size(1), G, seg_st, seg_sh);
  };
  if (pk && pv) launch(int8_decode_attn_kernel<true, true>);
  else if (pk) launch(int8_decode_attn_kernel<true, false>);
  else if (pv) launch(int8_decode_attn_kernel<false, true>);
  else launch(int8_decode_attn_kernel<false, false>);
  if (split) {
    split_reduce_kernel<<<dim3(num_seqs, Hq), D, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), so, sx, ss, cu_q.data_ptr<int>(), out.stride(0),
        out.stride(1), seg_st, seg_sh, S);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_cache", &quant_cache, "fused quantize + reshape_and_cache (int8)");
  m.def("quant_q", &quant_q, "fold k scale into q and quantize");
  m.def("int8_decode_attention", &int8_decode_attention, "INT8 tensor-core decode paged attention");
}
