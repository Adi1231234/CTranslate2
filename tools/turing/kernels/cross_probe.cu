// Which arithmetic does cuBLAS use for the Whisper decoder's cross-attention on this GPU? m queries (the beams)
// against 1500 keys of 64 dims per batch entry, as CTranslate2 calls it (strided batched, fp16, COMPUTE_32F):
// scores C = alpha q k^T (cutlass_80_wmma 32x32_64x1_tn on sm_120) and output O = P v (32x32_32x1_nn), compared bit
// for bit with candidates built from mma.sync m16n8k16 (f32 from zero). Scores (alpha 1/8):
//   q0: keys as A, one chain over the 4 dim groups          q1: queries as A, one chain
//   q2: 4 dim groups each from zero, summed in order         q3: two chains (dims 0-31, 32-63), summed
//   q4: 4 groups from zero, summed (g0 + g1) + (g2 + g3)
// Output (dims as A, queries as B unless noted):
//   a0: one chain, 32-key tiles' residue first ([0, 16), [16, 28) + zeros, then from 28)
//   a1: one chain from key 0, the last group partial        a2: as a0 with the queries as A
//   a3: residue-first tiles, each tile's two 16-key halves in two chains, summed at the end
//   a4: as a3 without the residue (tiles from key 0)
// usage: cross_probe [m, default 5] [batch, default 160] -> mismatches per candidate over three fills
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kKeys = 1500, kD = 64;

__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned row_pair(const __half* x, int rows, int r, int c) {   // x[r][c], x[r][c + 1]
  return r < rows ? *reinterpret_cast<const unsigned*>(x + (size_t)r * kD + c) : 0u;
}

// Scores of 16 keys (i0..) x 8 queries (keys as A), or of 16 queries x 8 keys (i0.., q1). One warp.
__global__ void qk_candidate(const __half* Q, const __half* K, __half* C, int m, float alpha, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, i0 = blockIdx.x * (mode == 1 ? 8 : 16);
  const __half* q = Q + (size_t)blockIdx.y * m * kD;
  const __half* k = K + (size_t)blockIdx.y * kKeys * kD;
  float part[4][4] = {}, d[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < 4; ++c) {
    const int kk = 16 * c;
    float* acc = mode == 2 || mode == 4 ? part[c] : mode == 3 ? part[c / 2] : d;
    if (mode == 1)   // queries as A: rows j = g, g + 8 (m <= 16), keys i0 + g as the n side
      mma(acc, row_pair(q, m, g, kk + 2 * t), row_pair(q, m, g + 8, kk + 2 * t), row_pair(q, m, g, kk + 8 + 2 * t),
          row_pair(q, m, g + 8, kk + 8 + 2 * t), row_pair(k, kKeys, i0 + g, kk + 2 * t),
          row_pair(k, kKeys, i0 + g, kk + 8 + 2 * t));
    else
      mma(acc, row_pair(k, kKeys, i0 + g, kk + 2 * t), row_pair(k, kKeys, i0 + g + 8, kk + 2 * t),
          row_pair(k, kKeys, i0 + g, kk + 8 + 2 * t), row_pair(k, kKeys, i0 + g + 8, kk + 8 + 2 * t),
          row_pair(q, m, g, kk + 2 * t), row_pair(q, m, g, kk + 8 + 2 * t));
  }
  for (int e = 0; e < 4; ++e) {
    if (mode == 2) d[e] = ((part[0][e] + part[1][e]) + part[2][e]) + part[3][e];
    if (mode == 3) d[e] = part[0][e] + part[1][e];
    if (mode == 4) d[e] = (part[0][e] + part[1][e]) + (part[2][e] + part[3][e]);
    const int j = mode == 1 ? g + 8 * (e / 2) : 2 * t + e % 2;           // query and key of d[e]
    const int i = mode == 1 ? i0 + 2 * t + e % 2 : i0 + g + 8 * (e / 2);
    if (j < m && i < kKeys) C[((size_t)blockIdx.y * m + j) * kKeys + i] = __float2half_rn(alpha * d[e]);
  }
}

__device__ __forceinline__ unsigned v_pair(const __half* v, int i, int d, int end) {       // V[i][d], V[i + 1][d]
  const __half lo = i < end ? v[(size_t)i * kD + d] : __float2half(0.f);
  const __half hi = i + 1 < end ? v[(size_t)(i + 1) * kD + d] : __float2half(0.f);
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
__device__ __forceinline__ unsigned p_pair(const __half* p, int m, int j, int i, int end) {
  return j < m && i < end ? *reinterpret_cast<const unsigned*>(p + (size_t)j * kKeys + i) : 0u;
}

// Output dims d0..d0+15 x 8 queries (dims as A), or 16 queries x dims d0..d0+7 (a2). One warp.
__global__ void av_candidate(const __half* P, const __half* V, __half* O, int m, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, d0 = blockIdx.x * (mode == 2 ? 8 : 16);
  const __half* p = P + (size_t)blockIdx.y * m * kKeys;
  const __half* v = V + (size_t)blockIdx.y * kKeys * kD;
  float part[2][4] = {}, d[4] = {0.f, 0.f, 0.f, 0.f};
  const int residue = mode == 1 || mode == 4 ? 0 : kKeys % 32;
  auto group = [&](float* acc, int s, int end) {
    const int i = s + 2 * t;
    if (mode == 2)
      mma(acc, p_pair(p, m, g, i, end), p_pair(p, m, g + 8, i, end), p_pair(p, m, g, i + 8, end),
          p_pair(p, m, g + 8, i + 8, end), v_pair(v, i, d0 + g, end), v_pair(v, i + 8, d0 + g, end));
    else
      mma(acc, v_pair(v, i, d0 + g, end), v_pair(v, i, d0 + g + 8, end), v_pair(v, i + 8, d0 + g, end),
          v_pair(v, i + 8, d0 + g + 8, end), p_pair(p, m, g, i, end), p_pair(p, m, g, i + 8, end));
  };
  const bool sliced = mode == 3 || mode == 4;
  if (residue) { group(sliced ? part[0] : d, 0, residue); group(sliced ? part[1] : d, 16, residue); }
  for (int s = residue, h = 0; s < kKeys; s += 16, h ^= 1) group(sliced ? part[h] : d, s, kKeys);
  for (int e = 0; e < 4; ++e) {
    if (sliced) d[e] = part[0][e] + part[1][e];
    const int j = mode == 2 ? g + 8 * (e / 2) : 2 * t + e % 2;           // query and dim of d[e]
    const int dim = mode == 2 ? d0 + 2 * t + e % 2 : d0 + g + 8 * (e / 2);
    if (j < m) O[((size_t)blockIdx.y * m + j) * kD + dim] = __float2half_rn(d[e]);
  }
}

int main(int argc, char** argv) {
  const int m = argc > 1 ? atoi(argv[1]) : 5, batch = argc > 2 ? atoi(argv[2]) : 160;
  Probe p(batch, m, kKeys, kD);                                    // Q [batch][m][64], K [batch][1500][64]
  __half *P, *V, *O, *R;
  CK(cudaMalloc(&P, 2ull * batch * m * kKeys)); CK(cudaMalloc(&V, 2ull * batch * kKeys * kD));
  CK(cudaMalloc(&O, 2ull * batch * m * kD)); CK(cudaMalloc(&R, 2ull * batch * m * kKeys));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long qbad[5] = {}, abad[5] = {};
  auto diff = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  for (int f = 0; f < 3; ++f) {
    fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * kD, 11u + f, -6 + f, 2 + f);
    fill<<<1024, 256>>>(p.dK, (size_t)batch * kKeys * kD, 97u + f, -7 + f, 1 + f);
    p.cublas_run(0.125f);
    for (int mode = 0; mode < 5; ++mode) {
      CK(cudaMemset(R, 0, 2ull * batch * m * kKeys));
      qk_candidate<<<dim3(mode == 1 ? (kKeys + 7) / 8 : (kKeys + 15) / 16, batch), 32>>>(p.dQ, p.dK, R, m, 0.125f,
                                                                                         mode);
      CK(cudaGetLastError()); qbad[mode] += diff(p.dC, R, (size_t)batch * m * kKeys);
    }
    fill<<<1024, 256>>>(P, (size_t)batch * m * kKeys, 5u + f, -14 + f, -6 + f);
    fill<<<1024, 256>>>(V, (size_t)batch * kKeys * kD, 9u + f, -6 + f, 2 + f);
    const float alpha = 1.f, beta = 0.f;
    CK(cublasGemmStridedBatchedEx(p.h, CUBLAS_OP_N, CUBLAS_OP_N, kD, m, kKeys, &alpha, V, CUDA_R_16F, kD,
                                  (long long)kKeys * kD, P, CUDA_R_16F, kKeys, (long long)m * kKeys, &beta,
                                  O, CUDA_R_16F, kD, (long long)m * kD, batch, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    for (int mode = 0; mode < 5; ++mode) {
      av_candidate<<<dim3(mode == 2 ? kD / 8 : kD / 16, batch), 32>>>(P, V, R, m, mode);
      CK(cudaGetLastError()); abad[mode] += diff(O, R, (size_t)batch * m * kD);
    }
  }
  for (int i = 0; i < 5; ++i) printf("scores q%d: %llu of %llu mismatched\n", i, qbad[i], 3ull * batch * m * kKeys);
  for (int i = 0; i < 5; ++i) printf("output a%d: %llu of %llu mismatched\n", i, abad[i], 3ull * batch * m * kD);
  return 0;
}
