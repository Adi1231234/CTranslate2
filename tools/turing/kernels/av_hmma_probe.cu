// Which arithmetic does cuBLAS use for the Whisper encoder's attention output on this GPU? C[b][j][d] =
// sum_i P[b][j][i] V[b][i][d] (MatMul, fp16, COMPUTE_32F, 1500 queries x 1500 keys x 64 dims, as CTranslate2
// calls it), compared bit for bit with mma.sync m16n8k16 chains (f32 from zero) over the keys in 16-key groups:
//   0: groups from key 0, the last one partial (zeros past key 1499)
//   1: residue first by 64-key tiles: [0, 16), [16, 28) + zeros, then groups from key 28 on
//   2: as 1, plus the tile's two all-zero groups [32, 48) and [48, 64) as mma steps
// usage: av_hmma_probe [batch, default 160] -> mismatches per candidate over three fills
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kKeys = 1500, kDims = 64;

__device__ __forceinline__ unsigned p_pair(const __half* p, int m, int j, int i) {     // P[j][i], P[j][i + 1]
  return j < m && i < kKeys ? *reinterpret_cast<const unsigned*>(p + (size_t)j * kKeys + i) : 0u;
}
__device__ __forceinline__ unsigned v_pair(const __half* v, int i, int d) {             // V[i][d], V[i + 1][d]
  const __half lo = i < kKeys ? v[(size_t)i * kDims + d] : __float2half(0.f);
  const __half hi = i + 1 < kKeys ? v[(size_t)(i + 1) * kDims + d] : __float2half(0.f);
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}

// One 16-key group starting at key i0; keys at or past `end` count as zero.
__device__ __forceinline__ void group(float* acc, const __half* p, const __half* v, int m, int j0, int d0,
                                      int i0, int end, int g, int t) {
  auto pp = [&](int j, int i) { return i < end ? p_pair(p, m, j, i) : 0u; };
  auto vp = [&](int i, int d) { return i < end ? v_pair(v, i, d) : 0u; };
  const unsigned a0 = pp(j0 + g, i0 + 2 * t), a1 = pp(j0 + g + 8, i0 + 2 * t);
  const unsigned a2 = pp(j0 + g, i0 + 8 + 2 * t), a3 = pp(j0 + g + 8, i0 + 8 + 2 * t);
  const unsigned b0 = vp(i0 + 2 * t, d0 + g), b1 = vp(i0 + 8 + 2 * t, d0 + g);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void av_candidate(const __half* P, const __half* V, __half* C, int m, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, j0 = blockIdx.y * 16, d0 = blockIdx.x * 8;
  const __half* p = P + (size_t)blockIdx.z * m * kKeys;
  const __half* v = V + (size_t)blockIdx.z * kKeys * kDims;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (mode == 0) {
    for (int i0 = 0; i0 < kKeys; i0 += 16) group(acc, p, v, m, j0, d0, i0, kKeys, g, t);
  } else {
    const int residue = kKeys % 64;                                     // 28
    group(acc, p, v, m, j0, d0, 0, residue, g, t);
    group(acc, p, v, m, j0, d0, 16, residue, g, t);
    if (mode == 2) { group(acc, p, v, m, j0, d0, 32, residue, g, t); group(acc, p, v, m, j0, d0, 48, residue, g, t); }
    for (int i0 = residue; i0 < kKeys; i0 += 16) group(acc, p, v, m, j0, d0, i0, kKeys, g, t);
  }
  for (int e = 0; e < 4; ++e) {
    const int j = j0 + g + 8 * (e / 2), d = d0 + 2 * t + e % 2;
    if (j < m) C[((size_t)blockIdx.z * m + j) * kDims + d] = __float2half_rn(acc[e]);
  }
}

int main(int argc, char** argv) {
  const int batch = argc > 1 ? atoi(argv[1]) : 160, m = 1500;
  __half *P, *V, *C, *R;
  CK(cudaMalloc(&P, 2ull * batch * m * kKeys)); CK(cudaMalloc(&V, 2ull * batch * kKeys * kDims));
  CK(cudaMalloc(&C, 2ull * batch * m * kDims)); CK(cudaMalloc(&R, 2ull * batch * m * kDims));
  cublasHandle_t h; CK(cublasCreate(&h));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long bad[3] = {0, 0, 0};
  for (int f = 0; f < 3; ++f) {
    fill<<<1024, 256>>>(P, (size_t)batch * m * kKeys, 5u + f, -14 + f, -6 + f);
    fill<<<1024, 256>>>(V, (size_t)batch * kKeys * kDims, 9u + f, -6 + f, 2 + f);
    const float alpha = 1.f, beta = 0.f;
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kDims, m, kKeys, &alpha,
                                  V, CUDA_R_16F, kDims, (long long)kKeys * kDims, P, CUDA_R_16F, kKeys,
                                  (long long)m * kKeys, &beta, C, CUDA_R_16F, kDims, (long long)m * kDims,
                                  batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    for (int mode = 0; mode < 3; ++mode) {
      av_candidate<<<dim3(kDims / 8, (m + 15) / 16, batch), 32>>>(P, V, R, m, mode);
      CK(cudaGetLastError());
      CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(C, R, (size_t)batch * m * kDims, dc);
      unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad[mode] += d;
    }
  }
  for (int mode = 0; mode < 3; ++mode)
    printf("candidate %d: %llu of %llu mismatched\n", mode, bad[mode], 3ull * batch * m * kDims);
  return 0;
}
