// For which shapes of the Whisper decoder's cross-attention output product (m queries x 1500 keys x 64 dims,
// strided batched over `batch` = clips x heads, as CTranslate2 calls it) is cuBLAS's arithmetic one mma.sync
// m16n8k16 chain over the keys with its k tile's residue r first ([0, 16), ... up to r with zeros past r, then
// 16 keys at a time from r)? cuBLAS picks 32-, 64- or 128-key tiles by shape (r = 28, 28, 92); r = 0 is a plain
// chain from key 0. Prints, per shape, the candidates with no mismatch (over enough fills for 400k outputs), and
// the scores' one chain over the dims (alpha 1/8) the same way ("qk ok").
// usage: cross_sweep [max m, default 8]
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
__device__ __forceinline__ unsigned v_pair(const __half* v, int i, int d, int end) {
  const __half lo = i < end ? v[(size_t)i * kD + d] : __float2half(0.f);
  const __half hi = i + 1 < end ? v[(size_t)(i + 1) * kD + d] : __float2half(0.f);
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
__device__ __forceinline__ unsigned p_pair(const __half* p, int m, int j, int i, int end) {
  return j < m && i < end ? *reinterpret_cast<const unsigned*>(p + (size_t)j * kKeys + i) : 0u;
}
__device__ __forceinline__ unsigned r_pair(const __half* x, int rows, int r, int c) {
  return r < rows ? *reinterpret_cast<const unsigned*>(x + (size_t)r * kD + c) : 0u;
}

// Output dims 16 blockIdx.x.. x queries 8 blockIdx.z.. of entry blockIdx.y (dims as A).
__global__ void av_residue(const __half* P, const __half* V, __half* O, int m, int r) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, d0 = blockIdx.x * 16, j0 = blockIdx.z * 8;
  const __half* p = P + (size_t)blockIdx.y * m * kKeys;
  const __half* v = V + (size_t)blockIdx.y * kKeys * kD;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  auto group = [&](int s, int end) {
    const int i = s + 2 * t;
    mma(d, v_pair(v, i, d0 + g, end), v_pair(v, i, d0 + g + 8, end), v_pair(v, i + 8, d0 + g, end),
        v_pair(v, i + 8, d0 + g + 8, end), p_pair(p, m, j0 + g, i, end), p_pair(p, m, j0 + g, i + 8, end));
  };
  for (int s = 0; s < r; s += 16) group(s, r);
  for (int s = r; s < kKeys; s += 16) group(s, kKeys);
  for (int e = 0; e < 4; ++e) {
    const int j = j0 + 2 * t + e % 2, dim = d0 + g + 8 * (e / 2);
    if (j < m) O[((size_t)blockIdx.y * m + j) * kD + dim] = __float2half_rn(d[e]);
  }
}

// Scores of keys 16 blockIdx.x.. x queries 8 blockIdx.z.. (keys as A), one chain over the dims.
__global__ void qk_chain(const __half* Q, const __half* K, __half* C, int m, float alpha) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, i0 = blockIdx.x * 16, j0 = blockIdx.z * 8;
  const __half* q = Q + (size_t)blockIdx.y * m * kD;
  const __half* k = K + (size_t)blockIdx.y * kKeys * kD;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kk = 0; kk < kD; kk += 16)
    mma(d, r_pair(k, kKeys, i0 + g, kk + 2 * t), r_pair(k, kKeys, i0 + g + 8, kk + 2 * t),
        r_pair(k, kKeys, i0 + g, kk + 8 + 2 * t), r_pair(k, kKeys, i0 + g + 8, kk + 8 + 2 * t),
        r_pair(q, m, j0 + g, kk + 2 * t), r_pair(q, m, j0 + g, kk + 8 + 2 * t));
  for (int e = 0; e < 4; ++e) {
    const int i = i0 + g + 8 * (e / 2), j = j0 + 2 * t + e % 2;
    if (j < m && i < kKeys) C[((size_t)blockIdx.y * m + j) * kKeys + i] = __float2half_rn(alpha * d[e]);
  }
}

int main(int argc, char** argv) {
  const int max_m = argc > 1 ? atoi(argv[1]) : 8, residues[] = {28, 92, 0, 220};
  const int batches[] = {20, 40, 60, 80, 100, 120, 140, 160};
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  auto diff = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  for (int m = 1; m <= max_m; m = m < 8 ? m + 1 : m * 2)
    for (int batch : batches) {
      Probe p(batch, m, kKeys, kD);
      __half *P, *V, *O, *R;
      CK(cudaMalloc(&P, 2ull * batch * m * kKeys)); CK(cudaMalloc(&V, 2ull * batch * kKeys * kD));
      CK(cudaMalloc(&O, 2ull * batch * m * kD)); CK(cudaMalloc(&R, 2ull * batch * m * kKeys));
      const int fills = (int)(400000 / ((size_t)batch * m * kD)) + 1;
      unsigned long long bad[4] = {}, qbad = 0;
      for (int f = 0; f < fills; ++f) {
        fill<<<1024, 256>>>(P, (size_t)batch * m * kKeys, 5u + 31 * f + m, -14 + f % 3, -6 + f % 3);
        fill<<<1024, 256>>>(V, (size_t)batch * kKeys * kD, 9u + 17 * f + batch, -6 + f % 3, 2 + f % 3);
        const float one = 1.f, zero = 0.f;
        CK(cublasGemmStridedBatchedEx(p.h, CUBLAS_OP_N, CUBLAS_OP_N, kD, m, kKeys, &one, V, CUDA_R_16F, kD,
                                      (long long)kKeys * kD, P, CUDA_R_16F, kKeys, (long long)m * kKeys, &zero,
                                      O, CUDA_R_16F, kD, (long long)m * kD, batch, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));
        for (int c = 0; c < 4; ++c) {
          av_residue<<<dim3(kD / 16, batch, (m + 7) / 8), 32>>>(P, V, R, m, residues[c]);
          CK(cudaGetLastError()); bad[c] += diff(O, R, (size_t)batch * m * kD);
        }
        if (f < 2) {
          fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * kD, 11u + f, -6 + f, 2 + f);
          fill<<<1024, 256>>>(p.dK, (size_t)batch * kKeys * kD, 97u + f, -7 + f, 1 + f);
          p.cublas_run(0.125f);
          qk_chain<<<dim3((kKeys + 15) / 16, batch, (m + 7) / 8), 32>>>(p.dQ, p.dK, R, m, 0.125f);
          CK(cudaGetLastError()); qbad += diff(p.dC, R, (size_t)batch * m * kKeys);
        }
      }
      printf("m %3d batch %3d fills %4d: av", m, batch, fills);
      bool any = false;
      for (int c = 0; c < 4; ++c)
        if (!bad[c]) { printf(" r%d", residues[c]); any = true; }
      if (!any) printf(" none (%llu %llu %llu %llu)", bad[0], bad[1], bad[2], bad[3]);
      printf(" | qk %s\n", qbad ? "MISMATCH" : "ok");
      fflush(stdout);
      cudaFree(P); cudaFree(V); cudaFree(O); cudaFree(R);
    }
  return 0;
}
