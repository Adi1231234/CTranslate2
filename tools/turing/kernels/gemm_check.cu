// Bit-for-bit check and timing of src/cuda/small_m_gemm.cuh against the cuBLAS call that
// CTranslate2 makes for its decoder Dense layers, on every routed shape (rows 5..40 by 5 x the
// decoder's N x K): four random fills with different ranges, and one where each split-K quarter has
// a single +-2^e product so that the order of the partial sums decides the result.
// usage: gemm_check      -> per-shape mismatches and times; must print TOTAL 0
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "small_m_gemm.cuh"

using namespace ctranslate2::cuda;

void extremes(__half* A, __half* W, int M, int N, int K, uint32_t seed) {
  std::vector<__half> a((size_t)M * K, __float2half(0.f)), w((size_t)N * K, __float2half(0.f));
  for (int m = 0; m < M; ++m)
    for (int q = 0; q < 4; ++q) a[(size_t)m * K + q * (K / 4) + (m % 7) * 8] = __float2half(1.f);
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < 7; ++m)
      for (int q = 0; q < 4; ++q) {
        seed = seed * 1664525u + 1013904223u;
        w[(size_t)n * K + q * (K / 4) + m * 8] = __float2half(((seed >> 31) ? -1.f : 1.f) * ldexpf(1.f, (int)((seed >> 8) % 40) - 24));
      }
  CK(cudaMemcpy(A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(W, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
}

template <typename F> float time_us(F run, int reps = 100) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main() {
  const int nk[4][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {1280, 5120}};
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *C, *R;
  CK(cudaMalloc(&A, 2ull * 40 * 5120)); CK(cudaMalloc(&W, 2ull * 5120 * 5120));
  CK(cudaMalloc(&C, 2ull * 40 * 5120)); CK(cudaMalloc(&R, 2ull * 40 * 5120));
  unsigned long long* dcount; CK(cudaMalloc(&dcount, 8));
  unsigned long long total = 0;
  for (int M = 5; M <= 40; M += 5)
    for (auto& s : nk) {
      const int N = s[0], K = s[1];
      const int recipe = small_m_gemm_recipe(M, N, K);
      auto ref = [&] {
        const float alpha = 1.f, beta = 0.f;
        CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                        &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      };
      auto mine = [&] { small_m_gemm(A, W, R, M, N, K, 0); };
      unsigned long long bad = 0;
      for (int t = 0; t < 5; ++t) {
        if (t < 4) {
          fill<<<256, 256>>>(A, (size_t)M * K, 13u * t + M, -8 + t, 2 + t);
          fill<<<256, 256>>>(W, (size_t)N * K, 101u * t + N + K, -14 + t, -2 + t);
        } else {
          extremes(A, W, M, N, K, 7u * M + N);
        }
        ref(); mine();
        CK(cudaGetLastError());
        CK(cudaMemset(dcount, 0, 8)); count_diff<<<256, 256>>>(C, R, (size_t)M * N, dcount);
        unsigned long long d; CK(cudaMemcpy(&d, dcount, 8, cudaMemcpyDeviceToHost)); bad += d;
      }
      total += bad;
      printf("M %2d N %4d K %4d recipe %d: %llu mismatches, cuBLAS %.1f us, replica %.1f us\n",
             M, N, K, recipe, bad, time_us(ref), time_us(mine));
    }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
