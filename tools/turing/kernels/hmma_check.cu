// Bit-for-bit check and timing of the library's decoder Dense replica (src/cuda/hmma_gemm.cuh) against the
// cuBLAS call CTranslate2 makes (C = A W^T, fp16, COMPUTE_32F, alpha 1, beta 0), on every routed shape
// (rows 2..48 x the decoder's N x K): six random fills with different ranges per shape, for each block width
// (1, 2, 4, 8 warps of 16 columns). Prints each shape's mismatches and times (weights read from DRAM, as in
// production); must end with TOTAL 0.
// usage: hmma_check [timing repetitions, default 50] [rows step, default 1]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/hmma_gemm.cuh"

using namespace ctranslate2::cuda;

template <typename F> float time_us(F run, int reps) {         // run(i): repetition i
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(0); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run(i);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 50, step = argc > 2 ? atoi(argv[2]) : 1;
  const int nk[5][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {1280, 5120}, {51866, 1280}};
  const int widths[4] = {1, 2, 4, 8};
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *C, *R; void* P;
  CK(cudaMalloc(&A, 2ull * 48 * 5120));
  const size_t pool = 2ull * 51866 * 1280;                 // >= 96 MB: timed runs rotate through weight
  CK(cudaMalloc(&W, pool));                                // copies, so reads miss the 32 MB L2
  fill<<<4096, 256>>>(W, pool / 2, 7u, -10, 0);
  CK(cudaMalloc(&C, 2ull * 48 * 51866)); CK(cudaMalloc(&R, 2ull * 48 * 51866));
  CK(cudaMalloc(&P, hmma_gemm_workspace_bytes(48, 1280, 5120)));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0;
  double sum_c = 0, sum_w[4] = {};
  for (auto& s : nk)
    for (int M = 2; M <= 48; M += step) {
      const int N = s[0], K = s[1], recipe = hmma_gemm_recipe(M, N, K);
      if (recipe == 0)
        continue;
      const size_t copies = pool / (2ull * N * K);                  // as production: weights come from DRAM
      auto w = [&](int i) { return W + (size_t)(i % copies) * N * K; };
      auto ref = [&](int i) {
        const float alpha = 1.f, beta = 0.f;
        CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w(i), CUDA_R_16F, K, A, CUDA_R_16F, K,
                        &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      };
      const float tc = time_us(ref, reps);
      printf("M %2d N %5d K %4d r%d: cuBLAS %6.1f us", M, N, K, recipe, tc);
      sum_c += tc;
      for (int x = 0; x < 4; ++x) {
        auto mine = [&](int i) { hmma_gemm(A, w(i), R, M, N, K, P, 0, widths[x]); };
        unsigned long long bad = 0;
        for (int f = 0; f < 6; ++f) {
          fill<<<256, 256>>>(A, (size_t)M * K, 29u * f + M, -12 + 2 * f, 1 + f);
          fill<<<1024, 256>>>(W, (size_t)N * K, 131u * f + N + K, -16 + f, -4 + 2 * f);
          ref(0); mine(0);
          CK(cudaGetLastError());
          CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(C, R, (size_t)M * N, dc);
          unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad += d;
        }
        total += bad;
        const float t = time_us(mine, reps);
        sum_w[x] += t;
        printf(" | w%d %6.1f us%s", widths[x], t, bad ? " MISMATCH" : "");
      }
      printf("\n");
    }
  printf("summed: cuBLAS %.0f us", sum_c);
  for (int x = 0; x < 4; ++x)
    printf(", w%d %.0f us (%.2fx)", widths[x], sum_w[x], sum_c / sum_w[x]);
  printf("\nTOTAL %llu mismatches\n", total);
  return 0;
}
