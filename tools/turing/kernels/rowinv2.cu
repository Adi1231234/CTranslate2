// Can one decoder product serve several batches with the bits each batch alone would get? For one Dense shape
// (CTranslate2's cublasGemmEx: C[m][n] = sum_k A[m][k] W[n][k], fp16, COMPUTE_32F, default algorithm and
// workspace), every row of one call with M rows (M = 1..320) against the same row computed by a call of 40 rows
// (the largest batch of 8 clips x 5 beams; for the shapes listed in clip_groups.cc every row count 2..48 runs one
// chain over k on the L40S, hmma_probe.cu), on 3 random fills. Prints the row counts with any mismatch.
// usage: rowinv2 N K
#include "probe_common.h"
#include "probe_data.cuh"

static const int MMAX = 320, MREF = 40, FILLS = 3;

int main(int argc, char** argv) {
  const int N = atoi(argv[1]), K = atoi(argv[2]);
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *REF, *OUT; unsigned long long* dc;
  CK(cudaMalloc(&A, 2ull * MMAX * K)); CK(cudaMalloc(&W, 2ull * N * K));
  CK(cudaMalloc(&REF, 2ull * MMAX * N)); CK(cudaMalloc(&OUT, 2ull * MMAX * N)); CK(cudaMalloc(&dc, 8));
  const float alpha = 1.f, beta = 0.f;
  auto gemm = [&](const __half* a, int m, __half* out) {
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, m, K, &alpha, W, CUDA_R_16F, K, a, CUDA_R_16F, K,
                    &beta, out, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  std::vector<unsigned long long> bad(MMAX + 1, 0);
  for (int f = 0; f < FILLS; ++f) {
    fill<<<256, 256>>>(A, (size_t)MMAX * K, 11u + 7 * f, -6 + f, 3 + f);
    fill<<<256, 256>>>(W, (size_t)N * K, 97u + 5 * f, -12 + f, -3 + f);
    for (int r = 0; r < MMAX; r += MREF)                   // every row as a 40-row batch computes it
      gemm(A + (size_t)r * K, MREF, REF + (size_t)r * N);
    for (int m = 1; m <= MMAX; ++m) {
      gemm(A, m, OUT);
      CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(REF, OUT, (size_t)m * N, dc);
      unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad[m] += d;
    }
  }
  int differ = 0;
  printf("N %d K %d: row counts with mismatches against 40-row calls:", N, K);
  for (int m = 1; m <= MMAX; ++m)
    if (bad[m]) { printf(" %d", m); ++differ; }
  printf("\n  %d of %d row counts differ (%d fills)\n", differ, MMAX, FILLS);
  return 0;
}
