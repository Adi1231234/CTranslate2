// Bit-for-bit check and timing of src/cuda/grouped_split_gemm.cuh (the Whisper decoder's second feed-forward,
// 1280 x 5120, for several batches' rows in one pass) against what it replaces: one cuBLAS call per group with
// that group's rows (C = A W^T, fp16, COMPUTE_32F, as CTranslate2 calls it). Every sequence of 1..4 groups of
// 5..40 rows (multiples of 5: 8 clips x 5 beams), 600 random sequences of 2..48-row groups up to 480 rows and 96
// groups (more than one launch holds: several launches), 16-row groups (the prompt), and 1..96 groups of 5 rows (a
// long-recording stream's windows); 3 fills each. Then timings with the weights read from DRAM (L2 flushed before
// each call).
// usage: grouped_split_check -> must end with TOTAL 0
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/grouped_split_gemm.cuh"

constexpr int N = 1280, K = 5120, MMAX = 480;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *R, *C; unsigned long long* dc; char* flush;
  CK(cudaMalloc(&A, 2ull * MMAX * K)); CK(cudaMalloc(&W, 2ull * N * K));
  CK(cudaMalloc(&R, 2ull * MMAX * N)); CK(cudaMalloc(&C, 2ull * MMAX * N)); CK(cudaMalloc(&dc, 8));
  CK(cudaMalloc(&flush, 256 << 20));
  const float alpha = 1.f, beta = 0.f;
  auto reference = [&](const std::vector<int64_t>& groups) {   // what each group's batch alone computes
    int row = 0;
    for (const int64_t m : groups) {
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, (int)m, K, &alpha, W, CUDA_R_16F, K, A + (size_t)row * K,
                      CUDA_R_16F, K, &beta, R + (size_t)row * N, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
      row += (int)m;
    }
    return row;
  };
  std::vector<std::vector<int64_t>> configs;
  for (int n = 1; n <= 4; ++n) {
    std::vector<int64_t> g(n, 5);
    while (true) {
      configs.push_back(g);
      int i = 0;
      while (i < n && g[i] == 40) g[i++] = 5;
      if (i == n) break;
      g[i] += 5;
    }
  }
  std::mt19937 rng(7);
  for (int c = 0; c < 600; ++c) {
    std::vector<int64_t> g; int total = 0;
    while (true) {
      const int m = 2 + (int)(rng() % 47);
      if (total + m > MMAX || g.size() == 96) break;
      g.push_back(m); total += m;
      if (rng() % (c < 300 ? 4 : 40) == 0) break;
    }
    if (!g.empty()) configs.push_back(g);
  }
  for (int n = 1; n <= 16; ++n) configs.push_back(std::vector<int64_t>(n, 16));
  for (int n = 5; n <= 8; ++n) configs.push_back(std::vector<int64_t>(n, 40));
  for (int n = 1; n <= 96; ++n) configs.push_back(std::vector<int64_t>(n, 5));
  unsigned long long total_bad = 0, worst_config = 0;
  for (int f = 0; f < 3; ++f) {
    fill<<<1024, 256>>>(A, (size_t)MMAX * K, 41u + 3 * f, -9 + 2 * f, 1 + f);
    fill<<<1024, 256>>>(W, (size_t)N * K, 59u + 5 * f, -15 + f, -3 + 2 * f);
    for (size_t i = 0; i < configs.size(); ++i) {
      const int m = reference(configs[i]);
      if (!ctranslate2::cuda::gsg_run(A, W, C, N, K, configs[i], 0)) { printf("config %zu not run\n", i); return 1; }
      CK(cudaGetLastError());
      CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(R, C, (size_t)m * N, dc);
      unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost));
      if (d && !total_bad) {
        printf("first mismatch: fill %d config", f);
        for (auto x : configs[i]) printf(" %lld", (long long)x);
        printf(": %llu values\n", d);
      }
      total_bad += d; worst_config = d > worst_config ? d : worst_config;
    }
  }
  printf("%zu group sequences x 3 fills: %llu mismatched values\n", configs.size(), total_bad);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto timed = [&](auto run) {
    float best = 1e9;
    for (int r = 0; r < 20; ++r) {
      CK(cudaMemset(flush, r, 256 << 20));                    // weights out of L2
      CK(cudaEventRecord(e0)); run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    return best * 1000;
  };
  for (auto g : std::vector<std::vector<int64_t>>{{40}, {40, 40}, {40, 40, 40, 40}, {35, 20, 10, 40}, {16, 16, 16, 16},
                                                  {40, 40, 40, 40, 40, 40, 40, 40}, {40, 35, 30, 25, 20, 15, 10, 5},
                                                  std::vector<int64_t>(34, 5), std::vector<int64_t>(48, 5),
                                                  std::vector<int64_t>(64, 5), std::vector<int64_t>(80, 5)}) {
    int m = 0; for (auto x : g) m += (int)x;
    const float tc = timed([&] { reference(g); });
    const float tk = timed([&] { ctranslate2::cuda::gsg_run(A, W, C, N, K, g, 0); });
    const float t1 = timed([&] {
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, m, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
                      CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); });
    printf("groups");
    for (auto x : g) printf(" %lld", (long long)x);
    printf(": cuBLAS per group %.1f us, grouped kernel %.1f us, one cuBLAS call (other bits) %.1f us\n", tc, tk, t1);
  }
  printf("TOTAL %llu\n", total_bad);
  return 0;
}
