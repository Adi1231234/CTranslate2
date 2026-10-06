// src/cuda/single_rows_gemv.cuh's kernel (several rows each with the bits of a product of one row, the weights read
// once) against what it replaces, cuBLAS's call for each row alone (primitives<CUDA>::gemm with m = 1), bit for bit:
// every recovered shape, 1..16 rows taken at scattered rows of a 40-row input, 3 fills; and the time of 1..16 rows
// against as many cuBLAS calls (the launch's whole point).
// usage: single_rows_check -> must end with TOTAL 0
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/single_rows_gemv.cuh"

using namespace ctranslate2::cuda;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const int shapes[][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {1280, 5120}, {51872, 1280}, {51866, 1280}};
  constexpr int M = 40;
  __half *W, *X, *Y, *Z; unsigned long long* dc;
  CK(cudaMalloc(&W, 2ull * 51872 * 5120)); CK(cudaMalloc(&X, 2ull * M * 5120));
  CK(cudaMalloc(&Y, 2ull * M * 51872)); CK(cudaMalloc(&Z, 2ull * M * 51872)); CK(cudaMalloc(&dc, 8));
  const float one = 1.f, zero = 0.f;
  auto differ = [&](const __half* a, const __half* b, size_t count) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(a, b, count, dc);
    unsigned long long v; CK(cudaMemcpy(&v, dc, 8, cudaMemcpyDeviceToHost)); return v;
  };
  unsigned long long total = 0, cases = 0, bad = 0;
  for (const auto& s : shapes) {
    const int n = s[0], k = s[1], T = single_rows_partials(n, k);
    if (T == 0) { printf("%d x %d: no recipe\n", n, k); ++bad; continue; }
    for (int fill_no = 0; fill_no < 3; ++fill_no) {
      fill<<<1024, 256>>>(W, (size_t)n * k, 31u + fill_no, -9, 0);
      fill<<<256, 256>>>(X, (size_t)M * k, 7u + fill_no, -4, 2);
      for (int count = 1; count <= sr_max_rows; ++count) {
        SingleRows rows{};
        rows.count = count;
        std::vector<int> picked;
        for (int r = 0; r < count; ++r) picked.push_back((r * 7 + fill_no * 3 + count) % M);   // scattered, may repeat
        for (int r = count; r < sr_max_rows; ++r)          // as single_rows_gemv.cu fills them
          rows.x[r] = X + (size_t)picked[0] * k;
        for (int r = 0; r < count; ++r) {
          rows.x[r] = X + (size_t)picked[r] * k;
          rows.y[r] = Z + (size_t)r * n;
          CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &one, W, CUDA_R_16F, k, X + (size_t)picked[r] * k,
                          CUDA_R_16F, k, &zero, Y + (size_t)r * n, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT));
        }
        const int blocks = single_rows_blocks(n, T);
        if (T == 32) single_rows_gemv_kernel<32><<<blocks, sr_threads>>>(W, rows, n, k);
        else if (T == 16) single_rows_gemv_kernel<16><<<blocks, sr_threads>>>(W, rows, n, k);
        else single_rows_gemv_kernel<8><<<blocks, sr_threads>>>(W, rows, n, k);
        CK(cudaGetLastError());
        const unsigned long long d = differ(Y, Z, (size_t)count * n);
        if (d && bad < 30) printf("%d x %d fill %d rows %d: %llu values differ\n", n, k, fill_no, count, d);
        bad += d != 0;
        total += d;
        ++cases;
      }
    }
  }
  // Time: the launch against one cuBLAS call a row, 50 repeats each, the weights evicted from L2 by the size of W.
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (const auto& s : shapes) {
    const int n = s[0], k = s[1], T = single_rows_partials(n, k);
    for (const int count : {1, 2, 3, 5, 8, 16}) {
      SingleRows rows{};
      rows.count = count;
      for (int r = 0; r < sr_max_rows; ++r) {
        rows.x[r] = X + (size_t)(r < count ? r : 0) * k;
        rows.y[r] = r < count ? Z + (size_t)r * n : nullptr;
      }
      float ms_kernel = 0, ms_cublas = 0;
      CK(cudaEventRecord(e0));
      for (int rep = 0; rep < 50; ++rep) {
        const int blocks = single_rows_blocks(n, T);
        if (T == 32) single_rows_gemv_kernel<32><<<blocks, sr_threads>>>(W, rows, n, k);
        else if (T == 16) single_rows_gemv_kernel<16><<<blocks, sr_threads>>>(W, rows, n, k);
        else single_rows_gemv_kernel<8><<<blocks, sr_threads>>>(W, rows, n, k);
      }
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms_kernel, e0, e1));
      CK(cudaEventRecord(e0));
      for (int rep = 0; rep < 50; ++rep)
        for (int r = 0; r < count; ++r)
          CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &one, W, CUDA_R_16F, k, X + (size_t)r * k, CUDA_R_16F,
                          k, &zero, Y + (size_t)r * n, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms_cublas, e0, e1));
      printf("time %5d x %4d rows %2d: kernel %7.1f us, cuBLAS %7.1f us\n", n, k, count, 1000 * ms_kernel / 50,
             1000 * ms_cublas / 50);
    }
  }
  printf("%llu of %llu (shape, fill, rows) differ\nTOTAL %llu\n", bad, cases, total);
  return 0;
}
