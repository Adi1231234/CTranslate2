// Can the decoder's self-attention caches stay in place (slots of capacity 448 a row) instead of being copied whole at
// every step (Gather of the beams' order + Concat of the step: CTranslate2's reorder_and_append, ~2/3 of a long
// recording's decoding traffic)? Only if cuBLAS gives every entry the same bits when its keys and values sit at a
// larger batch stride (T x 64 instead of t x 64), and when the entries come in another order (each row's query to its
// slot). For every t 1..448: one window's 100 entries (5 beams x 20 heads, one query of 64 dims each), scores =
// alpha q k^T and output = p v, as primitives<CUDA>::gemm_batch_strided calls cuBLAS: the contiguous layout as the
// reference, then the slot layout, then the slot layout with the entries in a shuffled order. 2 fills. Must end with
// TOTAL 0.
// usage: cache_stride_check
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kD = 64, kE = 100, kT = 448;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *Q, *K, *V, *P, *KS, *VS, *QS, *PS, *C1, *C2, *O1, *O2; unsigned long long* dc;
  const size_t cache = (size_t)kE * kT * kD;
  for (__half** p : {&K, &V, &KS, &VS}) CK(cudaMalloc(p, 2 * cache));
  for (__half** p : {&Q, &QS, &O1, &O2}) CK(cudaMalloc(p, 2ull * kE * kD));
  for (__half** p : {&P, &PS, &C1, &C2}) CK(cudaMalloc(p, 2ull * kE * kT));
  CK(cudaMalloc(&dc, 8));
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  // scores[e] = scale q[e] k[e]^T over t keys; output[e] = p[e] v[e]; entry e's keys at k + e * stride
  auto scores = [&](int t, const __half* k, long long stride, const __half* q, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, k, CUDA_R_16F, kD, stride, q,
                                  CUDA_R_16F, kD, kD, &zero, out, CUDA_R_16F, t, t, kE, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto output = [&](int t, const __half* v, long long stride, const __half* p, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, v, CUDA_R_16F, kD, stride, p,
                                  CUDA_R_16F, t, t, &zero, out, CUDA_R_16F, kD, kD, kE, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  std::vector<int> order(kE);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937(5));
  unsigned long long total = 0;
  int bad_t = 0;
  for (int fill_no = 0; fill_no < 2; ++fill_no) {
    fill<<<1024, 256>>>(Q, (size_t)kE * kD, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(KS, cache, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(VS, cache, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(P, (size_t)kE * kT, 13u + fill_no, -14, -2);
    for (int t = 1; t <= kT; ++t) {
      // the contiguous caches the stock code builds: entry e's first t slot rows, packed
      CK(cudaMemcpy2D(K, 2ull * t * kD, KS, 2ull * kT * kD, 2ull * t * kD, kE, cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy2D(V, 2ull * t * kD, VS, 2ull * kT * kD, 2ull * t * kD, kE, cudaMemcpyDeviceToDevice));
      scores(t, K, (long long)t * kD, Q, C1);
      output(t, V, (long long)t * kD, P, O1);
      scores(t, KS, (long long)kT * kD, Q, C2);
      output(t, VS, (long long)kT * kD, P, O2);
      unsigned long long d = differ(C1, C2, (size_t)kE * t) + differ(O1, O2, (size_t)kE * kD);
      // the same entries in another order: entry j of the call is entry order[j] (its query, keys, p, values)
      for (int j = 0; j < kE; ++j) {
        const int e = order[j];
        CK(cudaMemcpy(QS + (size_t)j * kD, Q + (size_t)e * kD, 2 * kD, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(PS + (size_t)j * t, P + (size_t)e * t, 2ull * t, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(K + (size_t)j * kT * kD, KS + (size_t)e * kT * kD, 2ull * t * kD, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(V + (size_t)j * kT * kD, VS + (size_t)e * kT * kD, 2ull * t * kD, cudaMemcpyDeviceToDevice));
      }
      scores(t, K, (long long)kT * kD, QS, C2);
      output(t, V, (long long)kT * kD, PS, O2);
      for (int j = 0; j < kE; ++j) {
        const int e = order[j];
        d += differ(C1 + (size_t)e * t, C2 + (size_t)j * t, t) + differ(O1 + (size_t)e * kD, O2 + (size_t)j * kD, kD);
      }
      if (d) {
        ++bad_t;
        printf("fill %d t %3d: %llu values differ\n", fill_no, t, d);
      }
      total += d;
    }
  }
  printf("%d of %d (fill, t) differ\nTOTAL %llu mismatches\n", bad_t, 2 * kT, total);
  return 0;
}
