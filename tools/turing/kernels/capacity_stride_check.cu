// The greedy search's capacity caches (src/layers/capacity_cache.h): the decoder's self-attention products of a step
// with each entry's keys and values at a batch stride of capacity x 64 instead of t x 64. They give the stock bits
// only if cuBLAS computes every entry the same at any such stride, for the batches a sampled ladder's groups make (1..5
// rows of a temperature x 20 heads: 20..100 entries). For every t 1..448, batches 20, 40, 60, 80 and 100, and
// capacities t + 1, t + 3, t + 64, 448 (from t) and 512: scores = alpha q k^T and output = p v as
// primitives<CUDA>::gemm_batch_strided calls cuBLAS, against the contiguous layout. 2 fills. Must end with TOTAL 0.
// usage: capacity_stride_check
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kD = 64, kE = 100, kT = 448, kC = 512;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *Q, *K, *V, *P, *KS, *VS, *C1, *C2, *O1, *O2; unsigned long long* dc;
  for (__half** p : {&K, &V, &KS, &VS}) CK(cudaMalloc(p, 2ull * kE * kC * kD));
  for (__half** p : {&Q, &O1, &O2}) CK(cudaMalloc(p, 2ull * kE * kD));
  for (__half** p : {&P, &C1, &C2}) CK(cudaMalloc(p, 2ull * kE * kT));
  CK(cudaMalloc(&dc, 8));
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  auto scores = [&](int t, int e, const __half* k, long long stride, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, k, CUDA_R_16F, kD, stride, Q,
                                  CUDA_R_16F, kD, kD, &zero, out, CUDA_R_16F, t, t, e, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto output = [&](int t, int e, const __half* v, long long stride, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, v, CUDA_R_16F, kD, stride, P,
                                  CUDA_R_16F, t, t, &zero, out, CUDA_R_16F, kD, kD, e, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  unsigned long long total = 0, cases = 0, bad = 0;
  for (int fill_no = 0; fill_no < 2; ++fill_no) {
    fill<<<1024, 256>>>(Q, (size_t)kE * kD, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(K, (size_t)kE * kT * kD, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V, (size_t)kE * kT * kD, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(P, (size_t)kE * kT, 13u + fill_no, -14, -2);
    for (int t = 1; t <= kT; ++t) {
      for (int e = 20; e <= kE; e += 20) {
        scores(t, e, K, (long long)t * kD, C1);              // the stock contiguous caches: entry e at e * t * 64
        output(t, e, V, (long long)t * kD, O1);
        for (const int c : {t + 1, t + 3, t + 64, kT, kC}) {
          if (c < t)
            continue;
          // the same entries at stride c * 64
          CK(cudaMemcpy2D(KS, 2ull * c * kD, K, 2ull * t * kD, 2ull * t * kD, e, cudaMemcpyDeviceToDevice));
          CK(cudaMemcpy2D(VS, 2ull * c * kD, V, 2ull * t * kD, 2ull * t * kD, e, cudaMemcpyDeviceToDevice));
          scores(t, e, KS, (long long)c * kD, C2);
          output(t, e, VS, (long long)c * kD, O2);
          const unsigned long long d = differ(C1, C2, (size_t)e * t) + differ(O1, O2, (size_t)e * kD);
          if (d && bad < 20)
            printf("fill %d t %3d batch %3d capacity %3d: %llu values differ\n", fill_no, t, e, c, d);
          bad += d != 0;
          total += d;
          ++cases;
        }
      }
    }
  }
  printf("%llu of %llu (fill, t, batch, capacity) differ\nTOTAL %llu mismatches\n", bad, cases, total);
  return 0;
}
