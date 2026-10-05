// Does the Whisper decoder's self-attention give an entry the same bits whatever the batch count of its call? The
// two strided batched products CTranslate2 runs per decoding step (one query a row against t cached positions of
// 64 dims, batch = rows x 20 heads): scores = alpha q k^T, output = p v, as primitives<CUDA>::gemm_batch_strided
// calls cuBLAS. Reference: every entry computed by a call of 100 entries (one clip's 5 beams); then calls of B
// entries (B = 100 .. 6400) over the same data, entry by entry. If no (t, B) differs from the 100-entry calls,
// the groups of a grouped batch (cuda/clip_groups.h) can share one call; else each group keeps its own.
// usage: selfattn_probe
#include <algorithm>
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kD = 64, kRef = 100, kMax = 6400;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const int times[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 127, 128, 129, 200,
                       255, 256, 257, 300, 383, 384, 445, 448};
  const int batches[] = {100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1600, 2400, 3200, 4800, 6400};
  const int max_t = 448;
  __half *Q, *K, *V, *P, *C1, *C2, *O1, *O2; unsigned long long* dc;
  CK(cudaMalloc(&Q, 2ull * kMax * kD)); CK(cudaMalloc(&K, 2ull * kMax * max_t * kD));
  CK(cudaMalloc(&V, 2ull * kMax * max_t * kD)); CK(cudaMalloc(&P, 2ull * kMax * max_t));
  CK(cudaMalloc(&C1, 2ull * kMax * max_t)); CK(cudaMalloc(&C2, 2ull * kMax * max_t));
  CK(cudaMalloc(&O1, 2ull * kMax * kD)); CK(cudaMalloc(&O2, 2ull * kMax * kD)); CK(cudaMalloc(&dc, 8));
  fill<<<1024, 256>>>(Q, (size_t)kMax * kD, 3u, -6, 1);
  fill<<<1024, 256>>>(K, (size_t)kMax * max_t * kD, 7u, -7, 1);
  fill<<<1024, 256>>>(V, (size_t)kMax * max_t * kD, 11u, -6, 2);
  fill<<<1024, 256>>>(P, (size_t)kMax * max_t, 13u, -14, -2);
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  auto scores = [&](int t, int first, int count, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, K + (size_t)first * t * kD,
                                  CUDA_R_16F, kD, (long long)t * kD, Q + (size_t)first * kD, CUDA_R_16F, kD, kD,
                                  &zero, out + (size_t)first * t, CUDA_R_16F, t, t, count, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto output = [&](int t, int first, int count, __half* out) {
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, V + (size_t)first * t * kD,
                                  CUDA_R_16F, kD, (long long)t * kD, P + (size_t)first * t, CUDA_R_16F, t, t, &zero,
                                  out + (size_t)first * kD, CUDA_R_16F, kD, kD, count, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
  };
  auto diff = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  int bad = 0;
  for (int t : times) {
    for (int first = 0; first < kMax; first += kRef) {      // the references: 100-entry calls
      scores(t, first, kRef, C1);
      output(t, first, kRef, O1);
    }
    std::vector<int> differ;
    for (int b : batches) {
      scores(t, 0, b, C2);
      output(t, 0, b, O2);
      if (diff(C1, C2, (size_t)b * t) || diff(O1, O2, (size_t)b * kD)) differ.push_back(b);
    }
    printf("t %3d:", t);
    if (differ.empty()) printf(" every batch count the same");
    for (int b : differ) printf(" %d", b);
    printf("\n");
    bad += (int)differ.size();
  }
  printf("TOTAL %d (t, batch) pairs differ from 100-entry calls\n", bad);
  return 0;
}
