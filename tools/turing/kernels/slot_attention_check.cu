// src/cuda/slot_attention.cuh's kernels (the stream's slot parts' self-attention in one launch) against what they
// replace, cuBLAS's strided batched products a part at a time (layers/slot_cache.cc: 5 beams x 20 heads, one query of
// 64 dims, the slots' stride of 448 x 64), bit for bit: for every t 32..448 (the kernels' range: slot_attention_applies;
// at d541ec0c t 2..21 differed in 25 of their cases, none from 22 on), the shared prompt [0, shared) alike in all
// the slots at shared 0, t / 2 and min(t, 227), 2 fills; the scores, then the output from the same probabilities.
// usage: slot_attention_check -> must end with TOTAL 0
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/slot_attention.cuh"

using namespace ctranslate2::cuda;

constexpr int kRows = sa_rows, kHeads = sa_heads, kD = sa_depth, kC = sa_capacity, kE = kRows * kHeads;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const size_t cache = (size_t)kE * kC * kD;
  __half *K, *V, *Q, *P, *S, *F, *O, *O2; SlotAttention* table; unsigned long long* dc;
  CK(cudaMalloc(&K, 2 * cache)); CK(cudaMalloc(&V, 2 * cache)); CK(cudaMalloc(&Q, 2ull * kE * kD));
  CK(cudaMalloc(&P, 2ull * kE * kC)); CK(cudaMalloc(&S, 2ull * kE * kC)); CK(cudaMalloc(&F, 2ull * kE * kC));
  CK(cudaMalloc(&O, 2ull * kE * kD)); CK(cudaMalloc(&O2, 2ull * kE * kD)); CK(cudaMalloc(&table, sizeof (SlotAttention)));
  CK(cudaMalloc(&dc, 8));
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  unsigned long long total = 0, bad_cases = 0, cases = 0;
  for (int fill_no = 0; fill_no < 2; ++fill_no) {
    fill<<<1024, 256>>>(K, cache, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V, cache, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(Q, (size_t)kE * kD, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(P, (size_t)kE * kC, 13u + fill_no, -14, -2);
    for (int t = 32; t <= kC; ++t)
      for (const int shared : {0, t / 2, t < 227 ? t : 227}) {
        // the prompt's positions alike in every slot: slot 0's copied into the others, every head
        for (int b = 1; b < kRows; ++b)
          if (shared > 0)
            CK(cudaMemcpy2D(K + (size_t)b * kHeads * kC * kD, 2ull * kC * kD, K, 2ull * kC * kD, 2ull * shared * kD,
                            kHeads, cudaMemcpyDeviceToDevice));
        for (int b = 1; b < kRows; ++b)
          if (shared > 0)
            CK(cudaMemcpy2D(V + (size_t)b * kHeads * kC * kD, 2ull * kC * kD, V, 2ull * kC * kD, 2ull * shared * kD,
                            kHeads, cudaMemcpyDeviceToDevice));
        const int sr = selfattn_scores_recipe_of_t[t], orc = selfattn_output_recipe_of_t[t];
        const SlotAttention part{K, V, K, V, F, kRows, t, shared, 0, kC, sr, orc,
                                 selfattn_scores_recipes[sr].kind == 1, selfattn_output_recipes[orc].kind == 1};
        CK(cudaMemcpy(table, &part, sizeof part, cudaMemcpyHostToDevice));
        // scores: cuBLAS as slot_scores calls it, then the fused kernels
        CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, K, CUDA_R_16F, kD,
                                      (long long)kC * kD, Q, CUDA_R_16F, kD, kD, &zero, S, CUDA_R_16F, t, t, kE,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        sa_scores_launch(table, 1, Q, kHeads, t, scale, 0);
        CK(cudaGetLastError());
        unsigned long long d = differ(S, F, (size_t)kE * t);
        // output from the same probabilities: cuBLAS as slot_values calls it, then the fused kernels (they read
        // the probabilities where the scores were)
        CK(cudaMemcpy(F, P, 2ull * kE * t, cudaMemcpyDeviceToDevice));
        CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, V, CUDA_R_16F, kD,
                                      (long long)kC * kD, F, CUDA_R_16F, t, t, &zero, O, CUDA_R_16F, kD, kD, kE,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        sa_output_launch(table, 1, O2, kHeads, 0);
        CK(cudaGetLastError());
        d += differ(O, O2, (size_t)kE * kD);
        if (d && bad_cases < 30)
          printf("fill %d t %3d shared %3d (scores %d, output %d): %llu values differ\n", fill_no, t, shared, sr, orc, d);
        bad_cases += d != 0;
        total += d;
        ++cases;
      }
  }
  printf("%llu of %llu (fill, t, shared) differ\nTOTAL %llu\n", bad_cases, cases, total);
  return 0;
}
