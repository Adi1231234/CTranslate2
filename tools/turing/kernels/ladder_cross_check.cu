// src/cuda/ladder_cross.cuh's kernels (a long recording's sampled ladder rows against their clip's memory) against
// what they replace, cuBLAS's pointer-array batched products a group at a time (src/cuda/shared_memory_rows.cu: one
// call of group x 20 entries per group of rows), bit for bit: every split of up to 25 rows into up to 5 groups of
// 1..5 rows, in every order (3,905 splits), 2 fills; the scores, then the output from the same probabilities; each
// split as one ladder, and (2 or more groups) as two ladders in one launch, the second half of its groups against
// another clip's memory (lc_plan: a joint step's greedy parts).
// usage: ladder_cross_check -> must end with TOTAL 0
#include <cstdio>
#include <functional>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/ladder_cross.cuh"

using namespace ctranslate2::cuda;

constexpr int kHeads = 20, kRows = 25, kE = kRows * kHeads;

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const size_t memory = (size_t)kHeads * lc_keys * lc_depth;
  __half *K, *V, *K2, *V2, *Q, *P, *S, *S2, *O, *O2; unsigned long long* dc; void** ptrs;
  CK(cudaMalloc(&K, 2 * memory)); CK(cudaMalloc(&V, 2 * memory)); CK(cudaMalloc(&Q, 2ull * kE * lc_depth));
  CK(cudaMalloc(&K2, 2 * memory)); CK(cudaMalloc(&V2, 2 * memory));
  CK(cudaMalloc(&P, 2ull * kE * lc_keys)); CK(cudaMalloc(&S, 2ull * kE * lc_keys)); CK(cudaMalloc(&S2, 2ull * kE * lc_keys));
  CK(cudaMalloc(&O, 2ull * kE * lc_depth)); CK(cudaMalloc(&O2, 2ull * kE * lc_depth)); CK(cudaMalloc(&dc, 8));
  CK(cudaMalloc(&ptrs, 3 * kE * sizeof (void*)));
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  // One group's cuBLAS call as shared_memory_rows.cu's batched() makes it.
  auto batched = [&](bool trans_a, int m, int k, float alpha, const __half* a, size_t a_stride, int lda,
                     const __half* b, size_t b_stride, int ldb, __half* c, size_t c_stride, int ldc, int first,
                     int count) {
    const int entries = count * kHeads;
    std::vector<const void*> host(3 * entries);
    for (int e = 0; e < entries; ++e) {
      host[e] = a + (size_t)(e % kHeads) * a_stride;
      host[entries + e] = b + (size_t)(first * kHeads + e) * b_stride;
      host[2 * entries + e] = c + (size_t)(first * kHeads + e) * c_stride;
    }
    CK(cudaMemcpy(ptrs, host.data(), host.size() * sizeof (void*), cudaMemcpyHostToDevice));
    CK(cublasGemmBatchedEx(h, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k, &alpha,
                           (const void**)ptrs, CUDA_R_16F, lda, (const void**)(ptrs + entries), CUDA_R_16F, ldb,
                           &zero, ptrs + 2 * entries, CUDA_R_16F, ldc, entries, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT));
  };
  // Every split: groups of 1..5 rows, up to 5 groups, at most kRows rows.
  std::vector<std::vector<int>> splits;
  std::function<void(std::vector<int>&, int)> grow = [&](std::vector<int>& groups, int rows) {
    if (!groups.empty())
      splits.push_back(groups);
    if (groups.size() == 5)
      return;
    for (int g = 1; g <= 5 && rows + g <= kRows; ++g) {
      groups.push_back(g);
      grow(groups, rows + g);
      groups.pop_back();
    }
  };
  std::vector<int> start;
  grow(start, 0);
  unsigned long long total = 0, bad = 0, cases = 0;
  for (int fill_no = 0; fill_no < 2; ++fill_no) {
    fill<<<1024, 256>>>(K, memory, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V, memory, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(K2, memory, 17u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V2, memory, 19u + fill_no, -6, 2);
    fill<<<1024, 256>>>(Q, (size_t)kE * lc_depth, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(P, (size_t)kE * lc_keys, 13u + fill_no, -14, -2);
    for (const auto& groups : splits) {
      int rows = 0;
      for (const int g : groups) rows += g;
      for (int ladders = 1; ladders <= (groups.size() > 1 ? 2 : 1); ++ladders) {
        const size_t cut = ladders == 1 ? groups.size() : groups.size() / 2;
        int rows0 = 0;
        for (size_t i = 0; i < cut; ++i) rows0 += groups[i];
        std::vector<LcLadder> list{{K, V, 0, std::vector<int>(groups.begin(), groups.begin() + cut)}};
        if (ladders == 2)
          list.push_back({K2, V2, rows0, std::vector<int>(groups.begin() + cut, groups.end())});
        LcPlan plan;
        if (!lc_plan(list, plan)) {
          printf("no plan for a split\n");
          ++bad;
          continue;
        }
        // scores: cuBLAS group by group, each against its ladder's keys, then the kernels
        int first = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
          batched(true, lc_keys, lc_depth, scale, i < cut ? K : K2, (size_t)lc_keys * lc_depth, lc_depth, Q, lc_depth,
                  lc_depth, S, lc_keys, lc_keys, first, groups[i]);
          first += groups[i];
        }
        lc_scores_launch(plan, Q, S2, kHeads, scale, 0);
        CK(cudaGetLastError());
        unsigned long long d = differ(S, S2, (size_t)rows * kHeads * lc_keys);
        // output from the same probabilities
        first = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
          batched(false, lc_depth, lc_keys, one, i < cut ? V : V2, (size_t)lc_keys * lc_depth, lc_depth, P, lc_keys,
                  lc_keys, O, lc_depth, lc_depth, first, groups[i]);
          first += groups[i];
        }
        lc_output_launch(plan, P, O2, kHeads, 0);
        CK(cudaGetLastError());
        d += differ(O, O2, (size_t)rows * kHeads * lc_depth);
        if (d && bad < 30) {
          printf("fill %d, %d ladder(s), groups", fill_no, ladders);
          for (const int g : groups) printf(" %d", g);
          printf(": %llu values differ\n", d);
        }
        bad += d != 0;
        total += d;
        ++cases;
      }
    }
  }
  printf("%llu of %llu (fill, split) differ\nTOTAL %llu\n", bad, cases, total);
  return 0;
}
