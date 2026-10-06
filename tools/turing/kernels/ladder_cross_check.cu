// src/cuda/ladder_cross.cuh's kernels (a long recording's sampled ladder rows against their clip's memory) against
// what they replace, cuBLAS's pointer-array batched products a group at a time (src/cuda/shared_memory_rows.cu: one
// call of group x 20 entries per group of rows), bit for bit: every split of up to 25 rows into up to 5 groups of
// 1..5 rows, in every order (3,905 splits), 2 fills; the scores, then the output from the same probabilities. The
// output kernel reads a head's values once for a block of rows (lc_output_blocks), so the splits cover every block
// a ladder makes, at every block size the launch may use (1..8 rows); then the output launch's time by
// block size.
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
  __half *K, *V, *Q, *P, *S, *S2, *O, *O2; unsigned long long* dc; void** ptrs;
  CK(cudaMalloc(&K, 2 * memory)); CK(cudaMalloc(&V, 2 * memory)); CK(cudaMalloc(&Q, 2ull * kE * lc_depth));
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
    fill<<<1024, 256>>>(Q, (size_t)kE * lc_depth, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(P, (size_t)kE * lc_keys, 13u + fill_no, -14, -2);
    for (const auto& groups : splits) {
      LadderRows all{}, gemv{}, mma{};
      int first = 0;
      for (const int g : groups) {
        for (int r = first; r < first + g; ++r) {
          all.row[all.count] = (int8_t)r;
          all.group[all.count++] = (int8_t)g;
          LadderRows& list = g <= 2 ? gemv : mma;          // as ladder_cross_scores splits them
          list.row[list.count] = (int8_t)r;
          list.group[list.count++] = (int8_t)g;
        }
        first += g;
      }
      const int rows = all.count;
      // scores: cuBLAS group by group, then the kernels
      first = 0;
      for (const int g : groups) {
        batched(true, lc_keys, lc_depth, scale, K, (size_t)lc_keys * lc_depth, lc_depth, Q, lc_depth, lc_depth, S,
                lc_keys, lc_keys, first, g);
        first += g;
      }
      if (gemv.count > 0)
        lc_scores_gemv<<<dim3((lc_keys + 127) / 128, kHeads, gemv.count), 128>>>(Q, K, S2, gemv, kHeads, scale);
      if (mma.count > 0)
        lc_scores_mma<<<dim3((lc_keys + 15) / 16, kHeads), 32>>>(Q, K, S2, mma, kHeads, scale);
      CK(cudaGetLastError());
      unsigned long long d = differ(S, S2, (size_t)rows * kHeads * lc_keys);
      // output from the same probabilities
      first = 0;
      for (const int g : groups) {
        batched(false, lc_depth, lc_keys, one, V, (size_t)lc_keys * lc_depth, lc_depth, P, lc_keys, lc_keys, O,
                lc_depth, lc_depth, first, g);
        first += g;
      }
      for (const int cap : {1, 2, 3, 4, 5, 8}) {            // every block size the launch may use
        size_t smem = 0;
        const LcOutputBlocks blocks = lc_output_blocks(all, smem, cap);
        lc_output<<<dim3(lc_depth / 32, kHeads, blocks.count), dim3(32, split_lanes), smem>>>(P, V, O2, blocks,
                                                                                               kHeads);
        CK(cudaGetLastError());
        d += differ(O, O2, (size_t)rows * kHeads * lc_depth);
      }
      if (d && bad < 30) {
        printf("fill %d groups", fill_no);
        for (const int g : groups) printf(" %d", g);
        printf(": %llu values differ\n", d);
      }
      bad += d != 0;
      total += d;
      ++cases;
    }
  }
  // Time of the output launch by block size: a ladder's 25 rows (5 groups of 5), a ladder late (groups 3, 2, 1, 1),
  // five single rows; 200 launches each.
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const std::vector<std::vector<int>> timed = {{5, 5, 5, 5, 5}, {3, 2, 1, 1}, {1, 1, 1, 1, 1}};
  for (const auto& groups : timed) {
    LadderRows all{};
    int first = 0;
    for (const int g : groups) {
      for (int r = first; r < first + g; ++r) { all.row[all.count] = (int8_t)r; all.group[all.count++] = (int8_t)g; }
      first += g;
    }
    for (const int cap : {1, 2, 3, 4, 5, 8}) {
      size_t smem = 0;
      const LcOutputBlocks blocks = lc_output_blocks(all, smem, cap);
      CK(cudaEventRecord(e0));
      for (int rep = 0; rep < 200; ++rep)
        lc_output<<<dim3(lc_depth / 32, kHeads, blocks.count), dim3(32, split_lanes), smem>>>(P, V, O2, blocks,
                                                                                               kHeads);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
      printf("time groups");
      for (const int g : groups) printf(" %d", g);
      printf(" cap %d: %6.1f us a launch (%d blocks)\n", cap, 1000 * ms / 200, blocks.count * 2 * kHeads);
    }
  }
  printf("%llu of %llu (fill, split) differ\nTOTAL %llu\n", bad, cases, total);
  return 0;
}
