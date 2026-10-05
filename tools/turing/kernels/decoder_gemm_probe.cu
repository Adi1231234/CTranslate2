// The Whisper decoder's products at the rows a step of several batches has (2..320): cuBLAS (one call, as
// CTranslate2 makes it) against src/cuda/tiled_split_gemm.cuh at several output tiles, for bits and time.
// Row-independent products (3840 / 1280 / 5120 / 51872 x 1280, cuda/clip_groups.h): one chain over k, every M
// 2..320 against the cuBLAS call, 2 fills. The second feed-forward (1280 x 5120): groups of rows with their own
// splits against one cuBLAS call per group (grouped_split_check's sequences, 1 fill). Then times with the weights
// read from DRAM (L2 flushed before each call), cuBLAS and every tile.
// usage: decoder_gemm_probe -> must end with TOTAL 0
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/tiled_split_gemm.cuh"

using namespace ctranslate2::cuda;
constexpr int MMAX = 320, KMAX = 5120, NMAX = 51872;

struct Tile {
  int tm, tn, stages;
  void (*run)(const __half*, const __half*, __half*, int, int, int, const SplitGroups&, cudaStream_t);
};
static const Tile tiles[] = {          // 16-row tiles: the 4 warps side by side; 16-column tiles: stacked
  {16, 64, 4, tsg_launch<16, 64, 1>}, {16, 128, 4, tsg_launch<16, 128, 1>},
  {32, 32, 4, tsg_launch<32, 32>}, {32, 64, 4, tsg_launch<32, 64>}, {64, 32, 4, tsg_launch<64, 32>},
  {64, 64, 4, tsg_launch<64, 64>}, {32, 128, 4, tsg_launch<32, 128>}, {64, 128, 4, tsg_launch<64, 128>},
  {64, 16, 4, tsg_launch<64, 16, 4>}, {64, 16, 8, tsg_launch<64, 16, 4, 8>}, {32, 32, 8, tsg_launch<32, 32, 2, 8>},
  {64, 32, 8, tsg_launch<64, 32, 2, 8>}, {16, 64, 8, tsg_launch<16, 64, 1, 8>},
};
constexpr int TILES = sizeof tiles / sizeof tiles[0];

static SplitGroups split_groups(const std::vector<int64_t>& rows, int k) {
  SplitGroups g{};
  int m = 0;
  for (const int64_t r : rows) {
    int slice = 0, slices = 0;
    if (!gsg_split_of(r, slice, slices)) { slice = k; slices = 1; }
    m += (int)r;
    g.row_end[g.count] = m; g.slice[g.count] = slice; g.slices[g.count] = slices; ++g.count;
  }
  return g;
}

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *R, *C; unsigned long long* dc; char* flush;
  CK(cudaMalloc(&A, 2ull * MMAX * KMAX)); CK(cudaMalloc(&W, 2ull * NMAX * 1280));
  CK(cudaMalloc(&R, 2ull * MMAX * NMAX)); CK(cudaMalloc(&C, 2ull * MMAX * NMAX)); CK(cudaMalloc(&dc, 8));
  CK(cudaMalloc(&flush, 256 << 20));
  const float alpha = 1.f, beta = 0.f;
  auto cublas = [&](int m, int n, int k, const __half* a, __half* c) {
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, W, CUDA_R_16F, k, a, CUDA_R_16F, k, &beta, c,
                    CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  auto diff = [&](size_t count) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(R, C, count, dc);
    unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); return d;
  };
  const int shapes[][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {51872, 1280}};
  unsigned long long total = 0;
  for (const auto& s : shapes) {
    const int n = s[0], k = s[1];
    unsigned long long bad[TILES] = {};
    for (int f = 0; f < 2; ++f) {
      fill<<<1024, 256>>>(A, (size_t)MMAX * k, 41u + 3 * f, -9 + 2 * f, 1 + f);
      fill<<<1024, 256>>>(W, (size_t)n * k, 59u + 5 * f, -15 + f, -3 + 2 * f);
      for (int m = 2; m <= MMAX; ++m) {
        cublas(m, n, k, A, R);
        for (int t = 0; t < TILES; ++t) {
          tiles[t].run(A, W, C, m, n, k, tsg_chain(m, k), 0);
          CK(cudaGetLastError());
          bad[t] += diff((size_t)m * n);
        }
      }
    }
    printf("%5d x %d, M 2..320 x 2 fills, mismatched values by tile:", n, k);
    for (int t = 0; t < TILES; ++t) {
      printf(" %dx%d/%d %llu", tiles[t].tm, tiles[t].tn, tiles[t].stages, bad[t]);
      total += bad[t];
    }
    printf("\n");
  }
  {                                                          // the second feed-forward, groups of rows
    const int n = 1280, k = 5120;
    fill<<<1024, 256>>>(A, (size_t)MMAX * k, 77u, -9, 1);
    fill<<<1024, 256>>>(W, (size_t)n * k, 91u, -15, -3);
    std::mt19937 rng(11);
    unsigned long long bad[TILES] = {};
    int sequences = 0;
    for (int c = 0; c < 400; ++c) {
      std::vector<int64_t> g; int m = 0;
      while (g.size() < 16) {
        const int r = 2 + (int)(rng() % 47);
        if (m + r > MMAX) break;
        g.push_back(r); m += r;
        if (rng() % 5 == 0) break;
      }
      int row = 0;
      for (const int64_t r : g) { cublas((int)r, n, k, A + (size_t)row * k, R + (size_t)row * n); row += (int)r; }
      for (int t = 0; t < TILES; ++t) {
        tiles[t].run(A, W, C, m, n, k, split_groups(g, k), 0);
        bad[t] += diff((size_t)m * n);
      }
      ++sequences;
    }
    printf(" 1280 x 5120, %d group sequences, mismatched values by tile:", sequences);
    for (int t = 0; t < TILES; ++t) {
      printf(" %dx%d/%d %llu", tiles[t].tm, tiles[t].tn, tiles[t].stages, bad[t]);
      total += bad[t];
    }
    printf("\n");
  }

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto timed = [&](auto run) {
    float best = 1e9;
    for (int r = 0; r < 15; ++r) {
      CK(cudaMemset(flush, r, 256 << 20));                    // weights out of L2
      CK(cudaEventRecord(e0)); run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    return best * 1000;
  };
  const int ms[] = {40, 80, 120, 160, 200, 240, 280, 320};
  for (const auto& s : shapes) {
    const int n = s[0], k = s[1];
    printf("%5d x %d (%.1f MB): us at M = cuBLAS | tiles", n, k, 2e-6 * n * k);
    for (int t = 0; t < TILES; ++t) printf(" %dx%d/%d", tiles[t].tm, tiles[t].tn, tiles[t].stages);
    printf("\n");
    for (const int m : ms) {
      printf("  M %3d: %7.1f |", m, timed([&] { cublas(m, n, k, A, C); }));
      for (int t = 0; t < TILES; ++t)
        printf(" %7.1f", timed([&] { tiles[t].run(A, W, C, m, n, k, tsg_chain(m, k), 0); }));
      printf("\n");
    }
  }
  printf(" 1280 x 5120 (13.1 MB), groups of 40 rows: cuBLAS per group | tiles\n");
  for (int groups = 1; groups <= 8; ++groups) {
    const std::vector<int64_t> g(groups, 40);
    printf("  %d x 40: %7.1f |", groups, timed([&] {
      for (int i = 0; i < groups; ++i) cublas(40, 1280, 5120, A + (size_t)i * 40 * 5120, C + (size_t)i * 40 * 1280);
    }));
    for (int t = 0; t < TILES; ++t)
      printf(" %7.1f", timed([&] { tiles[t].run(A, W, C, 40 * groups, 1280, 5120, split_groups(g, 5120), 0); }));
    printf("\n");
  }
  printf("TOTAL %llu\n", total);
  return 0;
}
