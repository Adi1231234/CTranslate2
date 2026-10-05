// Which cuBLASLt algorithm runs each of the Whisper encoder's products with the least energy, with the bits of the
// cuBLAS call CTranslate2 makes? The batched path sits at the L40S's 350 W limit, so a product's cost is its energy;
// cuBLAS's heuristic picks for time (energy_probe: its fc1 kernel, s1688 128x128, 2.5 pJ a FLOP against 2.05 for the
// qkv product's s16816 256x128). For each product at 8 clips (12000 rows) every algorithm and tile with no split-K
// that cuBLASLt accepts runs once against the reference's bits (two fills); the 15 fastest exact ones then run
// alone for ~1 s with NVML's energy counter around them. Prints, per product, the reference's and the exact
// algorithms' time and millijoules, best first.
// usage: encoder_algo_search
#include <algorithm>
#include <cstdio>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <cublasLt.h>
#include "probe_common.h"
#include "probe_data.cuh"

#define LT(x) CK(x)

static void* device;
static int (*nvml_energy)(void*, unsigned long long*);
static void nvml_open() {
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!lib) { fprintf(stderr, "no libnvidia-ml.so.1\n"); exit(1); }
  auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
  auto handle = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
  nvml_energy = reinterpret_cast<int (*)(void*, unsigned long long*)>(
    dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption"));
  CK(init()); CK(handle(0, &device));
}
static unsigned long long energy_mj() { unsigned long long e = 0; CK(nvml_energy(device, &e)); return e; }

struct Cost { double us, mj; };
template <typename F> double time_us(F run) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a)); for (int i = 0; i < 5; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return 1000.0 * ms / 5;
}
template <typename F> Cost cost(F run) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) run();
  CK(cudaEventRecord(a)); for (int i = 0; i < 5; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  const int n = std::max(10, int(1000.f / (ms / 5)));
  CK(cudaDeviceSynchronize());
  const unsigned long long e0 = energy_mj();
  CK(cudaEventRecord(a)); for (int i = 0; i < n; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  const unsigned long long e1 = energy_mj();
  CK(cudaEventElapsedTime(&ms, a, b));
  return {1000.0 * ms / n, double(e1 - e0) / n};
}

int main() {
  nvml_open();
  cublasHandle_t h; CK(cublasCreate(&h));
  cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
  const int m = 12000;
  __half *A, *W, *R, *C; unsigned long long* dc;
  CK(cudaMalloc(&A, 2ull * m * 5120)); CK(cudaMalloc(&W, 2ull * 5120 * 5120));
  CK(cudaMalloc(&R, 2ull * m * 5120)); CK(cudaMalloc(&C, 2ull * m * 5120)); CK(cudaMalloc(&dc, 8));
  void* ws; const size_t ws_size = 32 << 20; CK(cudaMalloc(&ws, ws_size));
  const float one = 1.f, zero = 0.f;
  const int shapes[][2] = {{5120, 1280}, {1280, 5120}, {3840, 1280}, {2560, 1280}, {1280, 1280}};
  const char* names[] = {"fc1", "fc2", "qkv", "memory K/V", "o"};
  for (int s = 0; s < 5; ++s) {
    const int n = shapes[s][0], k = shapes[s][1];
    auto reference = [&] {
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, W, CUDA_R_16F, k, A, CUDA_R_16F, k, &zero, R,
                      CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    cublasLtMatmulDesc_t desc; LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT));
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN));
    cublasLtMatrixLayout_t la, lb, lc;
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, k, m, k));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, n, m, n));
    int ids[256], count = 0;
    LT(cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                256, ids, &count));
    struct Found { std::string name; cublasLtMatmulAlgo_t algo; Cost c; };
    std::vector<Found> exact;
    int tried = 0, differ = 0;
    for (int f = 0; f < 2; ++f) {
      fill<<<1024, 256>>>(A, (size_t)m * k, 7u + f, -4 + f, 1 + f);
      fill<<<1024, 256>>>(W, (size_t)n * k, 9u + f, -9 + f, -3 + f);
      reference();
      for (int i = 0; i < count; ++i) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                   ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        size_t bytes = 0; int tiles[128] = {}, stages[128] = {};
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &bytes);
        const int nt = std::max(1, int(bytes / sizeof(int)));
        if (bytes) cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof tiles, &bytes);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &bytes);
        const int ns = std::max(1, int(bytes / sizeof(int)));
        if (bytes)
          cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, sizeof stages, &bytes);
        for (int ti = 0; ti < std::min(nt, 128); ++ti)
          for (int si = 0; si < std::min(ns, 128); ++si)
            for (int swz = 0; swz < 2; ++swz) {
              cublasLtMatmulAlgo_t a = algo;
              const int split = 1; const uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[ti], sizeof(int));
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages[si], sizeof(int));
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof split);
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red);
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz);
              cublasLtMatmulHeuristicResult_t check;
              if (cublasLtMatmulAlgoCheck(lt, desc, la, lb, lc, lc, &a, &check) != CUBLAS_STATUS_SUCCESS
                  || check.workspaceSize > ws_size) continue;
              CK(cudaMemset(C, 0xff, 2ull * m * n));
              if (cublasLtMatmul(lt, desc, &one, W, la, A, lb, &zero, C, lc, C, lc, &a, ws, ws_size, 0)
                  != CUBLAS_STATUS_SUCCESS) continue;
              CK(cudaDeviceSynchronize());
              CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(R, C, (size_t)m * n, dc);
              unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost));
              const std::string name = "algo " + std::to_string(ids[i]) + " tile " + std::to_string(tiles[ti])
                + " stages " + std::to_string(stages[si]) + " swizzle " + std::to_string(swz);
              if (f == 0) {
                ++tried;
                if (d == 0) exact.push_back({name, a, {0, 0}});
                else ++differ;
              } else if (d != 0) {
                exact.erase(std::remove_if(exact.begin(), exact.end(),
                                           [&](const Found& x) { return x.name == name; }), exact.end());
                ++differ;
              }
            }
      }
    }
    const Cost ref = cost(reference);
    auto run = [&](Found& x) {
      return [&] { cublasLtMatmul(lt, desc, &one, W, la, A, lb, &zero, C, lc, C, lc, &x.algo, ws, ws_size, 0); };
    };
    for (Found& x : exact) x.c.us = time_us(run(x));
    std::sort(exact.begin(), exact.end(), [](const Found& a, const Found& b) { return a.c.us < b.c.us; });
    if (exact.size() > 15) exact.resize(15);
    for (Found& x : exact) x.c = cost(run(x));
    std::sort(exact.begin(), exact.end(), [](const Found& a, const Found& b) { return a.c.mj < b.c.mj; });
    printf("== %s %dx%d at %d rows: %d configurations, %d differ; cuBLAS %.1f us %.1f mJ; the fastest exact ones:\n",
           names[s], n, k, m, tried, differ, ref.us, ref.mj);
    for (size_t i = 0; i < std::min<size_t>(exact.size(), 8); ++i)
      printf("  %-44s %8.1f us %8.1f mJ (%+.1f%%)\n", exact[i].name.c_str(), exact[i].c.us, exact[i].c.mj,
             100.0 * (exact[i].c.mj / ref.mj - 1));
    fflush(stdout);
  }
  return 0;
}
