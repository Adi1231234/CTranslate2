// Search cublasLt for configurations of CTranslate2's decoder GEMMs that give the same bits as the
// cublasGemmEx call CTranslate2 makes (Dense: C[m][n] = sum_k A[m][k] * W[n][k], fp16, COMPUTE_32F)
// and run faster: every algorithm id x tile x split-K x reduction scheme cublasLt accepts, compared
// bit for bit on random data (3 fills) and timed. Prints the default's time, then each identical
// configuration (algo, tile, split-K, scheme, stages, custom option) with its time.
// usage: lt_search M N K
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublasLt.h>
#include "probe_common.h"
#include "probe_data.cuh"

#define LT(x) CK(x)

template <typename F> float time_us(F run, int reps = 50) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

template <typename T> std::vector<T> cap(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t bytes = 0;
  cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes);
  std::vector<T> v(bytes / sizeof(T));
  if (bytes) LT(cublasLtMatmulAlgoCapGetAttribute(&algo, attr, v.data(), bytes, &bytes));
  return v;
}

int main(int argc, char** argv) {
  const int M = argc > 3 ? atoi(argv[1]) : 40, N = argc > 3 ? atoi(argv[2]) : 1280, K = argc > 3 ? atoi(argv[3]) : 1280;
  cublasHandle_t h; CK(cublasCreate(&h));
  cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
  __half *A, *W, *C, *R; void* ws; const size_t ws_bytes = 32 << 20;
  CK(cudaMalloc(&A, 2ull * M * K)); CK(cudaMalloc(&W, 2ull * N * K));
  CK(cudaMalloc(&C, 2ull * M * N * 3)); CK(cudaMalloc(&R, 2ull * M * N)); CK(cudaMalloc(&ws, ws_bytes));
  unsigned long long* dcount; CK(cudaMalloc(&dcount, 8));
  const float alpha = 1.f, beta = 0.f;
  auto gemm = [&](__half* out) {
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                    &beta, out, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  for (int t = 0; t < 3; ++t) {                           // references for 3 fills
    fill<<<256, 256>>>(A, (size_t)M * K, 11u + t, -6, 3); fill<<<256, 256>>>(W, (size_t)N * K, 97u + t, -12, -3);
    gemm(C + (size_t)t * M * N);
  }
  printf("M %d N %d K %d: default %.1f us\n", M, N, K, time_us([&] { gemm(R); }));
  cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  cublasLtMatrixLayout_t la, lb, lc;
  LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K)); LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
  int ids[256], nids = 0;
  LT(cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 256, ids, &nids));
  int tried = 0, same = 0;
  for (int i = 0; i < nids; ++i) {
    cublasLtMatmulAlgo_t algo; LT(cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, ids[i], &algo));
    auto tiles = cap<int>(algo, CUBLASLT_ALGO_CAP_TILE_IDS); if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = cap<int>(algo, CUBLASLT_ALGO_CAP_STAGES_IDS); if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    int splitk_ok = 0, custom_max = 0; uint32_t schemes = 0; size_t b;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, sizeof splitk_ok, &b);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &schemes, sizeof schemes, &b);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_max, sizeof custom_max, &b);
    for (int tile : tiles) for (int stage : stages) for (int sk : {1, 2, 4, 8}) for (uint32_t sc : {0u, 1u, 2u, 4u})
      for (int co = 0; co <= custom_max && co < 4; ++co) {
        if ((sk > 1 && !splitk_ok) || (sk == 1 && sc) || (sk > 1 && sc && !(schemes & sc)) || tried > 4000) continue;
        LT(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile));
        LT(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage, sizeof stage));
        LT(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk));
        LT(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &sc, sizeof sc));
        LT(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &co, sizeof co));
        cublasLtMatmulHeuristicResult_t res{};
        if (cublasLtMatmulAlgoCheck(lt, op, la, lb, lc, lc, &algo, &res) != CUBLAS_STATUS_SUCCESS || res.workspaceSize > ws_bytes) continue;
        ++tried;
        auto run = [&](__half* out) {
          return cublasLtMatmul(lt, op, &alpha, W, la, A, lb, &beta, out, lc, out, lc, &algo, ws, ws_bytes, 0);
        };
        unsigned long long bad = 0; bool ok = true;
        for (int t = 0; t < 3 && ok; ++t) {
          fill<<<256, 256>>>(A, (size_t)M * K, 11u + t, -6, 3); fill<<<256, 256>>>(W, (size_t)N * K, 97u + t, -12, -3);
          ok = run(R) == CUBLAS_STATUS_SUCCESS;
          CK(cudaMemset(dcount, 0, 8)); count_diff<<<256, 256>>>(C + (size_t)t * M * N, R, (size_t)M * N, dcount);
          unsigned long long d; CK(cudaMemcpy(&d, dcount, 8, cudaMemcpyDeviceToHost)); bad += d;
        }
        if (!ok || bad) continue;
        ++same;
        printf("  same bits: algo %d tile %d stages %d splitk %d scheme %u custom %d ws %zu: %.1f us\n",
               ids[i], tile, stage, sk, sc, co, res.workspaceSize, time_us([&] { run(R); }));
      }
  }
  printf("configurations tried %d, bit-identical %d\n", tried, same);
  return 0;
}
