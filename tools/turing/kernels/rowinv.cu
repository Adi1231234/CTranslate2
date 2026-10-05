// Does a decoder GEMM row depend on how many rows the call has? CTranslate2's Dense (cublasGemmEx, fp16,
// COMPUTE_32F, default algorithm: C[m][n] = sum_k A[m][k] W[n][k]) at row counts M a decode step can have (5 per
// clip: 5..40 in a batch of 8 clips, more in a bigger batch). The first rows of A are the same at every M, so
// rows shared by two calls are compared bit for bit (2 random fills). Part 1: which M give the M=40 bits on the
// shared rows (the default's classes). Part 2: each cublasLt configuration with the default's bits at M=40, at
// which M it still gives them (rows shared with M=40), so one fixed configuration could serve every batch size;
// with its time at M=40 and M=160 against the default's.
// usage: rowinv N K
#include <cublasLt.h>
#include "probe_common.h"
#include "probe_data.cuh"

static const int MS[] = {5, 10, 15, 20, 25, 30, 35, 40, 80, 120, 160, 240, 320};
static const int MMAX = 320, MREF = 40, FILLS = 2;

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
  if (bytes) CK(cublasLtMatmulAlgoCapGetAttribute(&algo, attr, v.data(), bytes, &bytes));
  return v;
}

int N, K;
__half *A[FILLS], *W[FILLS], *REF[FILLS], *OUT; void* WS; unsigned long long* DC;
const size_t WS_BYTES = 32 << 20;

unsigned long long diff(const __half* a, const __half* b, int rows) {
  CK(cudaMemset(DC, 0, 8)); count_diff<<<256, 256>>>(a, b, (size_t)rows * N, DC);
  unsigned long long d; CK(cudaMemcpy(&d, DC, 8, cudaMemcpyDeviceToHost)); return d;
}

int main(int argc, char** argv) {
  N = atoi(argv[1]); K = atoi(argv[2]);
  cublasHandle_t h; CK(cublasCreate(&h)); CK(cublasSetWorkspace(h, nullptr, 0));
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  for (int f = 0; f < FILLS; ++f) {
    CK(cudaMalloc(&A[f], 2ull * MMAX * K)); CK(cudaMalloc(&W[f], 2ull * N * K)); CK(cudaMalloc(&REF[f], 2ull * MMAX * N));
    fill<<<256, 256>>>(A[f], (size_t)MMAX * K, 11u + f, -6, 3); fill<<<256, 256>>>(W[f], (size_t)N * K, 97u + f, -12, -3);
  }
  CK(cudaMalloc(&OUT, 2ull * MMAX * N)); CK(cudaMalloc(&WS, WS_BYTES)); CK(cudaMalloc(&DC, 8));
  const float alpha = 1.f, beta = 0.f;
  auto gemm = [&](int f, int m, __half* out) {     // exactly CTranslate2's call (primitives<CUDA>::gemm)
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, m, K, &alpha, W[f], CUDA_R_16F, K, A[f], CUDA_R_16F, K,
                    &beta, out, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  for (int f = 0; f < FILLS; ++f) gemm(f, MREF, REF[f]);
  printf("N %d K %d: default at M=40 %.1f us, at M=160 %.1f us\n  default, rows shared with M=40:", N, K,
         time_us([&] { gemm(0, MREF, OUT); }), time_us([&] { gemm(0, 160, OUT); }));
  for (int m : MS) {
    unsigned long long d = 0;
    for (int f = 0; f < FILLS; ++f) { gemm(f, m, OUT); d += diff(REF[f], OUT, m < MREF ? m : MREF); }
    printf(" %d%s", m, d ? "x" : "=");
  }
  printf("   (= same bits, x different)\n");
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  cublasLtMatrixLayout_t la; CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K));
  auto lay = [&](int m, cublasLtMatrixLayout_t* lb, cublasLtMatrixLayout_t* lc) {
    CK(cublasLtMatrixLayoutCreate(lb, CUDA_R_16F, K, m, K)); CK(cublasLtMatrixLayoutCreate(lc, CUDA_R_16F, N, m, N));
  };
  int ids[256], nids = 0, tried = 0, same = 0;
  CK(cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 256, ids, &nids));
  for (int i = 0; i < nids; ++i) {
    cublasLtMatmulAlgo_t algo;
    CK(cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, ids[i], &algo));
    auto tiles = cap<int>(algo, CUBLASLT_ALGO_CAP_TILE_IDS); if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = cap<int>(algo, CUBLASLT_ALGO_CAP_STAGES_IDS); if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    int splitk_ok = 0, custom_max = 0; uint32_t schemes = 0; size_t b;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, sizeof splitk_ok, &b);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &schemes, sizeof schemes, &b);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_max, sizeof custom_max, &b);
    for (int tile : tiles) for (int stage : stages) for (int sk : {1, 2, 3, 4, 8}) for (uint32_t sc : {0u, 1u, 2u, 4u})
    for (int co = 0; co <= custom_max && co < 4; ++co) {
      if ((sk > 1 && !splitk_ok) || (sk == 1 && sc) || (sk > 1 && sc && !(schemes & sc)) || tried > 3000) continue;
      CK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile));
      CK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage, sizeof stage));
      CK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk));
      CK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &sc, sizeof sc));
      CK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &co, sizeof co));
      auto run = [&](int f, int m, __half* out) -> bool {
        cublasLtMatrixLayout_t lb, lc; lay(m, &lb, &lc);
        cublasLtMatmulHeuristicResult_t res{};
        bool ok = cublasLtMatmulAlgoCheck(lt, op, la, lb, lc, lc, &algo, &res) == CUBLAS_STATUS_SUCCESS
                  && res.workspaceSize <= WS_BYTES
                  && cublasLtMatmul(lt, op, &alpha, W[f], la, A[f], lb, &beta, out, lc, out, lc, &algo, WS, WS_BYTES, 0)
                     == CUBLAS_STATUS_SUCCESS;
        cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        return ok;
      };
      if (!run(0, MREF, OUT)) continue;
      ++tried;
      bool match40 = true;
      for (int f = 0; f < FILLS && match40; ++f) match40 = run(f, MREF, OUT) && !diff(REF[f], OUT, MREF);
      if (!match40) continue;
      ++same;
      printf("  lt algo %d tile %d stages %d splitk %d scheme %u custom %d: M=40 %.1f us, M=160 %.1f us; rows shared with M=40:",
             ids[i], tile, stage, sk, sc, co, time_us([&] { run(0, MREF, OUT); }), time_us([&] { run(0, 160, OUT); }));
      for (int m : MS) {
        unsigned long long d = 0; bool ok = true;
        for (int f = 0; f < FILLS && ok; ++f) { ok = run(f, m, OUT); if (ok) d += diff(REF[f], OUT, m < MREF ? m : MREF); }
        printf(" %d%s", m, !ok ? "-" : d ? "x" : "=");
      }
      printf("\n");
    }
  }
  printf("lt configurations tried %d, same bits as the default at M=40: %d\n", tried, same);
  return 0;
}
