// Which cuBLASLt algorithms compute the encoder's Dense product (12000 x 3840 x 1280, fp16, COMPUTE_32F) with
// the same bits as cuBLAS's own choice, and how well does each run beside the decoder? In pipe8 the default
// kernel (64x64, 6 stages, ~48 KB of shared memory per block, two per SM) leaves no room on an SM for the
// decoder's kernels until one of its blocks ends. Lists the heuristic's candidates with their tile, stages and
// split-K, checks each output bit for bit against cublasGemmEx on random data, and times it alone and beside
// a decoder-like chain (40 x 1280 x 1280 GEMMs on 64 weight matrices, high-priority stream).
// usage: lt_probe [n = 3840] [k = 1280]   (build for sm_120)
#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>
#include <utility>
#include "probe_common.h"

static const int M = 12000, dec_weights = 64, dec_count = 2000, enc_count = 16;

static __half* upload(size_t count, uint32_t seed) {
  std::vector<__half> h(count);
  for (size_t i = 0; i < count; ++i) h[i] = __float2half(((int)((i * 2654435761u + seed) % 2001) - 1000) / 2000.f);
  __half* d = nullptr;
  CK(cudaMalloc(&d, sizeof(__half) * count));
  CK(cudaMemcpy(d, h.data(), sizeof(__half) * count, cudaMemcpyHostToDevice));
  return d;
}

static int attr(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t a) {
  int v = 0; size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, a, &v, sizeof v, &written);
  return v;
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 3840, K = argc > 2 ? atoi(argv[2]) : 1280;
  __half *A = upload((size_t)M * K, 1), *W = upload((size_t)N * K, 2), *C = nullptr, *R = nullptr;
  CK(cudaMalloc(&C, sizeof(__half) * M * N)); CK(cudaMalloc(&R, sizeof(__half) * M * N));
  __half *dA = upload(40 * 1280, 3), *dW = upload((size_t)dec_weights * 1280 * 1280, 4), *dC = nullptr;
  CK(cudaMalloc(&dC, sizeof(__half) * 40 * 1280));
  int lo = 0, hi = 0;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t es, ds;
  CK(cudaStreamCreateWithPriority(&es, cudaStreamNonBlocking, lo));
  CK(cudaStreamCreateWithPriority(&ds, cudaStreamNonBlocking, hi));
  cublasHandle_t eh, dh;
  CK(cublasCreate(&eh)); CK(cublasSetStream(eh, es)); CK(cublasCreate(&dh)); CK(cublasSetStream(dh, ds));
  const float one = 1.f, zero = 0.f;
  // As CTranslate2 calls it: C^T (n x m) = W^T(k x n)^T * A^T (k x m), column-major.
  CK(cublasGemmEx(eh, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &zero, R,
                  CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CK(cudaStreamSynchronize(es));
  const std::vector<__half> ref = [&] { std::vector<__half> v((size_t)M * N);
    CK(cudaMemcpy(v.data(), R, sizeof(__half) * v.size(), cudaMemcpyDeviceToHost)); return v; }();

  cublasLtHandle_t lt;
  CK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t t = CUBLAS_OP_T;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws_bytes = 32 << 20;
  void* ws = nullptr;
  CK(cudaMalloc(&ws, ws_bytes));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes));
  cublasLtMatmulHeuristicResult_t found[32];
  int count = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 32, found, &count));

  cudaEvent_t e0, e1, d1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&d1));
  auto decoder = [&] { for (int i = 0; i < dec_count; ++i)
    CK(cublasGemmEx(dh, CUBLAS_OP_T, CUBLAS_OP_N, 1280, 40, 1280, &one, dW + (size_t)(i % dec_weights) * 1280 * 1280,
                    CUDA_R_16F, 1280, dA, CUDA_R_16F, 1280, &zero, dC, CUDA_R_16F, 1280, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT)); };
  // Returns {encoder ms per GEMM, decoder us per GEMM (0 without it)}.
  auto time = [&](const cublasLtMatmulAlgo_t& algo, bool with_decoder) {
    CK(cudaEventRecord(e0, es));
    if (with_decoder) { CK(cudaStreamWaitEvent(ds, e0, 0)); decoder(); CK(cudaEventRecord(d1, ds)); }
    for (int i = 0; i < enc_count; ++i)
      CK(cublasLtMatmul(lt, op, &one, W, la, A, lb, &zero, C, lc, C, lc, &algo, ws, ws_bytes, es));
    CK(cudaEventRecord(e1, es));
    CK(cudaDeviceSynchronize());
    float e = 0, d = 0;
    CK(cudaEventElapsedTime(&e, e0, e1));
    if (with_decoder) CK(cudaEventElapsedTime(&d, e0, d1));
    return std::make_pair(e / enc_count, with_decoder ? 1000 * d / dec_count : 0.f);
  };
  CK(cudaEventRecord(e0, ds)); decoder(); CK(cudaEventRecord(d1, ds)); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(e0, ds)); decoder(); CK(cudaEventRecord(d1, ds)); CK(cudaDeviceSynchronize());
  float dsolo = 0; CK(cudaEventElapsedTime(&dsolo, e0, d1));
  printf("%dx%dx%d: %d candidates; decoder chain alone %.1f us/GEMM\n", M, N, K, count, 1000 * dsolo / dec_count);
  for (int i = 0; i < count; ++i) {
    const cublasLtMatmulAlgo_t& algo = found[i].algo;
    if (cublasLtMatmul(lt, op, &one, W, la, A, lb, &zero, C, lc, C, lc, &algo, ws, ws_bytes, es) != 0) continue;
    CK(cudaStreamSynchronize(es));
    std::vector<__half> out((size_t)M * N);
    CK(cudaMemcpy(out.data(), C, sizeof(__half) * out.size(), cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t j = 0; j < out.size(); ++j) diff += half_bits(out[j]) != half_bits(ref[j]);
    time(algo, false);
    const auto solo = time(algo, false), both = time(algo, true);
    // Work done per ms beside each other, in units of each side's time alone.
    const double eff = (1.0 / both.first * solo.first + 1.0 / both.second * (1000 * dsolo / dec_count)) / 2;
    printf("#%2d algo %3d tile %3d stages %3d splitK %2d swizzle %d ws %7zu: %s  alone %.3f ms | beside: enc %.3f ms"
           " dec %5.1f us | combined %.2fx\n", i, attr(algo, CUBLASLT_ALGO_CONFIG_ID), attr(algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
           attr(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID), attr(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
           attr(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), found[i].workspaceSize,
           diff ? "DIFFERS" : "same bits", solo.first, both.first, both.second, 2 * eff);
  }
  return 0;
}
