// Bit-for-bit check and timing of the encoder GEMM's replica (src/cuda/encoder_gemm_kernel.cuh) against what
// production runs for the Whisper encoder's first feed-forward: the cuBLAS product (CTranslate2's call; on the
// L40S ampere_fp16_s1688gemm_fp16_128x128, an 8-wide chain), then BiasAdd's GELU (ops/bias_add_vec.cuh), against
// the replica with the bias and GELU in its epilogue, at chains 8 and 16 wide (16 must differ on sm_89: the check
// sees a wrong chain), 1, 2, 4, 6 and 8 clips (m = 1500 a clip), 5120 x 1280, three fills. Must end with TOTAL 0
// (the 16-wide counts are reported apart, not in the total).
// usage: encoder_ffn1_check [timing repetitions, default 10]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/encoder_gemm_kernel.cuh"
#include "ops/bias_add_vec.cuh"

using namespace ctranslate2::cuda;
using Launch = void (*)(const __half*, const __half*, __half*, int, int, int, const __half*, cudaStream_t, unsigned*,
                        int);
struct Config { const char* name; Launch narrow, wide; };
#define CONFIG(T, KT, S) {#T "x" #KT "s" #S, enc_gemm_launch<T, KT, S, EncBiasGeluOp, 8>, \
                          enc_gemm_launch<T, KT, S, EncBiasGeluOp, 16>}
static const Config configs[] = {CONFIG(64, 32, 6), CONFIG(64, 64, 4), CONFIG(128, 32, 4), CONFIG(128, 32, 3),
                                 CONFIG(128, 64, 3)};
constexpr int CONFIGS = sizeof configs / sizeof configs[0];

template <typename F> float time_us(F run, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 10, n = 5120, k = 1280, mmax = 8 * 1500;
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *B, *R, *C; unsigned long long* dc;
  CK(cudaMalloc(&A, 2ull * mmax * k)); CK(cudaMalloc(&W, 2ull * n * k)); CK(cudaMalloc(&B, 2ull * n));
  CK(cudaMalloc(&R, 2ull * mmax * n)); CK(cudaMalloc(&C, 2ull * mmax * n)); CK(cudaMalloc(&dc, 8));
  auto diff = [&](size_t count) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(R, C, count, dc);
    unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); return d;
  };
  unsigned long long total = 0;
  for (int clips : {1, 2, 4, 6, 8}) {
    const int m = 1500 * clips;
    auto reference = [&] {
      const float alpha = 1.f, beta = 0.f;
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, W, CUDA_R_16F, k, A, CUDA_R_16F, k, &beta, R,
                      CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      const size_t vecs = (size_t)m * n / 8;
      ctranslate2::ops::bias_add_vec_kernel<ctranslate2::ops::BiasAddVecMode::gelu><<<(vecs + 255) / 256, 256>>>(
        reinterpret_cast<const uint4*>(R), reinterpret_cast<const uint4*>(B), nullptr, reinterpret_cast<uint4*>(R),
        n / 8, vecs);
    };
    unsigned long long narrow[CONFIGS] = {}, wide[CONFIGS] = {};
    for (int f = 0; f < 3; ++f) {
      fill<<<1024, 256>>>(A, (size_t)m * k, 17u * f + clips, -4 + f, 1 + f);
      fill<<<1024, 256>>>(W, (size_t)n * k, 29u * f + clips, -9 + f, -3 + f);
      fill<<<64, 256>>>(B, (size_t)n, 31u * f + clips, -6, 1);
      reference();
      for (int c = 0; c < CONFIGS; ++c) {
        CK(cudaMemset(C, 0xff, 2ull * m * n));
        configs[c].narrow(A, W, C, m, n, k, B, 0, nullptr, 0);
        CK(cudaGetLastError());
        narrow[c] += diff((size_t)m * n);
        configs[c].wide(A, W, C, m, n, k, B, 0, nullptr, 0);
        wide[c] += diff((size_t)m * n);
      }
    }
    printf("%d clips: cuBLAS + BiasAdd GELU %8.1f us\n", clips, time_us(reference, reps));
    for (int c = 0; c < CONFIGS; ++c) {
      total += narrow[c];
      const float t = time_us([&] { configs[c].narrow(A, W, C, m, n, k, B, 0, nullptr, 0); }, reps);
      printf("  %-8s 8-wide %llu of %llu mismatched, %8.1f us | 16-wide %llu mismatched\n", configs[c].name,
             narrow[c], 3ull * m * n, t, wide[c]);
    }
  }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
