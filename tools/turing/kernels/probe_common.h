// Shared host plumbing for the kernel probes: device buffers and the exact cuBLAS call that
// CTranslate2 makes for attention scores (primitives<CUDA>::gemm_batch_strided, fp16, trans_b,
// COMPUTE_32F as Whisper runs it), so a probe reproduces what production computes.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { auto _s = (x); if ((int)_s != 0) { \
  fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)_s); exit(1); } } while (0)

inline uint16_t half_bits(__half h) { uint16_t b; memcpy(&b, &h, 2); return b; }

struct Probe {
  int batch, m, n, k;
  __half *dQ, *dK, *dC;
  cublasHandle_t h;
  Probe(int batch_, int m_, int n_, int k_) : batch(batch_), m(m_), n(n_), k(k_) {
    CK(cudaMalloc(&dQ, sizeof(__half) * batch * m * k));
    CK(cudaMalloc(&dK, sizeof(__half) * batch * n * k));
    CK(cudaMalloc(&dC, sizeof(__half) * batch * m * n));
    CK(cublasCreate(&h));
  }
  ~Probe() { cublasDestroy(h); cudaFree(dQ); cudaFree(dK); cudaFree(dC); }
  void upload(const std::vector<__half>& Q, const std::vector<__half>& K) {
    CK(cudaMemcpy(dQ, Q.data(), sizeof(__half) * batch * m * k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), sizeof(__half) * batch * n * k, cudaMemcpyHostToDevice));
  }
  std::vector<__half> download() {
    std::vector<__half> C((size_t)batch * m * n);
    CK(cudaMemcpy(C.data(), dC, sizeof(__half) * C.size(), cudaMemcpyDeviceToHost));
    return C;
  }
  // C[b][j][i] = alpha * dot(K[b][i], Q[b][j]): CTranslate2's MatMul(trans_b) with a = Q, b = K.
  void cublas_run(float alpha, int batch_count = -1) {
    const float beta = 0.f;
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                                  dK, CUDA_R_16F, k, (long long)n * k,
                                  dQ, CUDA_R_16F, k, (long long)m * k, &beta,
                                  dC, CUDA_R_16F, n, (long long)m * n,
                                  batch_count < 0 ? batch : batch_count,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  std::vector<__half> cublas(const std::vector<__half>& Q, const std::vector<__half>& K, float alpha) {
    upload(Q, K);
    cublas_run(alpha);
    return download();
  }
};
