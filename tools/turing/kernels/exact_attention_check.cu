// Bit-for-bit check and timing of the fused encoder attention (src/ops/exact_attention.cuh) against the three
// steps it replaces, exactly as production runs them: the cuBLAS MatMul(trans_b, alpha 1/8) of
// Probe::cublas_run, the library's softmax_rows dispatcher (in place, no lengths), then the cuBLAS MatMul of
// the probabilities and the values. Every batch size of an encoder call (20..160 = 1..8 clips x 20 heads),
// 1500 x 1500 x 64, three fills each; must end with TOTAL 0.
// usage: exact_attention_check [timing repetitions, default 10]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/exact_attention.cuh"

template <typename F> float time_us(F run, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 10, m = 1500, n = 1500, d = 64;
  const float alpha = 0.125f;
  Probe p(160, m, n, d);                                   // dQ, dK, and dC for the scores
  __half *V, *VT, *O, *F;
  const size_t vs = 160ull * n * d;
  CK(cudaMalloc(&V, 2 * vs)); CK(cudaMalloc(&VT, 2 * vs)); CK(cudaMalloc(&O, 2 * vs)); CK(cudaMalloc(&F, 2 * vs));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0;
  for (int batch = 20; batch <= 160; batch += 20) {
    auto three_steps = [&] {
      p.cublas_run(alpha, batch);
      at::native::softmax_rows<__half, at::native::SoftMaxForwardEpilogue>(0, p.dC, p.dC, batch * m, n, nullptr, true);
      const float one = 1.f, zero = 0.f;
      CK(cublasGemmStridedBatchedEx(p.h, CUBLAS_OP_N, CUBLAS_OP_N, d, m, n, &one, V, CUDA_R_16F, d, (long long)n * d,
                                    p.dC, CUDA_R_16F, n, (long long)m * n, &zero, O, CUDA_R_16F, d, (long long)m * d,
                                    batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    auto fused = [&] { at::native::exact_attention(p.dQ, p.dK, V, VT, F, batch, m, n, alpha, 0); };
    unsigned long long bad = 0;
    for (int f = 0; f < 3; ++f) {
      fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * d, 23u * f + batch, -6 + f, 1 + f);
      fill<<<1024, 256>>>(p.dK, (size_t)batch * n * d, 211u * f + batch, -7 + f, 1 + f);
      fill<<<1024, 256>>>(V, (size_t)batch * n * d, 307u * f + batch, -8 + 2 * f, 2 + f);
      three_steps(); fused();
      CK(cudaGetLastError());
      CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(O, F, (size_t)batch * m * d, dc);
      unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); bad += x;
    }
    total += bad;
    const float tt = time_us(three_steps, reps), tf = time_us(fused, reps);
    printf("batch %3d: %llu of %llu mismatched, MatMul + SoftMax + MatMul %8.1f us, fused %8.1f us (%.2fx)\n",
           batch, bad, 3ull * batch * m * d, tt, tf, tt / tf);
  }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
