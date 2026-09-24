// Bit-for-bit check and timing of the fused encoder attention scores + softmax (src/ops/
// attention_scores_softmax.cuh) against the two steps it replaces, exactly as production runs them: the
// cuBLAS MatMul(trans_b, alpha 1/8) of Probe::cublas_run, then the library's softmax_rows dispatcher (in
// place, no lengths). Every batch size of an encoder call (20..160 = 1..8 clips x 20 heads), 1500 x 1500,
// three fills each; must end with TOTAL 0.
// usage: scores_softmax_check [timing repetitions, default 10]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/attention_scores_softmax.cuh"

template <typename F> float time_us(F run, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 10, m = 1500, n = 1500;
  const float alpha = 0.125f;
  Probe p(160, m, n, 64);
  __half* F; CK(cudaMalloc(&F, sizeof(__half) * 160 * m * n));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0;
  for (int batch = 20; batch <= 160; batch += 20) {
    auto two_steps = [&] {
      p.cublas_run(alpha, batch);
      at::native::softmax_rows<__half, at::native::SoftMaxForwardEpilogue>(
        0, p.dC, p.dC, batch * m, n, nullptr, /*warp=*/true);
    };
    auto fused = [&] { at::native::attention_scores_softmax(p.dQ, p.dK, F, batch, m, n, alpha, 0); };
    unsigned long long bad = 0;
    for (int f = 0; f < 3; ++f) {
      fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * 64, 23u * f + batch, -6 + f, 2 + f);
      fill<<<1024, 256>>>(p.dK, (size_t)batch * n * 64, 211u * f + batch, -7 + f, 1 + f);
      two_steps(); fused();
      CK(cudaGetLastError());
      CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(p.dC, F, (size_t)batch * m * n, dc);
      unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad += d;
    }
    total += bad;
    const float tt = time_us(two_steps, reps), tf = time_us(fused, reps);
    printf("batch %3d: %llu of %llu mismatched, MatMul + SoftMax %8.1f us, fused %8.1f us (%.2fx)\n",
           batch, bad, 3ull * batch * m * n, tt, tf, tt / tf);
  }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
