// Bit-for-bit check and timing of src/cuda/attention_scores_k64.cuh against the cuBLAS call that
// CTranslate2 makes for Whisper cross-attention scores, over every query count 1..8 and every batch
// size 1..1024, on random data with a wide dynamic range and on signed zeros.
// Shapes outside the route (attention_scores_k64_verified_shape, e.g. batch 1, where cuBLAS runs
// another kernel) are counted apart as a control: the gate needs TOTAL (routed shapes) = 0.
// usage: qk_check      -> per-(m, batch) mismatch counts and the timings
#include "probe_common.h"
#include "probe_data.cuh"
#include "../../../src/cuda/attention_scores_k64.cuh"

using namespace ctranslate2::cuda;

int main() {
  const int max_batch = 1024, max_m = 8, n = 1500, k = 64;
  Probe p(max_batch, max_m, n, k);
  __half* dC2;
  unsigned long long* dcount;
  CK(cudaMalloc(&dC2, sizeof(__half) * max_batch * max_m * n));
  CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  int version = 0;
  CK(cublasGetVersion(p.h, &version));
  printf("cuBLAS %d\n", version);
  auto check = [&](int batch, int m, float alpha) {
    p.m = m;
    p.cublas_run(alpha, batch);
    attention_scores_k64(p.dQ, p.dK, dC2, batch, m, n, alpha, 0);
    CK(cudaMemset(dcount, 0, sizeof(unsigned long long)));
    count_diff<<<256, 256>>>(p.dC, dC2, (size_t)batch * m * n, dcount);
    unsigned long long diff = 0;
    CK(cudaMemcpy(&diff, dcount, sizeof(diff), cudaMemcpyDeviceToHost));
    return diff;
  };
  // Signed zeros: every product -0.
  set_bits<<<64, 256>>>(p.dK, (size_t)n * k, 0x8000);
  set_bits<<<64, 256>>>(p.dQ, (size_t)max_m * k, 0x3c00);          // 1.0
  printf("signed zeros: %llu mismatches\n", check(1, 5, 0.125f));
  std::vector<int> batches;                      // every batch size the route accepts, and 1
  for (int b = 1; b <= max_batch; ++b) batches.push_back(b);
  unsigned long long total = 0, outputs = 0, control = 0;
  for (int m = 1; m <= max_m; ++m) {
    unsigned long long m_total = 0;
    for (int batch : batches) {
      fill_case(p.dK, p.dQ, batch, m, n, k);
      const unsigned long long d = check(batch, m, 0.125f);
      const bool routed = attention_scores_k64_verified_shape(batch, m, n, k);
      if (d) printf("  mismatch m=%d batch=%d%s: %llu\n", m, batch, routed ? "" : " (not routed)", d);
      if (routed) { m_total += d; outputs += (unsigned long long)batch * m * n; } else control += d;
    }
    printf("m=%d: %llu mismatches\n", m, m_total);
    total += m_total;
  }
  printf("control, shapes not routed: %llu mismatches\n", control);
  printf("TOTAL %llu mismatches over %llu outputs of the routed shapes\n", total, outputs);
  // Timing at the production shape: 8 clips x 20 heads, 5 beams.
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int batch = 160, m = 5, iters = 200;
  p.m = m;
  for (int which = 0; which < 2; ++which) {
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it) {
      if (which == 0) p.cublas_run(0.125f, batch);
      else attention_scores_k64(p.dQ, p.dK, dC2, batch, m, n, 0.125f, 0);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("%s: %.1f us per call (batch %d, m %d, n %d)\n", which ? "attention_scores_k64" : "cublas",
           1000.f * ms / iters, batch, m, n);
  }
  return 0;
}
