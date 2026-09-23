// Replays one (m, batch) case of qk_check and dissects every output where attention_scores_k64 and
// cuBLAS disagree: position, both results, and host recomputations of the candidate orders.
// usage: qk_diff <m> <batch>
#include <cmath>
#include "probe_common.h"
#include "probe_data.cuh"
#include "../../../src/cuda/attention_scores_k64.cuh"

using namespace ctranslate2::cuda;

int main(int argc, char** argv) {
  const int m = atoi(argv[1]), batch = atoi(argv[2]), n = 1500, k = 64;
  Probe p(batch, m, n, k);
  __half* dC2;
  CK(cudaMalloc(&dC2, sizeof(__half) * batch * m * n));
  fill_case(p.dK, p.dQ, batch, m, n, k);
  p.cublas_run(0.125f);
  attention_scores_k64(p.dQ, p.dK, dC2, batch, m, n, 0.125f, 0);
  std::vector<__half> C = p.download(), C2(C.size()), Q((size_t)batch * m * k), K((size_t)batch * n * k);
  CK(cudaMemcpy(C2.data(), dC2, sizeof(__half) * C2.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Q.data(), p.dQ, sizeof(__half) * Q.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(K.data(), p.dK, sizeof(__half) * K.size(), cudaMemcpyDeviceToHost));
  int shown = 0, total = 0;
  for (int b = 0; b < batch; ++b)
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < n; ++i) {
        const size_t o = ((size_t)b * m + j) * n + i;
        if (half_bits(C[o]) == half_bits(C2[o])) continue;
        ++total;
        if (shown++ >= 8) continue;
        const __half* kr = &K[((size_t)b * n + i) * k];
        const __half* qr = &Q[((size_t)b * m + j) * k];
        float t[64]; double exact = 0;
        for (int e = 0; e < 64; ++e) { t[e] = __half2float(kr[e]) * __half2float(qr[e]); exact += t[e]; }
        float ours = 0, seq = 0;
        for (int r = 0; r < 16; ++r) ours += ((t[r] + t[r + 16]) + t[r + 32]) + t[r + 48];
        for (int e = 0; e < 64; ++e) seq += t[e];
        printf("b=%d j=%d i=%d (i%%8=%d, block %d): cublas %04x (%.9g) ours %04x (%.9g) | host ours %.9g"
               " seq %.9g exact %.12g -> ours*a %04x seq*a %04x exact*a %04x\n",
               b, j, i, i % 8, i / 8, half_bits(C[o]), __half2float(C[o]), half_bits(C2[o]),
               __half2float(C2[o]), ours, seq, exact, half_bits(__float2half_rn(0.125f * ours)),
               half_bits(__float2half_rn(0.125f * seq)), half_bits(__float2half_rn((float)(0.125 * exact))));
      }
  printf("m=%d batch=%d: %d mismatches\n", m, batch, total);
  return 0;
}
