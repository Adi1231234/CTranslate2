// Recovers the fp32 summation order of the cuBLAS kernel behind CTranslate2's Whisper cross-attention
// scores (cublasGemmStridedBatchedEx, fp16 in/out, COMPUTE_32F, k = 64, m = beams, n = 1500 frames).
// Every output is a 64-term dot product of halves; products are exact in fp32, so only the order of
// the additions (a binary tree) and the final alpha/rounding can differ between implementations.
// Tree test: terms +2^15 at a, -2^15 at b, 2^-10 at c, zeros elsewhere. c survives the fp32 sum only
// if it joins after a and b have cancelled, so {c : lost} + {a, b} is the leaf set under the node
// where a and b meet. Collecting that set for every pair (a, b) gives every node of the tree.
// usage: qk_probe   (prints the node leaf sets as 64-bit masks, the fp16/fp32 check, alpha rounding)
#include <algorithm>
#include <bitset>
#include <cmath>
#include <map>
#include "probe_common.h"

int main() {
  const int batch = 160, m = 5, n = 1500, k = 64;
  const int slots = batch * n;
  std::vector<int> ta, tb, tc;
  for (int a = 0; a < k; ++a)
    for (int b = a + 1; b < k; ++b)
      for (int c = 0; c < k; ++c)
        if (c != a && c != b) { ta.push_back(a); tb.push_back(b); tc.push_back(c); }
  const int tests = (int)ta.size();
  std::vector<__half> Q((size_t)batch * m * k, __float2half(1.f));
  Probe p(batch, m, n, k);
  std::vector<uint8_t> lost[2];
  for (int run = 0; run < 2; ++run) {          // the same tests at two different (batch, key) slots
    std::vector<__half> K((size_t)slots * k, __float2half(0.f));
    const int shift = run * 100003;
    for (int t = 0; t < tests; ++t) {
      __half* row = &K[(size_t)((t + shift) % slots) * k];
      row[ta[t]] = __float2half(32768.f); row[tb[t]] = __float2half(-32768.f);
      row[tc[t]] = __float2half(1.f / 1024.f);
    }
    std::vector<__half> C = p.cublas(Q, K, 0.125f);
    lost[run].resize(tests);
    int beam_mismatch = 0;
    for (int t = 0; t < tests; ++t) {
      const int s = (t + shift) % slots, bb = s / n, i = s % n;
      const float v = __half2float(C[((size_t)bb * m) * n + i]);
      for (int j = 1; j < m; ++j)
        beam_mismatch += __half2float(C[((size_t)bb * m + j) * n + i]) != v;
      lost[run][t] = v == 0.f;
    }
    printf("run %d: beam mismatches %d\n", run, beam_mismatch);
  }
  int slot_mismatch = 0;
  std::map<std::pair<int, int>, uint64_t> node;
  for (int t = 0; t < tests; ++t) {
    slot_mismatch += lost[0][t] != lost[1][t];
    uint64_t& s = node[{ta[t], tb[t]}];
    s |= (1ull << ta[t]) | (1ull << tb[t]);
    if (lost[0][t]) s |= 1ull << tc[t];
  }
  std::map<uint64_t, int> uniq;
  for (auto& e : node) uniq[e.second]++;
  printf("slot mismatches %d, distinct nodes %zu (a binary tree over 64 leaves has 63)\n",
         slot_mismatch, uniq.size());
  std::vector<std::pair<int, uint64_t>> byk;
  for (auto& e : uniq) byk.push_back({(int)std::bitset<64>(e.first).count(), e.first});
  std::sort(byk.begin(), byk.end());
  for (auto& e : byk) printf("node %2d leaves %016llx\n", e.first, (unsigned long long)e.second);
  // fp32 or fp16 partial sums: 2048 + 1 - 2048 is exactly 1 in fp32 in any order.
  {
    std::vector<__half> K((size_t)slots * k, __float2half(0.f));
    for (int a = 0; a < k; ++a) {
      __half* row = &K[(size_t)a * k];
      row[a] = __float2half(2048.f); row[(a + 1) % k] = __float2half(1.f);
      row[(a + 33) % k] = __float2half(-2048.f);
    }
    std::vector<__half> C = p.cublas(Q, K, 1.f);
    int ones = 0;
    for (int a = 0; a < k; ++a) ones += __half2float(C[a]) == 1.f;
    printf("fp32 partial sums: %d of %d rows exact\n", ones, k);
  }
  // alpha before or after the fp16 rounding: sum = 2^-12 + 1.25 * 2^-22, alpha = 1/8.
  {
    std::vector<__half> K((size_t)slots * k, __float2half(0.f));
    K[0] = __float2half(ldexpf(1.f, -12)); K[1] = __float2half(ldexpf(1.f, -22));
    K[2] = __float2half(ldexpf(1.f, -24));
    std::vector<__half> C = p.cublas(Q, K, 0.125f);
    const float acc = ldexpf(1.f, -12) + ldexpf(1.f, -22) + ldexpf(1.f, -24);
    printf("alpha rounding: got %04x, alpha-then-round %04x, round-then-alpha %04x\n", half_bits(C[0]),
           half_bits(__float2half_rn(acc * 0.125f)),
           half_bits(__float2half_rn(__half2float(__float2half_rn(acc)) * 0.125f)));
  }
  return 0;
}
