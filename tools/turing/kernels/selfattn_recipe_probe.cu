// The arithmetic of cuBLAS's kernels for the Whisper stream's self-attention products (layers/slot_cache.cc: one window's
// 5 beams x 20 heads = 100 entries, one query of 64 dims against its t cached positions at the slots' stride of
// 448 x 64): scores = alpha q k^T and output = p v, for every t 1..448, the candidates of gemv_candidates.cuh and mma
// chains, each against cuBLAS over 2 fills. Prints, by runs of t, the candidates that match every value: what a
// kernel running every window's products at once must reproduce (long13: these per-window calls held ~28% of the
// GPU's time).
//   scores: M an mma.sync m16n8k16 chain over the dims in 16-groups (the query a row of A); T<n><s><tree> n partials
//   over the dims, s = w1/w2/w4/w8 (strided in vectors of w) or c (contiguous), tree d (from the halves), n (from the
//   neighbours) or o (in order)
//   output: M an mma chain over the keys in 16-groups from key 0; R the same with the t % 64 keys first (in 16-groups,
//   zeros past them) then 16-groups from there; T<n><s><tree> as above over the keys (s = w1 or c)
// usage: selfattn_recipe_probe
#include <cstdio>
#include <string>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "gemv_candidates.cuh"

constexpr int kD = 64, kE = 100, kT = 448;
constexpr float kScale = 0.125f;

struct Candidate { int T, w; bool contiguous; int tree; };

static std::string name(const Candidate& c) {
  return "T" + std::to_string(c.T) + (c.contiguous ? "c" : "w" + std::to_string(c.w)) + "dno"[c.tree == 1 ? 0 : c.tree == 2 ? 1 : 2];
}

__global__ void scores_partials(const __half* K, const __half* Q, __half* C, int t, Candidate c) {
  const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= t) return;
  const float sum = reduce_partials(K + ((size_t)e * kT + i) * kD, 1, Q + (size_t)e * kD, kD, c.T, c.w, c.contiguous,
                                    c.tree);
  C[(size_t)e * t + i] = __float2half_rn(kScale * sum);
}

// A warp per (entry, 8 keys): A = the query (row 0), B = the keys' columns.
__global__ void scores_mma(const __half* K, const __half* Q, __half* C, int t) {
  const int e = blockIdx.y, lane = threadIdx.x, g = lane >> 2, tq = lane & 3, i0 = blockIdx.x * 8;
  const __half* kb = K + (size_t)e * kT * kD;
  const __half* q = Q + (size_t)e * kD;
  const __half zero = __float2half(0.f);
  auto kk = [&](int i, int d) { return i < t ? kb[(size_t)i * kD + d] : zero; };
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int d0 = 0; d0 < kD; d0 += 16) {
    const unsigned a0 = g == 0 ? pack(q[d0 + 2 * tq], q[d0 + 2 * tq + 1]) : 0u;
    const unsigned a2 = g == 0 ? pack(q[d0 + 8 + 2 * tq], q[d0 + 8 + 2 * tq + 1]) : 0u;
    const unsigned b0 = pack(kk(i0 + g, d0 + 2 * tq), kk(i0 + g, d0 + 2 * tq + 1));
    const unsigned b1 = pack(kk(i0 + g, d0 + 8 + 2 * tq), kk(i0 + g, d0 + 8 + 2 * tq + 1));
    mma16816(acc, a0, 0u, a2, 0u, b0, b1);
  }
  if (g == 0)
    for (int c = 0; c < 2; ++c) {
      const int i = i0 + 2 * tq + c;
      if (i < t) C[(size_t)e * t + i] = __float2half_rn(kScale * acc[c]);
    }
}

__global__ void output_partials(const __half* V, const __half* P, __half* O, int t, Candidate c) {
  const int e = blockIdx.x, d = threadIdx.x;
  const float sum = reduce_partials(V + (size_t)e * kT * kD + d, kD, P + (size_t)e * t, t, c.T, c.w, c.contiguous,
                                    c.tree);
  O[(size_t)e * kD + d] = __float2half_rn(sum);
}

// A warp per (entry, 8 dims): A = the probabilities (row 0), B = the values; keys in 16-groups from `first`, the
// groups before it covering [0, first) (residue mode: [0, first) in 16-groups, zeros past `first`).
__global__ void output_mma(const __half* V, const __half* P, __half* O, int t, int residue) {
  const int e = blockIdx.y, lane = threadIdx.x, g = lane >> 2, tq = lane & 3, d0 = blockIdx.x * 8;
  const __half* v = V + (size_t)e * kT * kD;
  const __half* p = P + (size_t)e * t;
  const __half zero = __float2half(0.f);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  auto group = [&](int i0, int end) {
    auto pp = [&](int i) { return i < end ? p[i] : zero; };
    auto vv = [&](int i, int d) { return i < end ? v[(size_t)i * kD + d] : zero; };
    const unsigned a0 = g == 0 ? pack(pp(i0 + 2 * tq), pp(i0 + 2 * tq + 1)) : 0u;
    const unsigned a2 = g == 0 ? pack(pp(i0 + 8 + 2 * tq), pp(i0 + 8 + 2 * tq + 1)) : 0u;
    const unsigned b0 = pack(vv(i0 + 2 * tq, d0 + g), vv(i0 + 2 * tq + 1, d0 + g));
    const unsigned b1 = pack(vv(i0 + 8 + 2 * tq, d0 + g), vv(i0 + 8 + 2 * tq + 1, d0 + g));
    mma16816(acc, a0, 0u, a2, 0u, b0, b1);
  };
  int first = 0;
  if (residue) {
    first = t % 64;
    for (int i0 = 0; i0 < first; i0 += 16) group(i0, first);
  }
  for (int i0 = first; i0 < t; i0 += 16) group(i0, t);
  if (g == 0)
    for (int c = 0; c < 2; ++c) O[(size_t)e * kD + d0 + 2 * tq + c] = __float2half_rn(acc[c]);
}

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *Q, *K, *V, *P, *S, *S2, *O, *O2; unsigned long long* dc;
  CK(cudaMalloc(&Q, 2ull * kE * kD)); CK(cudaMalloc(&K, 2ull * kE * kT * kD)); CK(cudaMalloc(&V, 2ull * kE * kT * kD));
  CK(cudaMalloc(&P, 2ull * kE * kT)); CK(cudaMalloc(&S, 2ull * kE * kT)); CK(cudaMalloc(&S2, 2ull * kE * kT));
  CK(cudaMalloc(&O, 2ull * kE * kD)); CK(cudaMalloc(&O2, 2ull * kE * kD)); CK(cudaMalloc(&dc, 8));
  const float scale = kScale, one = 1.f, zero = 0.f;
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  std::vector<Candidate> sc, oc;
  for (int T : {2, 4, 8, 16, 32})
    for (int tree : {1, 2, 0}) {
      for (int w : {1, 2, 4, 8})
        if (T * w <= kD) sc.push_back({T, w, false, tree});
      sc.push_back({T, 1, true, tree});
    }
  for (int T : {2, 4, 8, 16, 32, 64})
    for (int tree : {1, 2, 0}) {
      oc.push_back({T, 1, false, tree});
      oc.push_back({T, 1, true, tree});
    }
  // matches[t]: the candidate names with no mismatch over both fills
  std::vector<std::vector<bool>> sok(kT + 1, std::vector<bool>(sc.size() + 1, true));
  std::vector<std::vector<bool>> ook(kT + 1, std::vector<bool>(oc.size() + 2, true));
  for (int fill_no = 0; fill_no < 2; ++fill_no) {
    fill<<<1024, 256>>>(Q, (size_t)kE * kD, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(K, (size_t)kE * kT * kD, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V, (size_t)kE * kT * kD, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(P, (size_t)kE * kT, 13u + fill_no, -14, -2);
    for (int t = 1; t <= kT; ++t) {
      // as primitives<CUDA>::gemm_batch_strided calls cuBLAS for slot_scores and slot_values
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, K, CUDA_R_16F, kD,
                                    (long long)kT * kD, Q, CUDA_R_16F, kD, kD, &zero, S, CUDA_R_16F, t, t, kE,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, V, CUDA_R_16F, kD,
                                    (long long)kT * kD, P, CUDA_R_16F, t, t, &zero, O, CUDA_R_16F, kD, kD, kE,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      const size_t ns = (size_t)kE * t, no = (size_t)kE * kD;
      for (size_t c = 0; c < sc.size(); ++c) {
        if (!sok[t][c]) continue;
        scores_partials<<<dim3((t + 127) / 128, kE), 128>>>(K, Q, S2, t, sc[c]);
        if (differ(S, S2, ns)) sok[t][c] = false;
      }
      if (sok[t][sc.size()]) {
        scores_mma<<<dim3((t + 7) / 8, kE), 32>>>(K, Q, S2, t);
        if (differ(S, S2, ns)) sok[t][sc.size()] = false;
      }
      for (size_t c = 0; c < oc.size(); ++c) {
        if (!ook[t][c]) continue;
        output_partials<<<kE, kD>>>(V, P, O2, t, oc[c]);
        if (differ(O, O2, no)) ook[t][c] = false;
      }
      for (int residue = 0; residue < 2; ++residue) {
        if (!ook[t][oc.size() + residue]) continue;
        output_mma<<<dim3(kD / 8, kE), 32>>>(V, P, O2, t, residue);
        if (differ(O, O2, no)) ook[t][oc.size() + residue] = false;
      }
      CK(cudaGetLastError());
    }
  }
  auto names = [&](const std::vector<bool>& ok, const std::vector<Candidate>& cs, const char* m1, const char* m2) {
    std::string s;
    for (size_t c = 0; c < cs.size(); ++c) if (ok[c]) s += name(cs[c]) + " ";
    if (ok[cs.size()]) s += std::string(m1) + " ";
    if (m2 && ok[cs.size() + 1]) s += std::string(m2) + " ";
    return s.empty() ? std::string("NONE") : s;
  };
  for (int which = 0; which < 2; ++which) {
    printf("%s by t:\n", which ? "output" : "scores");
    std::string prev; int from = 1;
    for (int t = 1; t <= kT + 1; ++t) {
      const std::string cur = t > kT ? "" : which ? names(ook[t], oc, "M", "R") : names(sok[t], sc, "M", nullptr);
      if (t > 1 && cur != prev) { printf("  t %3d-%3d: %s\n", from, t - 1, prev.c_str()); from = t; }
      prev = cur;
    }
  }
  return 0;
}
