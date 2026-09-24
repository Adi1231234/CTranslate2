// Which summation order do cuBLAS's kernels use for the Whisper encoder's attention matmuls, as
// CTranslate2 calls them (gemm_batch_strided, fp16, COMPUTE_32F)?
//   QK: S[b][m][n] = 0.125 * sum_k Q[b][m][k] K[b][n][k]  (1500 x 1500, k 64, trans_b)
//   AV: O[b][m][d] = sum_k P[b][m][k] V[b][k][d]          (1500 x 64, k 1500)
// A candidate is a list of k-groups (start, count <= 8; the rest of a group is zero) accumulated in
// order from zero with mma.sync m16n8k8, then alpha * sum rounded to half. For k = 1500 the
// candidates differ in where the partial group sits: last (natural) or first (residue-first, as a
// CUTLASS mainloop that predicates its first k-tile of 32 or 64). 0 mismatches = the order.
// XAV=1: the decoder's cross-attention AV instead: rows = 5 beams per (clip, head), batch = clips
// x 20 heads (O[b][beam][d] = sum_k P[b][beam][k] V[b][k][d], k 1500).
// usage: attn_probe [batch=4]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

__device__ __forceinline__ void mma1688(float* d, unsigned a0, unsigned a1, unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ unsigned pk(__half a, __half b) {
  __half2 h = __halves2half2(a, b);
  return *reinterpret_cast<unsigned*>(&h);
}

// One warp per 16 x 8 tile of one batch entry. A [M, K] row-major; B element (k, n) at
// B[n * bk + k * bn] (QK: K rows, bk = 1, bn = K; AV: V rows, bk = N... see callers).
__global__ void replica(const __half* A, const __half* B, __half* C, int M, int N, int K,
                        long long sa, long long sb, long long sc, int b_row, int b_col,
                        const int2* groups, int ngroups, float alpha) {
  const int lane = threadIdx.x, g4 = lane / 4, t4 = lane % 4;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
  const __half* a = A + blockIdx.z * sa;
  const __half* b = B + blockIdx.z * sb;
  const __half z = __float2half(0.f);
  float acc[4] = {};
  for (int g = 0; g < ngroups; ++g) {
    const int start = groups[g].x, count = groups[g].y;
    auto kv = [&](int i) { return i < count ? start + i : -1; };   // position i of the group
    auto av = [&](int row, int i) { const int k = kv(i); return row < M && k >= 0 ? a[(size_t)row * K + k] : z; };
    auto bv = [&](int col, int i) { const int k = kv(i); return col < N && k >= 0 ? b[(size_t)col * b_col + (size_t)k * b_row] : z; };
    const int r0 = m0 + g4, r1 = r0 + 8, c = n0 + g4;
    mma1688(acc, pk(av(r0, 2 * t4), av(r0, 2 * t4 + 1)), pk(av(r1, 2 * t4), av(r1, 2 * t4 + 1)),
            pk(bv(c, 2 * t4), bv(c, 2 * t4 + 1)));
  }
  __half* out = C + blockIdx.z * sc;
  for (int e = 0; e < 4; ++e) {
    const int row = m0 + g4 + 8 * (e / 2), col = n0 + 2 * t4 + e % 2;
    if (row < M && col < N) out[(size_t)row * N + col] = __float2half_rn(alpha * acc[e]);
  }
}

std::vector<int2> natural(int K) {
  std::vector<int2> g;
  for (int s = 0; s < K; s += 8) g.push_back({s, K - s < 8 ? K - s : 8});
  return g;
}

std::vector<int2> residue_first(int K, int tile) {
  std::vector<int2> g;
  const int r = K % tile ? K % tile : tile;
  for (int s = 0; s < r; s += 8) g.push_back({s, r - s < 8 ? r - s : 8});
  for (int s = r; s < K; s += 8) g.push_back({s, 8});
  return g;
}

int main(int argc, char** argv) {
  const int batch = argc > 1 ? atoi(argv[1]) : 4, T = 1500, D = 64;
  const bool xav = getenv("XAV") != nullptr;
  const int rows = xav ? 5 : T;                   // query rows of the AV matmul
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *Q, *Kt, *S, *R, *P, *V, *O;
  CK(cudaMalloc(&Q, 2ull * batch * T * D)); CK(cudaMalloc(&Kt, 2ull * batch * T * D));
  CK(cudaMalloc(&S, 2ull * batch * T * T)); CK(cudaMalloc(&R, 2ull * batch * T * T));
  CK(cudaMalloc(&P, 2ull * batch * T * T)); CK(cudaMalloc(&V, 2ull * batch * T * D)); CK(cudaMalloc(&O, 2ull * batch * T * D));
  int2* dg; CK(cudaMalloc(&dg, sizeof(int2) * 256));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  auto diff = [&](const __half* x, const __half* y, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(x, y, n, dc);
    unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); return d;
  };
  auto run = [&](const std::vector<int2>& g, const __half* A, const __half* B, __half* C, int M, int N, int K,
                 int b_row, int b_col, float alpha) {
    CK(cudaMemcpy(dg, g.data(), sizeof(int2) * g.size(), cudaMemcpyHostToDevice));
    replica<<<dim3((N + 7) / 8, (M + 15) / 16, batch), 32>>>(A, B, C, M, N, K, (long long)M * K,
      (long long)(b_row == 1 ? N * K : K * N), (long long)M * N, b_row, b_col, dg, (int)g.size(), alpha);
  };
  for (int trial = 0; trial < 2; ++trial) {
    fill<<<1024, 256>>>(Q, (size_t)batch * T * D, 5u + trial, -6, 3);
    fill<<<1024, 256>>>(Kt, (size_t)batch * T * D, 9u + trial, -6, 3);
    fill<<<1024, 256>>>(P, (size_t)batch * T * T, 17u + trial, -14, 0);
    fill<<<1024, 256>>>(V, (size_t)batch * T * D, 23u + trial, -6, 3);
    const float qk_alpha = 0.125f, one = 1.f, zero = 0.f;
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &qk_alpha, Kt, CUDA_R_16F, D, (long long)T * D,
                                  Q, CUDA_R_16F, D, (long long)T * D, &zero, S, CUDA_R_16F, T, (long long)T * T, batch,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, rows, T, &one, V, CUDA_R_16F, D, (long long)T * D,
                                  P, CUDA_R_16F, T, (long long)rows * T, &zero, O, CUDA_R_16F, D, (long long)rows * D, batch,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    // QK: B = K rows [n][k]: element (k, n) at n * D + k.
    run(natural(D), Q, Kt, R, T, T, D, 1, D, qk_alpha);
    printf("trial %d QK natural: %llu mismatches of %d\n", trial, diff(S, R, (size_t)batch * T * T), batch * T * T);
    // AV: B = V [k][d]: element (k, d) at k * D + d.
    struct { const char* name; std::vector<int2> g; } av[] = {
      {"natural", natural(T)}, {"residue-first 32", residue_first(T, 32)}, {"residue-first 64", residue_first(T, 64)}};
    for (auto& c : av) {
      run(c.g, P, V, R, rows, D, T, D, 1, 1.f);
      printf("trial %d AV %s: %llu mismatches of %d\n", trial, c.name, diff(O, R, (size_t)batch * rows * D), batch * rows * D);
    }
  }
  return 0;
}
