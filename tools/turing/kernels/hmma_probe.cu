// Which summation order does cuBLAS use for CTranslate2's decoder Dense layers on this GPU? For every
// decoder shape it runs the exact cuBLAS call (C = A W^T, fp16, COMPUTE_32F, alpha 1, beta 0) and compares
// it bit for bit with candidate orders built from the same tensor-core instruction, on four fills:
//   chain KI        one chain from zero over k in KI-wide groups (mma.sync m16n8kKI), out = half(acc)
//   serial S/G      split-K in S slices of ceil(K / S) rounded up to G, each a KI=16 chain from zero;
//                   out = half(acc0), then out = half(acc_s + out) (a serial split-K epilogue, beta 1)
//   fp32 S/G        the slices' accumulators summed forward in fp32, out = half(sum)
//   halves S/G      each slice rounded to half, summed forward in fp32, out = half(sum)
// usage: hmma_probe [max M, default 48] -> per shape the first matching candidate; "NONE" if none
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

__device__ __forceinline__ unsigned pair_at(const __half* p, int rows, int r, size_t stride, int c) {
  return r < rows ? *reinterpret_cast<const unsigned*>(p + r * stride + c) : 0u;
}

template <int KI>
__device__ __forceinline__ void mma_step(float* d, const __half* A, const __half* W, int M, int N, int K,
                                         int m0, int n0, int k, int g, int t) {
  const unsigned b0 = pair_at(W, N, n0 + g, K, k + 2 * t);
  if (KI == 16) {
    const unsigned b1 = pair_at(W, N, n0 + g, K, k + 8 + 2 * t);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(pair_at(A, M, m0 + g, K, k + 2 * t)), "r"(pair_at(A, M, m0 + g + 8, K, k + 2 * t)),
                   "r"(pair_at(A, M, m0 + g, K, k + 8 + 2 * t)), "r"(pair_at(A, M, m0 + g + 8, K, k + 8 + 2 * t)),
                   "r"(b0), "r"(b1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(pair_at(A, M, m0 + g, K, k + 2 * t)), "r"(pair_at(A, M, m0 + g + 8, K, k + 2 * t)), "r"(b0));
  }
}

__device__ __forceinline__ float rh(float x) { return __half2float(__float2half_rn(x)); }

// One warp per 16 x 8 tile of C; mode 0 serial, 1 fp32 partials, 2 half partials (same for S = 1).
template <int KI>
__global__ void order_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K, int S, int G, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, n0 = blockIdx.x * 8, m0 = blockIdx.y * 16;
  const int slice = ((K + S - 1) / S + G - 1) / G * G;
  float out[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < S; ++s) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = s * slice; k < min(K, (s + 1) * slice); k += KI)
      mma_step<KI>(acc, A, W, M, N, K, m0, n0, k, g, t);
    for (int i = 0; i < 4; ++i)
      out[i] = mode == 0 ? rh(s == 0 ? acc[i] : acc[i] + out[i])
             : mode == 1 ? (s == 0 ? acc[i] : out[i] + acc[i])
                         : (s == 0 ? rh(acc[i]) : out[i] + rh(acc[i]));
  }
  for (int i = 0; i < 4; ++i) {
    const int row = m0 + g + 8 * (i / 2), col = n0 + 2 * t + i % 2;
    if (row < M && col < N) C[(size_t)row * N + col] = __float2half_rn(out[i]);
  }
}

struct Cand { std::string name; int ki, s, g, mode; };

int main(int argc, char** argv) {
  const int max_m = argc > 1 ? atoi(argv[1]) : 48;
  const int nk[5][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {1280, 5120}, {51866, 1280}};
  std::vector<Cand> cands = {{"chain 16", 16, 1, 16, 0}, {"chain 8", 8, 1, 8, 0}};
  const char* modes[3] = {"serial", "fp32", "halves"};
  for (int s : {2, 3, 4, 5, 6, 8})
    for (int g : {16, 32, 64, 128, 256})
      for (int mode = 0; mode < 3; ++mode)
        cands.push_back({std::string(modes[mode]) + " " + std::to_string(s) + "/" + std::to_string(g), 16, s, g, mode});
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *C, *R;
  CK(cudaMalloc(&A, 2ull * 64 * 5120)); CK(cudaMalloc(&W, 2ull * 51866 * 1280));
  CK(cudaMalloc(&C, 2ull * 64 * 51866)); CK(cudaMalloc(&R, 2ull * 64 * 51866));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  int unmatched = 0;
  for (auto& s : nk)
    for (int M = 1; M <= max_m; ++M) {
      const int N = s[0], K = s[1];
      std::vector<unsigned long long> bad(cands.size(), 0);
      for (size_t upto : {size_t(1), cands.size()}) {       // chain 16 first; search the rest only if it fails
        if (upto > 1 && bad[0] == 0) break;
        std::fill(bad.begin(), bad.end(), 0ull);
        for (int f = 0; f < 4; ++f) {
          fill<<<256, 256>>>(A, (size_t)M * K, 17u * f + M, -9 + 2 * f, 1 + f);
          fill<<<1024, 256>>>(W, (size_t)N * K, 101u * f + N + K, -15 + f, -3 + 2 * f);
          const float alpha = 1.f, beta = 0.f;
          CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                          &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
          for (size_t c = 0; c < upto; ++c) {
            const dim3 grid((N + 7) / 8, (M + 15) / 16);
            if (cands[c].ki == 16) order_kernel<16><<<grid, 32>>>(A, W, R, M, N, K, cands[c].s, cands[c].g, cands[c].mode);
            else order_kernel<8><<<grid, 32>>>(A, W, R, M, N, K, 1, 8, 0);
            CK(cudaGetLastError());
            CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(C, R, (size_t)M * N, dc);
            unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad[c] += d;
          }
        }
        if (upto == 1) std::fill(bad.begin() + 1, bad.end(), ~0ull);
      }
      const size_t best = std::min_element(bad.begin(), bad.end()) - bad.begin();
      if (bad[best] != 0) ++unmatched;
      printf("M %2d N %5d K %4d: %s%s (%llu of %d mismatched)\n", M, N, K, bad[best] ? "NONE, closest " : "",
             cands[best].name.c_str(), bad[best], 4 * M * N);
    }
  printf("UNMATCHED %d shapes\n", unmatched);
  return 0;
}
