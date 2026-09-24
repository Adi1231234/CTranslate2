// Which summation order does cuBLAS use for CTranslate2's decoder GEMMs (Dense: C[m][n] =
// sum_k A[m][k] * W[n][k], fp16, COMPUTE_32F, alpha 1)? Candidate replicas built from the same
// tensor-core instruction (mma.sync m16n8k8 f32.f16.f16.f32) are compared with cuBLAS bit for bit.
// A candidate splits the 8-wide k-groups into nz split-K ranges, each into ns chains (ns = 2: group g
// of a range goes to chain (g / w) % 2), accumulates each chain in order from zero, adds the two
// chains, optionally rounds that partial to half (hp 1, as cuBLAS's split-K workspace in fp16), then
// reduces the nz partials in fp32 forward (red 0), pairwise (red 1) or backward (red 2) and rounds
// once to half. Split-K ranges are contiguous (il 0) or interleaved by 64-wide k-tiles (il 1).
// The candidate with 0 mismatches on every trial is the order to replicate.
// ORDER=<nz>: instead of random data, one nonzero product per split-K range of nz (A = 1 at the
// first k of each range, W = +-2^[-24, 15] there, subnormals included), so that the order of the
// partial reduction decides the result wherever the partials do not add exactly (spread > 24 bits).
// usage: gemm_probe [M N K ...]   (default: the decoder shapes at 40 rows)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

struct Candidate { int nz, ns, w, red, hp, il; };

__device__ __forceinline__ void mma1688(float* d, const unsigned* a, unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ unsigned pack(const __half* p, int stride) {   // p[0], p[stride]
  __half2 h = __halves2half2(p[0], p[stride]);
  return *reinterpret_cast<unsigned*>(&h);
}

// One warp per 16 x 8 output tile. A [M, K] (rows past M read as zero), W [N, K], C [M, N].
__global__ void replica(const __half* A, const __half* W, __half* C, int M, int N, int K, Candidate c) {
  const int lane = threadIdx.x, m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
  const int g4 = lane / 4, t4 = lane % 4, groups = K / 8, per_z = groups / c.nz, tiles_z = per_z / 8;
  const __half zero = __float2half(0.f);
  float partial[8][4];
  for (int z = 0; z < c.nz; ++z) {
    float chain[2][4] = {};
    for (int r = 0; r < per_z; ++r) {
      const int g = c.il ? ((r / 8) * c.nz + z) * 8 + r % 8 : z * per_z + r;   // il: 64-wide tiles
      const int s = c.ns == 1 ? 0 : (r / c.w) % 2, k = g * 8 + t4 * 2;
      (void)tiles_z;
      __half a[4];
      for (int i = 0; i < 2; ++i) {
        const int row = m0 + g4 + 8 * i;
        a[2 * i] = row < M ? A[(size_t)row * K + k] : zero;
        a[2 * i + 1] = row < M ? A[(size_t)row * K + k + 1] : zero;
      }
      const unsigned af[2] = {pack(a, 1), pack(a + 2, 1)};
      mma1688(chain[s], af, pack(W + (size_t)(n0 + g4) * K + k, 1));
    }
    for (int e = 0; e < 4; ++e) {
      partial[z][e] = c.ns == 1 ? chain[0][e] : chain[0][e] + chain[1][e];
      if (c.hp) partial[z][e] = __half2float(__float2half_rn(partial[z][e]));
    }
  }
  for (int e = 0; e < 4; ++e) {
    float total = partial[0][e];
    if (c.red == 1 && c.nz > 1) {
      float level[8];
      for (int z = 0; z < c.nz; ++z) level[z] = partial[z][e];
      for (int n = c.nz; n > 1; n /= 2)
        for (int z = 0; z < n / 2; ++z) level[z] = level[2 * z] + level[2 * z + 1];
      total = level[0];
    } else if (c.red == 2) {
      total = partial[c.nz - 1][e];
      for (int z = c.nz - 2; z >= 0; --z) total = total + partial[z][e];
    } else {
      for (int z = 1; z < c.nz; ++z) total = total + partial[z][e];
    }
    const int row = m0 + g4 + 8 * (e / 2), col = n0 + t4 * 2 + e % 2;
    if (row < M) C[(size_t)row * N + col] = __float2half_rn(total);
  }
}

void order_data(__half* A, __half* W, int M, int N, int K, int nz, uint32_t seed) {
  std::vector<__half> a((size_t)M * K, __float2half(0.f)), w((size_t)N * K, __float2half(0.f));
  for (int m = 0; m < M; ++m)
    for (int z = 0; z < nz; ++z) a[(size_t)m * K + z * (K / nz)] = __float2half(1.f);
  for (int n = 0; n < N; ++n)
    for (int z = 0; z < nz; ++z) {
      seed = seed * 1664525u + 1013904223u;
      w[(size_t)n * K + z * (K / nz)] = __float2half(((seed >> 31) ? -1.f : 1.f) * ldexpf(1.f, (int)((seed >> 8) % 40) - 24));
    }
  CK(cudaMemcpy(A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(W, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
  std::vector<int> shapes = {40, 1280, 1280, 40, 3840, 1280, 40, 5120, 1280, 40, 1280, 5120};
  if (argc > 3) { shapes.clear(); for (int i = 1; i + 2 < argc; i += 3) for (int j = 0; j < 3; ++j) shapes.push_back(atoi(argv[i + j])); }
  std::vector<Candidate> cands;
  for (int nz : {1, 2, 4, 8}) for (int ns : {1, 2}) for (int w : {1, 2, 4}) for (int red : {0, 1, 2})
    for (int hp : {0, 1}) for (int il : {0, 1}) {
      if ((ns == 1 && w > 1) || (nz == 1 && (red || hp || il)) || (ns == 2 && w != 4 && (hp || il))) continue;
      cands.push_back({nz, ns, w, red, hp, il});
    }
  cublasHandle_t h;
  CK(cublasCreate(&h));
  unsigned long long* dcount;
  CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  for (size_t s = 0; s < shapes.size(); s += 3) {
    const int M = shapes[s], N = shapes[s + 1], K = shapes[s + 2];
    __half *A, *W, *C, *R;
    CK(cudaMalloc(&A, 2ull * M * K)); CK(cudaMalloc(&W, 2ull * N * K));
    CK(cudaMalloc(&C, 2ull * M * N)); CK(cudaMalloc(&R, 2ull * M * N));
    printf("M %d N %d K %d:", M, N, K);
    for (const Candidate& c : cands) {
      unsigned long long bad = 0;
      for (int trial = 0; trial < 3; ++trial) {
        if (getenv("ORDER")) {
          order_data(A, W, M, N, K, atoi(getenv("ORDER")), 31u * trial + s);
        } else {
          fill<<<256, 256>>>(A, (size_t)M * K, 1000u * trial + s, -6, 3);   // activations ~ 2^[-6, 3)
          fill<<<256, 256>>>(W, (size_t)N * K, 7000u * trial + s, -12, -3); // weights ~ 2^[-12, -3)
        }
        const float alpha = 1.f, beta = 0.f;
        CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                        &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if ((K / 8) % c.nz || (c.il && (K / 8 / c.nz) % 8)) { bad = ~0ull; break; }
        replica<<<dim3(N / 8, (M + 15) / 16), 32>>>(A, W, R, M, N, K, c);
        CK(cudaMemset(dcount, 0, sizeof(*dcount)));
        count_diff<<<256, 256>>>(C, R, (size_t)M * N, dcount);
        unsigned long long d = 0;
        CK(cudaMemcpy(&d, dcount, sizeof(d), cudaMemcpyDeviceToHost));
        bad += d;
      }
      if (bad != ~0ull && (bad == 0 || !getenv("ONLY_ZERO")))
        printf(" [z%d s%d w%d r%d h%d i%d]=%llu", c.nz, c.ns, c.w, c.red, c.hp, c.il, bad);
    }
    printf("\n");
    cudaFree(A); cudaFree(W); cudaFree(C); cudaFree(R);
  }
  return 0;
}
