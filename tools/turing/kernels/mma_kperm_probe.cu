// Does mma.sync m16n8k16 (fp16 inputs, fp32 accumulator) give the same bits when the 16 k positions are permuted
// the same way in A and B? If it does, a kernel may feed a k group's values to the instruction in any order (e.g.
// so that a lane's operands are consecutive in memory: wider loads, fewer instructions), keeping cuBLAS's chain.
// Random 16 x 16 A, 16 x 8 B and accumulators over wide exponent ranges (every alignment and rounding case the
// sums meet), several permutations; counts outputs that differ from the identity order.
// usage: mma_kperm_probe [fills, default 4000]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"

__constant__ int perm[8][16];

// One warp per tile: d = A B + c with A [16][16], B [16][8] (k-major columns), positions permuted by perm[p].
__global__ void kperm(const __half* A, const __half* B, const float* C, float* D, int p) {
  const int lane = threadIdx.x, g = lane / 4, t = lane % 4, tile = blockIdx.x;
  const __half* a = A + tile * 256;
  const __half* b = B + tile * 128;
  auto at = [&](const __half* m, int row, int k, int ld) { return m[row * ld + perm[p][k]]; };
  auto pack = [](__half lo, __half hi) {
    return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
  };
  const unsigned a0 = pack(at(a, g, 2 * t, 16), at(a, g, 2 * t + 1, 16));
  const unsigned a1 = pack(at(a, g + 8, 2 * t, 16), at(a, g + 8, 2 * t + 1, 16));
  const unsigned a2 = pack(at(a, g, 2 * t + 8, 16), at(a, g, 2 * t + 9, 16));
  const unsigned a3 = pack(at(a, g + 8, 2 * t + 8, 16), at(a, g + 8, 2 * t + 9, 16));
  const unsigned b0 = pack(at(b, g, 2 * t, 16), at(b, g, 2 * t + 1, 16));      // B stored [n][k]
  const unsigned b1 = pack(at(b, g, 2 * t + 8, 16), at(b, g, 2 * t + 9, 16));
  float d[4];
  for (int e = 0; e < 4; ++e) d[e] = C[tile * 128 + (g + 8 * (e / 2)) * 8 + 2 * t + e % 2];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  for (int e = 0; e < 4; ++e) D[tile * 128 + (g + 8 * (e / 2)) * 8 + 2 * t + e % 2] = d[e];
}

int main(int argc, char** argv) {
  const int fills = argc > 1 ? atoi(argv[1]) : 4000, tiles = 4096;
  int h[8][16];
  for (int k = 0; k < 16; ++k) {
    h[0][k] = k;                                             // identity
    h[1][k] = 15 - k;                                        // reversed
    h[2][k] = (k % 4) * 4 + k / 4;                           // transposed 4 x 4
    h[3][k] = k < 8 ? 4 * (k / 2) + k % 2 : 4 * ((k - 8) / 2) + 2 + k % 2;   // lane t: dims 4t .. 4t + 3
    h[4][k] = (k + 1) % 16;                                  // rotated by one
    h[5][k] = k ^ 8;                                         // halves swapped
    h[6][k] = k ^ 1;                                         // pairs swapped
    h[7][k] = (k * 5) % 16;                                  // stride 5
  }
  CK(cudaMemcpyToSymbol(perm, h, sizeof h));
  __half *A, *B; float *C, *D0, *D;
  CK(cudaMalloc(&A, 2ull * tiles * 256)); CK(cudaMalloc(&B, 2ull * tiles * 128));
  CK(cudaMalloc(&C, 4ull * tiles * 128)); CK(cudaMalloc(&D0, 4ull * tiles * 128)); CK(cudaMalloc(&D, 4ull * tiles * 128));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long bad[8] = {}, outputs = 0;
  for (int f = 0; f < fills; ++f) {
    const int lo = -14 + f % 12, span = 2 + f % 18;
    fill<<<256, 256>>>(A, (size_t)tiles * 256, 11u * f + 1, lo, lo + span);
    fill<<<256, 256>>>(B, (size_t)tiles * 128, 13u * f + 2, lo, lo + span);
    fill<<<256, 256>>>(C, (size_t)tiles * 128, 17u * f + 3, 2 * lo, 2 * lo + span);
    if (f % 3 == 0) CK(cudaMemset(C, 0, 4ull * tiles * 128));
    kperm<<<tiles, 32>>>(A, B, C, D0, 0);
    for (int p = 1; p < 8; ++p) {
      kperm<<<tiles, 32>>>(A, B, C, D, p);
      CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(D0, D, (size_t)tiles * 128, dc);
      unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); bad[p] += x;
    }
    outputs += (size_t)tiles * 128;
  }
  const char* names[8] = {"identity", "reversed", "transposed", "lane-consecutive", "rotated", "halves swapped",
                          "pairs swapped", "stride 5"};
  for (int p = 1; p < 8; ++p) printf("%-17s %llu of %llu outputs differ\n", names[p], bad[p], outputs);
  return 0;
}
