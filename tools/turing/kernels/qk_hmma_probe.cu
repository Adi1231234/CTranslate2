// Which arithmetic does cuBLAS use for the Whisper encoder's attention scores on this GPU (sm_80+ kernels)?
// C[b][j][i] = alpha * dot(Q[b][j][0:64], K[b][i][0:64]) as CTranslate2 calls it (Probe::cublas_run: strided
// batched, fp16, COMPUTE_32F, 1500 x 1500, alpha 1/8), compared bit for bit with candidates built from the
// tensor-core instruction mma.sync m16n8k16 (f32 accumulators from zero, 16-wide k groups in order):
//   0: out = half(alpha * acc)      1: out = half(acc * alpha)      2: chain with m16n8k8 steps, half(alpha * acc)
// usage: qk_hmma_probe [batch, default 160] -> mismatches per candidate over three fills
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

__device__ __forceinline__ unsigned pair_at(const __half* p, int rows, int r, int c) {
  return r < rows ? *reinterpret_cast<const unsigned*>(p + (size_t)r * 64 + c) : 0u;
}

// One warp per 16 queries x 8 keys of one batch entry.
__global__ void qk_candidate(const __half* Q, const __half* K, __half* C, int m, int n, float alpha, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, j0 = blockIdx.y * 16, i0 = blockIdx.x * 8;
  const __half* q = Q + (size_t)blockIdx.z * m * 64;
  const __half* k = K + (size_t)blockIdx.z * n * 64;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kk = 0; kk < 64; kk += 16) {
    const unsigned a0 = pair_at(q, m, j0 + g, kk + 2 * t), a1 = pair_at(q, m, j0 + g + 8, kk + 2 * t);
    const unsigned a2 = pair_at(q, m, j0 + g, kk + 8 + 2 * t), a3 = pair_at(q, m, j0 + g + 8, kk + 8 + 2 * t);
    const unsigned b0 = pair_at(k, n, i0 + g, kk + 2 * t), b1 = pair_at(k, n, i0 + g, kk + 8 + 2 * t);
    if (mode == 2) {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a2), "r"(a3), "r"(b1));
    } else {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  for (int e = 0; e < 4; ++e) {
    const int j = j0 + g + 8 * (e / 2), i = i0 + 2 * t + e % 2;
    if (j < m && i < n)
      C[((size_t)blockIdx.z * m + j) * n + i] = __float2half_rn(mode == 1 ? d[e] * alpha : alpha * d[e]);
  }
}

int main(int argc, char** argv) {
  const int batch = argc > 1 ? atoi(argv[1]) : 160, m = 1500, n = 1500;
  Probe p(batch, m, n, 64);
  __half* R; CK(cudaMalloc(&R, sizeof(__half) * batch * m * n));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long bad[3] = {0, 0, 0};
  for (int f = 0; f < 3; ++f) {
    fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * 64, 11u + f, -6 + f, 2 + f);
    fill<<<1024, 256>>>(p.dK, (size_t)batch * n * 64, 97u + f, -7 + f, 1 + f);
    p.cublas_run(0.125f);
    for (int mode = 0; mode < 3; ++mode) {
      qk_candidate<<<dim3((n + 7) / 8, (m + 15) / 16, batch), 32>>>(p.dQ, p.dK, R, m, n, 0.125f, mode);
      CK(cudaGetLastError());
      CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(p.dC, R, (size_t)batch * m * n, dc);
      unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad[mode] += d;
    }
  }
  for (int mode = 0; mode < 3; ++mode)
    printf("candidate %d: %llu of %llu mismatched\n", mode, bad[mode], 3ull * batch * m * n);
  return 0;
}
