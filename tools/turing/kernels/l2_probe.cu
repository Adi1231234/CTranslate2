// Do L2 prefetches let the decoder's 1280 x 1280 Dense layer (cuBLAS, called as CTranslate2 calls it) read its
// weights from L2? Times that GEMM alone with its weights cold (a rotation of copies far larger than L2), warm
// (read just before), and after a prefetch issued ~20 us earlier (where the self-attention's small kernels run):
// prefetch.global.L2 per 128 or per 32 bytes, with evict_last, and cp.async.bulk.prefetch.L2 (sm_90+). Then the
// same with MB of other traffic in between (the beam reorder's cache copy), and the prefetch kernels' own time.
// usage: l2_probe [rows, default 40] [traffic MB, default 20]
// build: build_probe.ps1 l2_probe -Gencode '-gencode=arch=compute_120,code=sm_120'
#include <cstdio>
#include "probe_common.h"

constexpr int K = 1280, N = 1280, copies = 48;             // 48 x 3.3 MB = 157 MB, far past the 32 MB of L2
constexpr size_t wbytes = size_t(N) * K * sizeof(__half);

__global__ void spin_ns(long long ns) {                     // holds the GPU while the host queues the rest
  long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

__global__ void fill(__half* p, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2half(float(int(i % 97) - 48) / 64.f);
}

__global__ void prefetch_lines(const char* p, size_t bytes, int stride, int last) {
  for (size_t o = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * stride; o < bytes;
       o += size_t(gridDim.x) * blockDim.x * stride) {
    if (last)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + o));
    else
      asm volatile("prefetch.global.L2 [%0];" :: "l"(p + o));
  }
}

__global__ void prefetch_bulk(const char* p, size_t bytes) {
#if __CUDA_ARCH__ >= 900
  constexpr size_t chunk = size_t(1) << 16;
  for (size_t o = size_t(threadIdx.x) * chunk; o < bytes; o += size_t(blockDim.x) * chunk) {
    const unsigned n = unsigned(bytes - o < chunk ? bytes - o : chunk);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(n) : "memory");
  }
#endif
}

__global__ void traffic(const uint4* src, uint4* dst, size_t n) {   // a plain copy, as the cache reorder does
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

int main(int argc, char** argv) {
  const int m = argc > 1 ? atoi(argv[1]) : 40;
  const size_t traffic_mb = argc > 2 ? atoi(argv[2]) : 20, tn = (traffic_mb << 20) / sizeof(uint4);
  char* W; __half *A, *C; uint4 *ts, *td;
  CK(cudaMalloc(&W, wbytes * copies)); CK(cudaMalloc(&A, 2ull * m * K)); CK(cudaMalloc(&C, 2ull * m * N));
  CK(cudaMalloc(&ts, tn * sizeof(uint4))); CK(cudaMalloc(&td, tn * sizeof(uint4)));
  fill<<<512, 256>>>(reinterpret_cast<__half*>(W), wbytes * copies / 2);
  fill<<<64, 256>>>(A, size_t(m) * K);
  CK(cudaMemset(ts, 0, tn * sizeof(uint4)));
  cublasHandle_t h; CK(cublasCreate(&h));
  cudaEvent_t e0, e1, p0, p1;
  for (cudaEvent_t* e : {&e0, &e1, &p0, &p1}) CK(cudaEventCreate(e));
  const float one = 1.f, zero = 0.f;
  auto gemm = [&](int c) {
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, m, K, &one, W + c * wbytes, CUDA_R_16F, K, A, CUDA_R_16F, K,
                    &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  auto prefetch = [&](int mode, int c) {
    const char* p = W + c * wbytes;
    if (mode == 1) gemm(c);
    if (mode == 2) prefetch_lines<<<100, 256>>>(p, wbytes, 128, 0);
    if (mode == 3) prefetch_lines<<<400, 256>>>(p, wbytes, 32, 0);
    if (mode == 4) prefetch_lines<<<100, 256>>>(p, wbytes, 128, 1);
    if (mode == 5) prefetch_bulk<<<1, 64>>>(p, wbytes);
  };
  const char* names[] = {"cold", "warm", "pf 128 B", "pf 32 B", "pf 128 B evict_last", "bulk prefetch"};
  const int reps = 200, skip = 10;
  for (int with_traffic = 0; with_traffic < 2; ++with_traffic)
    for (int mode = 0; mode < 6; ++mode) {
      double gemm_us = 0, pre_us = 0;
      for (int r = 0; r < reps + skip; ++r) {
        const int c = r % copies;
        spin_ns<<<1, 1>>>(20000);
        CK(cudaEventRecord(p0)); prefetch(mode, c); CK(cudaEventRecord(p1));
        if (with_traffic) traffic<<<288, 256>>>(ts, td, tn);
        spin_ns<<<1, 1>>>(20000);
        CK(cudaEventRecord(e0)); gemm(c); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float g, p; CK(cudaEventElapsedTime(&g, e0, e1)); CK(cudaEventElapsedTime(&p, p0, p1));
        if (r >= skip) { gemm_us += g * 1e3; pre_us += p * 1e3; }
      }
      printf("rows %2d traffic %3zu MB  %-20s GEMM %6.2f us   prefetch step %6.2f us\n", m,
             with_traffic ? traffic_mb : 0, names[mode], gemm_us / reps, pre_us / reps);
    }
  CK(cudaDeviceSynchronize());
  return 0;
}
