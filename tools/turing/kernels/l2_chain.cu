// Can a side stream keep the decoder's memory busy? One decoding step stood in for: 32 layers of the Dense layers
// (cuBLAS, as CTranslate2 calls them, `rows` rows), a copy of the self-attention cache (the beam reorder), a 61 MB
// read (the cross-attention of 8 clips) and short spins (the small kernels), on one stream as in production. Then
// with a second stream that, as each Dense layer starts (an event), bulk-prefetches into L2 the weights of the
// Dense layer `lead` places later; and with only the out and cross-query weights prefetched, once the
// self-attention's projection is done. Prints ms per step.
// usage: l2_chain [rows, default 40] [cache MB per layer, default 6.5] [steps, default 20]
// build: build_probe.ps1 l2_chain -Gencode '-gencode=arch=compute_120,code=sm_120'
#include <cstdio>
#include "probe_common.h"

constexpr int layers = 32, denses = 6;
constexpr int dense_n[denses] = {3840, 1280, 1280, 1280, 5120, 1280};   // qkv, out, cross q, cross out, ffn1, ffn2
constexpr int dense_k[denses] = {1280, 1280, 1280, 1280, 1280, 5120};
constexpr size_t cross_bytes = size_t(61) << 20;

__global__ void spin_ns(long long ns) {
  long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
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

__global__ void copy16(const uint4* src, uint4* dst, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

__global__ void read16(const uint4* src, size_t n, uint4* sink) {          // a streaming read, as the cross kernel
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint4 v = src[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0x12345678u) *sink = acc;
}

int main(int argc, char** argv) {
  const int m = argc > 1 ? atoi(argv[1]) : 40, steps = argc > 3 ? atoi(argv[3]) : 20;
  const size_t cache = size_t((argc > 2 ? atof(argv[2]) : 6.5) * (1 << 20)) & ~size_t(15);
  std::vector<__half*> W(layers * denses);
  for (int i = 0; i < layers * denses; ++i) {
    const size_t bytes = size_t(dense_n[i % denses]) * dense_k[i % denses] * 2;
    CK(cudaMalloc(&W[i], bytes)); CK(cudaMemset(W[i], 0x21, bytes));
  }
  char *cross, *cache_a, *cache_b; __half *x, *y; uint4* sink;
  CK(cudaMalloc(&cross, cross_bytes * layers)); CK(cudaMemset(cross, 1, cross_bytes * layers));
  CK(cudaMalloc(&cache_a, cache * layers)); CK(cudaMalloc(&cache_b, cache * layers));
  CK(cudaMemset(cache_a, 2, cache * layers));
  CK(cudaMalloc(&x, 2ull * m * 5120)); CK(cudaMalloc(&y, 2ull * m * 5120)); CK(cudaMalloc(&sink, 16));
  CK(cudaMemset(x, 0, 2ull * m * 5120));
  cudaStream_t s, side; CK(cudaStreamCreate(&s)); CK(cudaStreamCreate(&side));
  cublasHandle_t h; CK(cublasCreate(&h)); CK(cublasSetStream(h, s));
  std::vector<cudaEvent_t> ev(layers * denses);
  for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  const float one = 1.f, zero = 0.f;
  auto prefetch = [&](int i) {
    if (i >= layers * denses) return;
    prefetch_bulk<<<1, 128, 0, side>>>(reinterpret_cast<const char*>(W[i]),
                                         size_t(dense_n[i % denses]) * dense_k[i % denses] * 2);
  };
  auto dense = [&](int i, int lead) {                            // lead > 0: prefetch Dense i + lead as i starts
    if (lead > 0) {
      CK(cudaEventRecord(ev[i], s)); CK(cudaStreamWaitEvent(side, ev[i]));
      prefetch(i + lead);
    }
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, dense_n[i % denses], m, dense_k[i % denses], &one, W[i],
                    CUDA_R_16F, dense_k[i % denses], x, CUDA_R_16F, dense_k[i % denses], &zero, y, CUDA_R_16F,
                    dense_n[i % denses], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  // mode 0: one stream; 1, 2: prefetch `mode` Dense layers ahead as each starts; 3: out + cross q after qkv
  auto step = [&](int mode) {
    const int lead = mode == 3 ? 0 : mode;
    for (int l = 0; l < layers; ++l) {
      const int b = l * denses;
      spin_ns<<<1, 1, 0, s>>>(3000);
      dense(b, lead);
      spin_ns<<<1, 1, 0, s>>>(1200);
      if (mode == 3) {
        CK(cudaEventRecord(ev[b + 1], s)); CK(cudaStreamWaitEvent(side, ev[b + 1]));
        prefetch(b + 1); prefetch(b + 2);
      }
      copy16<<<288, 256, 0, s>>>(reinterpret_cast<const uint4*>(cache_a + l * cache),
                                 reinterpret_cast<uint4*>(cache_b + l * cache), cache / 16);
      spin_ns<<<1, 1, 0, s>>>(9000);
      dense(b + 1, lead);
      spin_ns<<<1, 1, 0, s>>>(3000);
      dense(b + 2, lead);
      spin_ns<<<1, 1, 0, s>>>(1200);
      read16<<<288, 256, 0, s>>>(reinterpret_cast<const uint4*>(cross + l * cross_bytes), cross_bytes / 16, sink);
      dense(b + 3, lead);
      spin_ns<<<1, 1, 0, s>>>(3000);
      dense(b + 4, lead);
      spin_ns<<<1, 1, 0, s>>>(1700);
      dense(b + 5, lead);
    }
  };
  const char* names[4] = {"one stream", "prefetch 1 ahead", "prefetch 2 ahead", "out + cross q after qkv"};
  for (int round = 0; round < 2; ++round)
    for (int mode = 0; mode < 4; ++mode) {
      step(mode);                                                // warm-up step
      CK(cudaEventRecord(t0, s));
      for (int i = 0; i < steps; ++i) step(mode);
      CK(cudaEventRecord(t1, s)); CK(cudaEventSynchronize(t1)); CK(cudaDeviceSynchronize());
      float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
      printf("rows %d cache %.1f MB  %-26s %7.3f ms per step\n", m, cache / 1048576.0, names[mode], ms / steps);
    }
  return 0;
}
