// Does programmatic dependent launch (PDL, sm_90+) shorten the gaps between dependent kernels on this GPU and
// driver? Times a chain of dependent small kernels (each reads the previous one's output and a slice of a large
// buffer, ~the decoder's elementwise kernels) launched plainly and with PDL (cudaLaunchKernelEx with programmatic
// stream serialization; the kernel waits with griddepcontrol.wait before reading), then the same after a cuBLAS
// GEMM of the decoder's shape (40 x 1280 x 1280), whose kernel does not trigger its dependents early.
// usage: pdl_bench [chain length, default 2000]   (build for sm_120: -gencode arch=compute_120,code=sm_120)
#include <chrono>
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"

__global__ void step(const float* in, float* out, const __half* big, int n, int pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 0.5f + __half2float(big[(size_t)i * 8]);
}

// Holds the GPU (~ms) so that the host queues the whole chain first: the chain then runs at the GPU's own pace,
// as in production (the host is ~2 ms ahead there), and the times are the kernels plus the GPU-side gaps.
__global__ void spin(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {}
}

int main(int argc, char** argv) {
  const int chain = argc > 1 ? atoi(argv[1]) : 2000, n = 40 * 1280, blocks = (n + 255) / 256;
  float *a, *b; __half* big;
  CK(cudaMalloc(&a, 4ull * n)); CK(cudaMalloc(&b, 4ull * n)); CK(cudaMalloc(&big, 2ull * n * 8));
  CK(cudaMemset(a, 0, 4ull * n)); CK(cudaMemset(big, 0, 2ull * n * 8));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cublasHandle_t h; CK(cublasCreate(&h)); CK(cublasSetStream(h, s));
  __half *A, *W, *C;
  CK(cudaMalloc(&A, 2ull * 40 * 1280)); CK(cudaMalloc(&W, 2ull * 1280 * 1280)); CK(cudaMalloc(&C, 2ull * 40 * 1280));
  auto launch = [&](const float* in, float* out, int pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(256); cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, step, in, out, (const __half*)big, n, pdl));
  };
  auto gemm = [&]() {
    const float one = 1.f, zero = 0.f;
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 1280, 40, 1280, &one, W, CUDA_R_16F, 1280, A, CUDA_R_16F, 1280,
                    &zero, C, CUDA_R_16F, 1280, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto enqueue = [&](int pdl, int with_gemm) {
    for (int i = 0; i < chain; ++i) {
      if (with_gemm) gemm();
      launch(i % 2 ? b : a, i % 2 ? a : b, pdl);
    }
  };
  const char* names[6] = {"plain", "pdl", "gemm+plain", "gemm+pdl", "graph", "graph+gemm"};
  for (int mode = 0; mode < 6; ++mode) {        // 0-3 streams (plain, PDL, each after a GEMM), 4-5 one graph
    const int pdl = mode < 4 ? mode % 2 : 0, with_gemm = mode < 4 ? mode / 2 : mode - 4;
    cudaGraphExec_t exec = nullptr;
    if (mode >= 4) {
      cudaGraph_t graph;
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      enqueue(0, with_gemm);
      CK(cudaStreamEndCapture(s, &graph));
      CK(cudaGraphInstantiate(&exec, graph, 0));
    }
    for (int rep = 0; rep < 2; ++rep) {          // the first repetition warms up
      spin<<<1, 1, 0, s>>>(3000000000ll);        // ~1 s at 2.8 GHz: the host queues the chain meanwhile
      CK(cudaEventRecord(e0, s));
      if (exec) CK(cudaGraphLaunch(exec, s)); else enqueue(pdl, with_gemm);
      CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      if (rep) printf("%-14s %8.2f us per step\n", names[mode], 1000.f * ms / chain);
    }
  }
  // Host cost of a graph made fresh per decoding step (the cache length changes every step): capture, instantiate,
  // launch and destroy a chain of 285 GEMMs + 285 small kernels (~570, a Whisper decoder step), against plain
  // launches of the same.
  const int step_len = 285;
  for (int mode = 0; mode < 2; ++mode) {
    CK(cudaStreamSynchronize(s));
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 20; ++rep) {
      if (mode == 0) {
        for (int i = 0; i < step_len; ++i) { gemm(); launch(i % 2 ? b : a, i % 2 ? a : b, 0); }
        continue;
      }
      cudaGraph_t graph; cudaGraphExec_t exec;
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < step_len; ++i) { gemm(); launch(i % 2 ? b : a, i % 2 ? a : b, 0); }
      CK(cudaStreamEndCapture(s, &graph));
      CK(cudaGraphInstantiate(&exec, graph, 0));
      CK(cudaGraphLaunch(exec, s));
      CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    }
    const auto t1 = std::chrono::steady_clock::now();
    CK(cudaStreamSynchronize(s));
    const auto t2 = std::chrono::steady_clock::now();
    printf("%-14s host %8.1f us per step, until done %8.1f us per step\n", mode ? "fresh graph" : "plain step",
           std::chrono::duration<double, std::micro>(t1 - t0).count() / 20,
           std::chrono::duration<double, std::micro>(t2 - t0).count() / 20);
  }
  return 0;
}
