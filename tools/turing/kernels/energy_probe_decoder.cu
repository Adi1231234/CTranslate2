// energy_probe.cu's measurement for the decoding step's small kernels (the rest of a step's ~10.8 J): each part's
// self-attention products (cuBLAS strided batched, one query a row against t cached steps), the parts' softmax in
// one launch, the head split, the residual + LayerNorm, the feed-forward's BiasAdd GELU, and the vocabulary's
// log-softmax; 27 clips of 5 beams in 4 parts (40, 40, 35, 20 rows), 25 steps cached. Launches a step: 32 layers.
// usage: energy_probe_decoder
#include <cstdio>
#include <dlfcn.h>
#include <functional>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/bias_add_vec.cuh"
#include "cuda/softmax_parts.cu"
#include "cuda/split_heads.cu"
#include "cuda/residual_norm.cu"

namespace ctranslate2 {
  namespace cuda {
    cudaStream_t get_cuda_stream() { return 0; }              // the probe links no library
    bool use_stock_kernels() { return false; }
  }
}
using namespace ctranslate2::cuda;

static void* device;
static int (*nvml_energy)(void*, unsigned long long*);
static void nvml_open() {
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!lib) { fprintf(stderr, "no libnvidia-ml.so.1\n"); exit(1); }
  auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
  auto handle = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
  nvml_energy = reinterpret_cast<int (*)(void*, unsigned long long*)>(
    dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption"));
  CK(init()); CK(handle(0, &device));
}
static unsigned long long energy_mj() { unsigned long long e = 0; CK(nvml_energy(device, &e)); return e; }

static double total = 0;
static void measure(const char* name, double per_step, const std::function<void()>& run) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) run();
  CK(cudaEventRecord(a)); for (int i = 0; i < 10; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  const int n = std::max(20, int(1500.f / (ms / 10)));
  CK(cudaDeviceSynchronize());
  const unsigned long long e0 = energy_mj();
  CK(cudaEventRecord(a)); for (int i = 0; i < n; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  const unsigned long long e1 = energy_mj();
  CK(cudaEventElapsedTime(&ms, a, b)); CK(cudaGetLastError());
  const double mj = double(e1 - e0) / n, us = 1000.0 * ms / n;
  total += mj * per_step;
  printf("%-40s %6.1f/step %6.0f W %9.1f us %8.3f mJ/launch %8.1f mJ/step\n", name, per_step, mj / us * 1000.0, us,
         mj, mj * per_step);
  fflush(stdout);
}

int main() {
  nvml_open();
  cublasHandle_t h; CK(cublasCreate(&h));
  const int heads = 20, depth = 64, t = 25, rows = 135, parts_rows[4] = {40, 40, 35, 20};
  __half *Q, *K, *V, *S, *O, *X, *Bv, *R, *G, *L;
  CK(cudaMalloc(&Q, 2ull * rows * heads * depth)); CK(cudaMalloc(&K, 2ull * rows * heads * t * depth));
  CK(cudaMalloc(&V, 2ull * rows * heads * t * depth)); CK(cudaMalloc(&S, 2ull * rows * heads * t));
  CK(cudaMalloc(&O, 2ull * rows * heads * depth)); CK(cudaMalloc(&X, 2ull * rows * 51872));
  CK(cudaMalloc(&Bv, 2ull * 5120)); CK(cudaMalloc(&R, 2ull * rows * 5120)); CK(cudaMalloc(&G, 2ull * 1280));
  CK(cudaMalloc(&L, 2ull * rows * 51872));
  fill<<<256, 256>>>(Q, (size_t)rows * heads * depth, 1u, -6, 1);
  fill<<<256, 256>>>(K, (size_t)rows * heads * t * depth, 2u, -6, 1);
  fill<<<256, 256>>>(V, (size_t)rows * heads * t * depth, 3u, -6, 1);
  fill<<<256, 256>>>(X, (size_t)rows * 51872, 4u, -6, 3);
  fill<<<16, 256>>>(Bv, (size_t)5120, 5u, -8, -2); fill<<<64, 256>>>(R, (size_t)rows * 5120, 6u, -6, 1);
  fill<<<16, 256>>>(G, (size_t)1280, 7u, -1, 1);
  const float alpha = 0.125f, one = 1.f, zero = 0.f;
  // Each part: scores = alpha q K^T (batch rows x heads, 1 x t), then output = P V (1 x 64).
  auto per_part = [&](bool scores) {
    int row = 0;
    for (int p = 0; p < 4; ++p) {
      const int batch = parts_rows[p] * heads;
      const __half* q = Q + (size_t)row * heads * depth;
      const __half* k = K + (size_t)row * heads * t * depth;
      if (scores)
        CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, depth, &alpha, k, CUDA_R_16F, depth,
                                      (long long)t * depth, q, CUDA_R_16F, depth, depth, &zero,
                                      S + (size_t)row * heads * t, CUDA_R_16F, t, t, batch, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));
      else
        CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, depth, 1, t, &one,
                                      V + (size_t)row * heads * t * depth, CUDA_R_16F, depth, (long long)t * depth,
                                      S + (size_t)row * heads * t, CUDA_R_16F, t, t, &zero,
                                      O + (size_t)row * heads * depth, CUDA_R_16F, depth, depth, batch,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      row += parts_rows[p];
    }
  };
  measure("self-attention scores, 4 parts", 32, [&] { per_part(true); });
  measure("self-attention output, 4 parts", 32, [&] { per_part(false); });
  SoftmaxParts sp; unsigned end = 0;
  for (int p = 0, row = 0; p < 4; row += parts_rows[p++]) {
    end += parts_rows[p] * heads;
    sp.data[sp.count] = S + (size_t)row * heads * t; sp.rows_end[sp.count] = end; sp.cols[sp.count++] = t;
  }
  measure("softmax_parts 4 parts", 32, [&] { softmax_parts(sp); });
  float16_t* outs[3] = {reinterpret_cast<float16_t*>(Q), reinterpret_cast<float16_t*>(K),
                        reinterpret_cast<float16_t*>(V)};
  measure("split_heads_bias qkv 135 rows", 32, [&] {
    split_heads_bias(reinterpret_cast<const float16_t*>(X), reinterpret_cast<const float16_t*>(Bv), outs, 3, rows,
                     1, heads, depth); });
  measure("residual_norm 135 x 1280", 96, [&] {
    residual_norm(reinterpret_cast<const float16_t*>(X), reinterpret_cast<const float16_t*>(Bv),
                  reinterpret_cast<const float16_t*>(R), reinterpret_cast<const float16_t*>(G),
                  reinterpret_cast<const float16_t*>(G), 1e-5f, reinterpret_cast<float16_t*>(X),
                  reinterpret_cast<float16_t*>(L), rows, 1280); });
  measure("ffn1 BiasAdd GELU 135 x 5120", 32, [&] {
    const size_t vecs = (size_t)rows * 5120 / 8;
    ctranslate2::ops::bias_add_vec_kernel<ctranslate2::ops::BiasAddVecMode::gelu><<<(vecs + 255) / 256, 256>>>(
      reinterpret_cast<const uint4*>(R), reinterpret_cast<const uint4*>(Bv), nullptr, reinterpret_cast<uint4*>(R),
      5120 / 8, vecs); });
  measure("vocabulary log-softmax 135 x 51866", 1, [&] {
    at::native::softmax_rows<__half, at::native::LogSoftMaxForwardEpilogue>(0, X, L, rows, 51866, nullptr, true); });
  printf("sum of the kernels above: %.1f mJ a step\n", total);
  return 0;
}
