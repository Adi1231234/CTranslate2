// Where does the stream's power go? The batched path runs at the L40S's 350 W limit (the SM clock ~1900 of
// 2520 MHz), so a kernel's cost is its energy, not its time. Each heavy kernel of a decoding step and of the
// encoder, at production shapes (27 clips of 5 beams = 135 decoder rows; encoder batches of 8 clips), runs alone
// in a loop for ~1.5 s with NVML's energy counter read around it: watts, microseconds and millijoules a launch,
// and millijoules a decoding step (launches a step from the round23 profile: encoder work is ~2.5 layer-batches
// a step). NVML comes from the driver at run time (dlopen; the build machine has no GPU driver).
// usage: energy_probe
#include <cstdio>
#include <dlfcn.h>
#include <functional>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/tiled_split_gemm.cuh"
#include "ops/cross_attention.cuh"
#include "ops/exact_attention_launch.cuh"
#include "ops/bias_add_vec.cuh"
#include "cuda/cache_reorder.cu"

namespace ctranslate2 {
  namespace cuda {
    cudaStream_t get_cuda_stream() { return 0; }              // the probe links no library
  }
}

// The three NVML calls the probe makes (nvml.h's signatures; a device handle is an opaque pointer).
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
static unsigned long long energy_mj() {
  unsigned long long e = 0; CK(nvml_energy(device, &e)); return e;
}

struct Row { const char* name; double per_step, watts, us, mj; };
static std::vector<Row> rows;

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
  rows.push_back({name, per_step, mj / us * 1000.0, us, mj});
  printf("%-34s %6.1f/step %6.0f W %9.1f us %8.3f mJ/launch %8.1f mJ/step\n", name, per_step, mj / us * 1000.0, us,
         mj, mj * per_step);
  fflush(stdout);
}

int main() {
  nvml_open();
  cublasHandle_t h; CK(cublasCreate(&h));
  const int dm = 135, em = 12000, heads = 20, clips = 27, beams = 5;
  __half *A, *W, *C, *B; CK(cudaMalloc(&A, 2ull * em * 5120)); CK(cudaMalloc(&W, 2ull * 51872 * 1280));
  CK(cudaMalloc(&C, 2ull * em * 5120)); CK(cudaMalloc(&B, 2ull * 5120));
  fill<<<1024, 256>>>(A, (size_t)em * 5120, 1u, -6, 1);
  fill<<<1024, 256>>>(W, (size_t)51872 * 1280, 2u, -12, -4);
  fill<<<16, 256>>>(B, (size_t)5120, 3u, -8, -2);
  auto gemm = [&](int m, int n, int k) {
    const float one = 1.f, zero = 0.f;
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, W, CUDA_R_16F, k, A, CUDA_R_16F, k, &zero, C,
                    CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  // The decoder's products at 135 rows (cuBLAS; the second feed-forward in the grouped kernel).
  measure("decoder qkv 3840x1280", 32, [&] { gemm(dm, 3840, 1280); });
  measure("decoder o/cross-q/cross-o 1280x1280", 96, [&] { gemm(dm, 1280, 1280); });
  measure("decoder ffn1 5120x1280", 32, [&] { gemm(dm, 5120, 1280); });
  measure("decoder vocab 51866x1280", 1, [&] { gemm(dm, 51866, 1280); });
  SplitGroups groups{}; int end = 0;
  for (int r : {40, 40, 35, 20}) {
    int slice = 0, slices = 0; gsg_split_of(r, slice, slices); end += r;
    groups.row_end[groups.count] = end; groups.slice[groups.count] = slice; groups.slices[groups.count++] = slices;
  }
  measure("decoder ffn2 grouped 64x64k32/3", 32, [&] {
    tsg_launch<64, 64, 2, 3>(A, W, C, dm, 1280, 5120, groups, 0); });
  measure("decoder ffn2 grouped 64x64k64/5", 32, [&] {
    tsg_launch<64, 64, 2, 5, 64>(A, W, C, dm, 1280, 5120, groups, 0); });
  // The cross-attention of 27 clips x 5 beams against 1500 positions.
  __half *Kc, *Vc, *Q, *O; const size_t kv = (size_t)clips * heads * 1500 * 64;
  CK(cudaMalloc(&Kc, 2 * kv)); CK(cudaMalloc(&Vc, 2 * kv)); CK(cudaMalloc(&Q, 2ull * clips * heads * beams * 64));
  CK(cudaMalloc(&O, 2ull * clips * heads * beams * 64));
  fill<<<1024, 256>>>(Kc, kv, 4u, -6, 1); fill<<<1024, 256>>>(Vc, kv, 5u, -6, 1);
  fill<<<256, 256>>>(Q, (size_t)clips * heads * beams * 64, 6u, -6, 1);
  const int smem = beams * at::native::ca_pitch * 2;
  measure("cross_attention 27 clips", 32, [&] {
    at::native::cross_attention_kernel<<<clips * heads, at::native::ca_warps * 32, smem>>>(
      at::native::CaQueries{Q, nullptr, nullptr, nullptr, 0}, Kc, Vc, O, heads, beams, beams, 28, 0.125f, 4, nullptr,
      ctranslate2::cuda::CrossResidues(), nullptr); });
  // The self-attention caches: 4 parts of ~34 rows with 25 steps cached.
  ctranslate2::cuda::CacheParts parts;
  std::vector<int32_t> ord(34);
  for (int r = 0; r < 34; ++r) ord[r] = (r * 7) % 34;
  int32_t* order; CK(cudaMalloc(&order, 4 * 34)); CK(cudaMemcpy(order, ord.data(), 4 * 34, cudaMemcpyHostToDevice));
  for (int p = 0; p < 4; ++p) {
    for (int c = 0; c < 2; ++c) {
      parts.cache[p][c] = Kc + (size_t)(2 * p + c) * 34 * heads * 25 * 64;
      parts.fresh[p][c] = Q;
      parts.out[p][c] = Vc + (size_t)(2 * p + c) * 34 * heads * 26 * 64;
    }
    parts.order[p] = order; parts.rows[p] = 34; parts.time[p] = 25; ++parts.count;
  }
  measure("reorder_append_parts 4 x 34 rows t25", 32, [&] {
    ctranslate2::cuda::reorder_append_parts(parts, heads, 64); });
  // The encoder, a batch of 8 clips (12000 rows): its products, the GELU pass, the attention.
  measure("encoder qkv 3840x1280", 2.5, [&] { gemm(em, 3840, 1280); });
  measure("encoder o 1280x1280", 2.5, [&] { gemm(em, 1280, 1280); });
  measure("encoder fc1 5120x1280", 2.5, [&] { gemm(em, 5120, 1280); });
  measure("encoder fc1 BiasAdd GELU", 2.5, [&] {
    const size_t vecs = (size_t)em * 5120 / 8;
    ctranslate2::ops::bias_add_vec_kernel<ctranslate2::ops::BiasAddVecMode::gelu><<<(vecs + 255) / 256, 256>>>(
      reinterpret_cast<const uint4*>(C), reinterpret_cast<const uint4*>(B), nullptr, reinterpret_cast<uint4*>(C),
      5120 / 8, vecs); });
  measure("encoder fc2 1280x5120", 2.5, [&] { gemm(em, 1280, 5120); });
  measure("decoder memory K/V 2560x1280", 2.5, [&] { gemm(em, 2560, 1280); });
  void* ws; CK(cudaMalloc(&ws, at::native::exact_attention_workspace(160, 1500, true)));
  measure("encoder attention 8 clips", 2.5, [&] {
    at::native::exact_attention(Kc, Kc + 160ull * 1500 * 64, Vc, ws, C, 160, heads, 1500, 1500, 0.125f, 0); });
  double total = 0; for (const Row& r : rows) total += r.mj * r.per_step;
  printf("sum of the kernels above: %.1f mJ a step (at the measured ~31.6 steps a second the GPU draws ~340 W)\n",
         total);
  return 0;
}
