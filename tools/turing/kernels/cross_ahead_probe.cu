// The cross-attention kernel (src/ops/cross_attention.cuh) moves ~225 MB from DRAM a launch where its keys and values
// are 192 MB (round27 ncu, 25 clips): is the L2 prefetch distance (`ahead`, CT2_CROSS_AHEAD, default 4) behind the
// extra? 27 clips x 5 beams, each distance alone for ~1.5 s: time, watts and millijoules a launch (NVML's energy
// counter). The outputs of every distance are compared with distance 4's (prefetching changes no value).
// usage: cross_ahead_probe
#include <cstdio>
#include <dlfcn.h>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/cross_attention.cuh"

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

int main() {
  nvml_open();
  const int clips = 27, heads = 20, beams = 5;
  const size_t kv = (size_t)clips * heads * 1500 * 64;
  __half *K, *V, *Q, *O, *O4; unsigned long long* dc;
  CK(cudaMalloc(&K, 2 * kv)); CK(cudaMalloc(&V, 2 * kv)); CK(cudaMalloc(&Q, 2ull * clips * heads * beams * 64));
  CK(cudaMalloc(&O, 2ull * clips * heads * beams * 64)); CK(cudaMalloc(&O4, 2ull * clips * heads * beams * 64));
  CK(cudaMalloc(&dc, 8));
  fill<<<1024, 256>>>(K, kv, 4u, -6, 1); fill<<<1024, 256>>>(V, kv, 5u, -6, 1);
  fill<<<256, 256>>>(Q, (size_t)clips * heads * beams * 64, 6u, -6, 1);
  const int smem = beams * at::native::ca_pitch * 2;
  auto launch = [&](int ahead, __half* out) {
    at::native::cross_attention_kernel<<<clips * heads, at::native::ca_warps * 32, smem>>>(
      at::native::CaQueries{Q, nullptr, nullptr, nullptr, 0}, K, V, out, heads, beams, beams, 28, 0.125f, ahead,
      nullptr, ctranslate2::cuda::CrossResidues(), nullptr);
  };
  launch(4, O4);
  for (int ahead : {0, 1, 2, 4, 8, 16}) {
    launch(ahead, O);
    CK(cudaMemset(dc, 0, 8)); count_diff<<<64, 256>>>(O4, O, (size_t)clips * heads * beams * 64, dc);
    unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch(ahead, O);
    const int n = 5000;
    CK(cudaDeviceSynchronize());
    const unsigned long long e0 = energy_mj();
    CK(cudaEventRecord(a)); for (int i = 0; i < n; ++i) launch(ahead, O); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    const unsigned long long e1 = energy_mj();
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    const double us = 1000.0 * ms / n, mj = double(e1 - e0) / n;
    printf("ahead %2d: %7.1f us %5.0f W %7.2f mJ a launch, %.0f GB/s of keys and values; %llu outputs differ\n",
           ahead, us, mj / us * 1000.0, mj, 2.0 * kv * 2 / (us * 1e3), d);
    fflush(stdout);
  }
  return 0;
}
