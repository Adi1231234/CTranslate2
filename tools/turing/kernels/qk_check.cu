// Bit-for-bit check and timing of src/cuda/attention_scores_k64.cuh against the cuBLAS call that
// CTranslate2 makes for Whisper cross-attention scores, over every query count 1..8 and a sweep of
// batch sizes, on random data with a wide dynamic range and on signed zeros.
// usage: qk_check      -> per-(m, batch) mismatch counts (must all be 0) and the timings
#include "probe_common.h"
#include "../../../src/cuda/attention_scores_k64.cuh"

using namespace ctranslate2::cuda;

// Halves with random sign and magnitude 2^[lo, hi): exercises rounding in every partial sum.
__global__ void fill(__half* x, size_t count, uint32_t seed, int lo, int hi) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < count; t += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)t * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const float mant = 1.f + (h & 1023) / 1024.f;
    const int e = lo + (int)((h >> 10) % (uint32_t)(hi - lo));
    x[t] = __float2half(((h >> 31) ? -1.f : 1.f) * ldexpf(mant, e));
  }
}

__global__ void set_bits(__half* x, size_t count, unsigned short bits) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < count; t += (size_t)gridDim.x * blockDim.x)
    x[t] = __ushort_as_half(bits);
}

__global__ void count_diff(const __half* a, const __half* b, size_t count, unsigned long long* out) {
  unsigned long long local = 0;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < count; t += (size_t)gridDim.x * blockDim.x)
    local += __half_as_ushort(a[t]) != __half_as_ushort(b[t]);
  if (local) atomicAdd(out, local);
}

int main() {
  const int max_batch = 1024, max_m = 8, n = 1500, k = 64;
  Probe p(max_batch, max_m, n, k);
  __half* dC2;
  unsigned long long* dcount;
  CK(cudaMalloc(&dC2, sizeof(__half) * max_batch * max_m * n));
  CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  int version = 0;
  CK(cublasGetVersion(p.h, &version));
  printf("cuBLAS %d\n", version);
  auto check = [&](int batch, int m, float alpha) {
    p.m = m;
    p.cublas_run(alpha, batch);
    attention_scores_k64(p.dQ, p.dK, dC2, batch, m, n, alpha, 0);
    CK(cudaMemset(dcount, 0, sizeof(unsigned long long)));
    count_diff<<<256, 256>>>(p.dC, dC2, (size_t)batch * m * n, dcount);
    unsigned long long diff = 0;
    CK(cudaMemcpy(&diff, dcount, sizeof(diff), cudaMemcpyDeviceToHost));
    return diff;
  };
  // Signed zeros: every product -0.
  set_bits<<<64, 256>>>(p.dK, (size_t)n * k, 0x8000);
  set_bits<<<64, 256>>>(p.dQ, (size_t)max_m * k, 0x3c00);          // 1.0
  printf("signed zeros: %llu mismatches\n", check(1, 5, 0.125f));
  std::vector<int> batches;
  for (int b = 1; b <= 256; ++b) batches.push_back(b);
  for (int b = 320; b <= max_batch; b += 80) batches.push_back(b);
  batches.push_back(max_batch);
  unsigned long long total = 0, outputs = 0;
  for (int m = 1; m <= max_m; ++m) {
    unsigned long long m_total = 0;
    for (int batch : batches) {
      const uint32_t seed = (uint32_t)(m * 100003 + batch);
      const int lo = -10 + (batch % 5), hi = 1 + (batch % 4);    // |q|, |k| in [2^-10, 16)
      fill<<<1024, 256>>>(p.dK, (size_t)batch * n * k, seed, lo, hi);
      fill<<<64, 256>>>(p.dQ, (size_t)batch * m * k, seed ^ 0x9e3779b9u, lo, hi);
      const unsigned long long d = check(batch, m, 0.125f);
      if (d) printf("  m=%d batch=%d: %llu mismatches\n", m, batch, d);
      m_total += d; outputs += (unsigned long long)batch * m * n;
    }
    printf("m=%d: %llu mismatches\n", m, m_total);
    total += m_total;
  }
  printf("TOTAL %llu mismatches over %llu outputs\n", total, outputs);
  // Timing at the production shape: 8 clips x 20 heads, 5 beams.
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int batch = 160, m = 5, iters = 200;
  p.m = m;
  for (int which = 0; which < 2; ++which) {
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it) {
      if (which == 0) p.cublas_run(0.125f, batch);
      else attention_scores_k64(p.dQ, p.dK, dC2, batch, m, n, 0.125f, 0);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("%s: %.1f us per call (batch %d, m %d, n %d)\n", which ? "attention_scores_k64" : "cublas",
           1000.f * ms / iters, batch, m, n);
  }
  return 0;
}
