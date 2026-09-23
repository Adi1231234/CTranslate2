// Device-side data for the probes: deterministic random halves, constant fills, bitwise diff counts.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#define GRID_STRIDE(t, count) \
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (count); t += (size_t)gridDim.x * blockDim.x)

// Halves with random sign and magnitude 2^[lo, hi): exercises rounding in every partial sum.
__global__ void fill(__half* x, size_t count, uint32_t seed, int lo, int hi) {
  GRID_STRIDE(t, count) {
    uint32_t h = (uint32_t)t * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const float mant = 1.f + (h & 1023) / 1024.f;
    const int e = lo + (int)((h >> 10) % (uint32_t)(hi - lo));
    x[t] = __float2half(((h >> 31) ? -1.f : 1.f) * ldexpf(mant, e));
  }
}

__global__ void set_bits(__half* x, size_t count, unsigned short bits) {
  GRID_STRIDE(t, count) x[t] = __ushort_as_half(bits);
}

__global__ void count_diff(const __half* a, const __half* b, size_t count, unsigned long long* out) {
  unsigned long long local = 0;
  GRID_STRIDE(t, count) local += __half_as_ushort(a[t]) != __half_as_ushort(b[t]);
  if (local) atomicAdd(out, local);
}

// The check's data for one (m, batch) case, so a failing case can be replayed exactly.
inline void fill_case(__half* dK, __half* dQ, int batch, int m, int n, int k) {
  const uint32_t seed = (uint32_t)(m * 100003 + batch);
  const int lo = -10 + (batch % 5), hi = 1 + (batch % 4);        // |q|, |k| in [2^-10, 16)
  fill<<<1024, 256>>>(dK, (size_t)batch * n * k, seed, lo, hi);
  fill<<<64, 256>>>(dQ, (size_t)batch * m * k, seed ^ 0x9e3779b9u, lo, hi);
}
