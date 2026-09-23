// Device-side data for the probes: deterministic random values, constant fills, bitwise diff counts.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#define GRID_STRIDE(t, count) \
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (count); t += (size_t)gridDim.x * blockDim.x)

__device__ __forceinline__ uint32_t mix(uint32_t h) {
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  return h;
}

// Random sign and magnitude 2^[lo, hi) (10 mantissa bits): exercises rounding in every partial sum.
template <typename T>
__global__ void fill(T* x, size_t count, uint32_t seed, int lo, int hi) {
  GRID_STRIDE(t, count) {
    const uint32_t h = mix((uint32_t)t * 2654435761u ^ seed);
    const float mant = 1.f + (h & 1023) / 1024.f;
    const int e = lo + (int)((h >> 10) % (uint32_t)(hi - lo));
    x[t] = T(((h >> 31) ? -1.f : 1.f) * ldexpf(mant, e));
  }
}

// Random integers in [0, bound] (per-row lengths).
__global__ void fill_int(int32_t* x, size_t count, uint32_t seed, uint32_t bound) {
  GRID_STRIDE(t, count) x[t] = (int32_t)(mix((uint32_t)t * 2654435761u ^ seed) % (bound + 1));
}

__global__ void set_bits(__half* x, size_t count, unsigned short bits) {
  GRID_STRIDE(t, count) x[t] = __ushort_as_half(bits);
}

__device__ __forceinline__ uint32_t bits_of(__half v) { return __half_as_ushort(v); }
__device__ __forceinline__ uint32_t bits_of(float v) { return __float_as_uint(v); }

template <typename T>
__global__ void count_diff(const T* a, const T* b, size_t count, unsigned long long* out) {
  unsigned long long local = 0;
  GRID_STRIDE(t, count) local += bits_of(a[t]) != bits_of(b[t]);
  if (local) atomicAdd(out, local);
}

// The qk_check data for one (m, batch) case, so a failing case can be replayed exactly.
inline void fill_case(__half* dK, __half* dQ, int batch, int m, int n, int k) {
  const uint32_t seed = (uint32_t)(m * 100003 + batch);
  const int lo = -10 + (batch % 5), hi = 1 + (batch % 4);        // |q|, |k| in [2^-10, 16)
  fill<<<1024, 256>>>(dK, (size_t)batch * n * k, seed, lo, hi);
  fill<<<64, 256>>>(dQ, (size_t)batch * m * k, seed ^ 0x9e3779b9u, lo, hi);
}
