// Check of src/cuda/cache_reorder.cu's reorder_append_parts (several searches' self-attention caches: each part's
// beam order applied and its step appended, in one launch) against a plain per-element gather: random parts (1..16,
// 5..40 rows of 20 heads x 64 fp16 dims, 1..448 cached steps, a random beam order or none). Pure data movement:
// every output vector must equal the reference's. Must end with TOTAL 0.
// usage: cache_reorder_check
#include <cstdio>
#include <random>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/cache_reorder.cu"

namespace ctranslate2 {
  namespace cuda {
    cudaStream_t get_cuda_stream() { return 0; }              // the probe links no library
  }
}

// out[r, h, s] = cache[order[r], h, s] for s < time, fresh[r, h, 0] at s = time (one 16-byte vector per thread).
__global__ void reference(const uint4* cache, const uint4* fresh, const int32_t* order, uint4* out, int rows,
                          int heads, int time) {
  const int vecs = 8;
  GRID_STRIDE(v, (size_t)rows * heads * (time + 1) * vecs) {
    const size_t i = v % vecs, s = (v / vecs) % (time + 1), rh = v / (vecs * (time + 1));
    const size_t r = rh / heads, h = rh % heads, from = order ? size_t(order[r]) : r;
    out[v] = s < (size_t)time ? cache[((from * heads + h) * time + s) * vecs + i] : fresh[rh * vecs + i];
  }
}

int main() {
  std::mt19937 rng(5);
  const int heads = 20, max_rows = 40, max_time = 448;
  const size_t cache_vecs = (size_t)max_rows * heads * max_time * 8;
  const size_t out_vecs = (size_t)max_rows * heads * (max_time + 1) * 8;
  const int P = ctranslate2::cuda::CacheParts::max_parts;
  std::vector<uint4*> cache(2 * P), fresh(2 * P), out(2 * P), ref(2 * P);
  std::vector<int32_t*> order(P);
  for (int i = 0; i < 2 * P; ++i) {
    CK(cudaMalloc(&cache[i], 16 * cache_vecs)); CK(cudaMalloc(&fresh[i], 16ull * max_rows * heads * 8));
    CK(cudaMalloc(&out[i], 16 * out_vecs)); CK(cudaMalloc(&ref[i], 16 * out_vecs));
  }
  for (int p = 0; p < P; ++p) CK(cudaMalloc(&order[p], 4 * max_rows));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0, vectors = 0;
  for (int trial = 0; trial < 300; ++trial) {
    ctranslate2::cuda::CacheParts parts;
    parts.count = 1 + int(rng() % P);
    for (int p = 0; p < parts.count; ++p) {
      const int rows = 5 + int(rng() % (max_rows - 4)), time = 1 + int(rng() % max_time);
      std::vector<int32_t> o(rows);
      for (int r = 0; r < rows; ++r) o[r] = int32_t(rng() % rows);
      CK(cudaMemcpy(order[p], o.data(), 4 * rows, cudaMemcpyHostToDevice));
      const bool ordered = rng() % 4 != 0;
      for (int c = 0; c < 2; ++c) {
        const int i = 2 * p + c;
        fill<<<256, 256>>>(reinterpret_cast<__half*>(cache[i]), (size_t)rows * heads * time * 64, 97u * trial + i,
                           -8, 4);
        fill<<<64, 256>>>(reinterpret_cast<__half*>(fresh[i]), (size_t)rows * heads * 64, 31u * trial + i, -8, 4);
        parts.cache[p][c] = cache[i]; parts.fresh[p][c] = fresh[i]; parts.out[p][c] = out[i];
        reference<<<256, 256>>>(cache[i], fresh[i], ordered ? order[p] : nullptr, ref[i], rows, heads, time);
      }
      parts.order[p] = ordered ? order[p] : nullptr;
      parts.rows[p] = rows; parts.time[p] = time;
    }
    ctranslate2::cuda::reorder_append_parts(parts, heads, 64);
    CK(cudaGetLastError());
    for (int p = 0; p < parts.count; ++p)
      for (int c = 0; c < 2; ++c) {
        const size_t n = (size_t)parts.rows[p] * heads * (parts.time[p] + 1) * 64;
        CK(cudaMemset(dc, 0, 8));
        count_diff<<<256, 256>>>(reinterpret_cast<__half*>(ref[2 * p + c]), reinterpret_cast<__half*>(out[2 * p + c]),
                                 n, dc);
        unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost));
        total += d; vectors += n;
      }
  }
  printf("300 launches, %llu values: %llu mismatched\nTOTAL %llu mismatches\n", vectors, total, total);
  return 0;
}
