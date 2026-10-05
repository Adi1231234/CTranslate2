// Bit-for-bit check of src/cuda/softmax_parts.cu (the softmax of several score tensors in one launch) against the
// library's softmax_rows on each tensor alone, as ops::SoftMax runs it: random parts of rows x 20 heads x 1 query
// x t keys, t 1..448 (every length a decoding step's self-attention has) and some up to 1024, fp16 scores of a
// decoder's range. Must end with TOTAL 0.
// usage: softmax_parts_check
#include <cstdio>
#include <random>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/softmax_parts.cu"

namespace ctranslate2 {
  namespace cuda {
    bool use_stock_kernels() { return false; }                // the probe links no library
    cudaStream_t get_cuda_stream() { return 0; }
  }
}

int main() {
  std::mt19937 rng(7);
  const int max_parts = ctranslate2::cuda::SoftmaxParts::max_parts;
  __half *X, *R; unsigned long long* dc;
  const size_t cap = 16ull * 40 * 20 * 1024;
  CK(cudaMalloc(&X, 2 * cap)); CK(cudaMalloc(&R, 2 * cap)); CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0, values = 0;
  for (int trial = 0; trial < 600; ++trial) {
    ctranslate2::cuda::SoftmaxParts parts;
    std::vector<size_t> offset, rows, cols;
    size_t used = 0;
    unsigned all_rows = 0;
    const int count = 1 + int(rng() % max_parts);
    for (int p = 0; p < count; ++p) {
      const unsigned r = 20 * (1 + rng() % 40), t = trial % 50 == 0 ? 449 + rng() % 576 : 1 + rng() % 448;
      offset.push_back(used); rows.push_back(r); cols.push_back(t);
      fill<<<256, 256>>>(X + used, (size_t)r * t, 1000u * trial + p, -6, 4);
      all_rows += r;
      parts.data[parts.count] = R + used;
      parts.rows_end[parts.count] = all_rows;
      parts.cols[parts.count++] = t;
      used += (size_t)r * t;
    }
    CK(cudaMemcpy(R, X, 2 * used, cudaMemcpyDeviceToDevice));
    for (int p = 0; p < count; ++p)                            // the reference, in place in X
      at::native::softmax_rows<__half, at::native::SoftMaxForwardEpilogue>(0, X + offset[p], X + offset[p],
                                                                            rows[p], cols[p], nullptr, true);
    ctranslate2::cuda::softmax_parts(parts);
    CK(cudaGetLastError());
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(X, R, used, dc);
    unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost));
    total += d; values += used;
  }
  printf("600 launches, %llu values: %llu mismatched\nTOTAL %llu mismatches\n", values, total, total);
  return 0;
}
