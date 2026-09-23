// Bit-for-bit check of warp_softmax_forward against the upstream cunn_SoftMaxForward (both from
// src/ops/softmax_kernels.cuh, the code the library runs) on EVERY row length 1..2048: softmax and
// log-softmax, fp16 and fp32, without and with per-row lengths (masked rows, including 0 and full).
// Outputs start as different garbage, so an element either kernel leaves unwritten also counts.
// usage: softmax_check      -> mismatching row lengths per case; must print TOTAL 0
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/softmax_kernels.cuh"

using namespace at::native;

template <typename T>
unsigned long long run_case(bool log, bool masked, unsigned rows, int* bad_cols, int* n_bad) {
  const unsigned max_cols = warp_softmax_max_cols;
  T *x, *y1, *y2;
  int32_t* len;
  unsigned long long *dcount, total = 0;
  CK(cudaMalloc(&x, sizeof(T) * rows * max_cols));
  CK(cudaMalloc(&y1, sizeof(T) * rows * max_cols));
  CK(cudaMalloc(&y2, sizeof(T) * rows * max_cols));
  CK(cudaMalloc(&len, sizeof(int32_t) * rows));
  CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  for (unsigned cols = 1; cols <= max_cols; ++cols) {
    const uint32_t seed = cols * 7919u + log * 104729u + masked * 1299709u + sizeof(T);
    fill<<<256, 256>>>(x, (size_t)rows * cols, seed, -8, 6);          // |x| in [2^-8, 64)
    fill_int<<<1, 128>>>(len, rows, seed ^ 0x5bd1e995u, cols);
    CK(cudaMemset(y1, 0x11, sizeof(T) * rows * cols));
    CK(cudaMemset(y2, 0x22, sizeof(T) * rows * cols));
    const int32_t* lengths = masked ? len : nullptr;
    if (log) {
      softmax_rows<T, LogSoftMaxForwardEpilogue>(0, x, y1, rows, cols, lengths, false);
      softmax_rows<T, LogSoftMaxForwardEpilogue>(0, x, y2, rows, cols, lengths, true);
    } else {
      softmax_rows<T, SoftMaxForwardEpilogue>(0, x, y1, rows, cols, lengths, false);
      softmax_rows<T, SoftMaxForwardEpilogue>(0, x, y2, rows, cols, lengths, true);
    }
    CK(cudaGetLastError());
    CK(cudaMemset(dcount, 0, sizeof(unsigned long long)));
    count_diff<<<256, 256>>>(y1, y2, (size_t)rows * cols, dcount);
    unsigned long long d = 0;
    CK(cudaMemcpy(&d, dcount, sizeof(d), cudaMemcpyDeviceToHost));
    if (d && *n_bad < 16) bad_cols[(*n_bad)++] = cols;
    else if (d) ++*n_bad;
    total += d;
  }
  cudaFree(x); cudaFree(y1); cudaFree(y2); cudaFree(len); cudaFree(dcount);
  return total;
}

int main() {
  const unsigned rows = 67;                       // several 4-row blocks and a partial one
  unsigned long long total = 0;
  for (int t = 0; t < 2; ++t)
    for (int log = 0; log < 2; ++log)
      for (int masked = 0; masked < 2; ++masked) {
        int bad_cols[16], n_bad = 0;
        const unsigned long long d = t == 0 ? run_case<__half>(log, masked, rows, bad_cols, &n_bad)
                                            : run_case<float>(log, masked, rows, bad_cols, &n_bad);
        printf("%s %-11s %-9s: %llu mismatching values in %d of 2048 row lengths", t == 0 ? "fp16" : "fp32",
               log ? "log-softmax" : "softmax", masked ? "masked" : "unmasked", d, n_bad);
        for (int i = 0; i < n_bad && i < 16; ++i) printf("%s%d", i ? "," : " (cols ", bad_cols[i]);
        printf("%s\n", n_bad ? ")" : "");
        total += d;
      }
  printf("TOTAL %llu mismatching values\n", total);
  return 0;
}
