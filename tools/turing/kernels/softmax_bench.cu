// Speed of the row softmax kernels on one shape (default: the Whisper encoder's attention of a
// batch of 8 clips, 8 x 20 heads x 1500 rows of 1500), fp16, and a bitwise comparison of their
// outputs there: the legacy cunn_SoftMaxForward, warp_softmax_forward and what softmax_rows picks.
// usage: softmax_bench [rows=240000] [cols=1500] [repeats=20]
#include <cstdio>
#include <cstdlib>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/softmax_kernels.cuh"

using namespace at::native;

template <typename F>
float time_ms(F run, int repeats) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run();                                            // warm-up
  CK(cudaEventRecord(a));
  for (int i = 0; i < repeats; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
  return ms / repeats;
}

unsigned long long diff(const __half* a, const __half* b, size_t n, unsigned long long* d) {
  CK(cudaMemset(d, 0, sizeof(*d)));
  count_diff<<<1024, 256>>>(a, b, n, d);
  unsigned long long h = 0;
  CK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  return h;
}

int main(int argc, char** argv) {
  const unsigned rows = argc > 1 ? atoi(argv[1]) : 240000, cols = argc > 2 ? atoi(argv[2]) : 1500;
  const int repeats = argc > 3 ? atoi(argv[3]) : 20;
  const size_t n = (size_t)rows * cols;
  __half *x, *y0, *y1, *y2;
  unsigned long long* d;
  CK(cudaMalloc(&x, n * 2)); CK(cudaMalloc(&y0, n * 2)); CK(cudaMalloc(&y1, n * 2)); CK(cudaMalloc(&y2, n * 2));
  CK(cudaMalloc(&d, sizeof(*d)));
  fill<<<1024, 256>>>(x, n, 12345u, -8, 3);          // |x| in [2^-8, 8): attention-score range
  const unsigned block = ctranslate2::cuda::get_block_size(cols).x;
  const size_t smem = warp_softmax_rows_per_block * (warp_softmax_slot(cols) + 1) * sizeof(float);
  const unsigned wgrid = (rows + warp_softmax_rows_per_block - 1) / warp_softmax_rows_per_block;
  const float t0 = time_ms([&] { softmax_rows<__half, SoftMaxForwardEpilogue>(0, x, y0, rows, cols, nullptr, false); }, repeats);
  const float t1 = time_ms([&] {
    warp_softmax_forward<__half, false><<<wgrid, warp_softmax_rows_per_block * 32, smem>>>(y1, x, rows, cols, block, nullptr);
  }, repeats);
  const float t2 = time_ms([&] { softmax_rows<__half, SoftMaxForwardEpilogue>(0, x, y2, rows, cols, nullptr, true); }, repeats);
  CK(cudaGetLastError());
  const double gb = 2.0 * n * 2 / 1e9;              // read + write, fp16
  printf("rows %u cols %u: legacy %.3f ms, warp %.3f ms, softmax_rows %.3f ms (%.0f GB/s)\n",
         rows, cols, t0, t1, t2, gb / (t2 / 1e3));
  const unsigned long long d1 = diff(y0, y1, n, d), d2 = diff(y0, y2, n, d);
  printf("mismatching values vs legacy: warp %llu, softmax_rows %llu\n", d1, d2);
  printf("TOTAL %llu mismatch\n", d1 + d2);
  return 0;
}
