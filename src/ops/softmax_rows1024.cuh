#pragma once

// Register-resident variant of warp_softmax_forward (softmax_kernels.cuh includes this file after
// its functors) for the case the Whisper encoder spends most of its softmax time on: fp16 rows of
// 1026..2048 values (1500 there), no lengths, where the legacy kernel's block B is 1024. Legacy
// thread v then owns x[v] and x[v + 1024], so warp lane L, which replays legacy threads
// 32L..32L+31, owns exactly x[32L, 32L + 32) and x[1024 + 32L, 1024 + 32L + 32): two contiguous
// chunks, loaded straight into registers with 8-byte loads. No shared memory, so many more rows
// are in flight than with warp_softmax_forward (whose row buffers cap an SM at 8 warps). Every
// float operation is warp_softmax_forward's on the same values in the same order: the max
// (order-free), exp(x - max) once per element, each legacy thread's sum, the 32 lane sums in
// order, then exp / sum. Checked on every row length by tools/turing/kernels/softmax_check.cu.

namespace at {
  namespace native {

    constexpr unsigned rows1024_per_block = 4;

    inline bool softmax_rows1024_applies(const void* x, const void* y, unsigned cols,
                                         unsigned legacy_block, bool is_half, bool is_log,
                                         bool has_lengths) {
      return is_half && !is_log && !has_lengths && legacy_block == 1024 && cols > 1024
        && cols <= 2048 && cols % 4 == 0 && reinterpret_cast<uintptr_t>(x) % 8 == 0
        && reinterpret_cast<uintptr_t>(y) % 8 == 0;
    }

    __device__ __forceinline__ void rows1024_load(const __half* p, unsigned n, float* v) {
      #pragma unroll
      for (unsigned q = 0; q < 8; ++q) {
        if (4 * q < n) {
          const uint2 raw = reinterpret_cast<const uint2*>(p)[q];
          const __half2 a = *reinterpret_cast<const __half2*>(&raw.x);
          const __half2 b = *reinterpret_cast<const __half2*>(&raw.y);
          v[4 * q] = __low2float(a); v[4 * q + 1] = __high2float(a);
          v[4 * q + 2] = __low2float(b); v[4 * q + 3] = __high2float(b);
        }
      }
    }

    __device__ __forceinline__ void rows1024_store(__half* p, unsigned n, const float* e, float sum) {
      #pragma unroll
      for (unsigned q = 0; q < 8; ++q) {
        if (4 * q < n) {
          __half2 a = __halves2half2(static_cast<__half>(e[4 * q] / sum),
                                     static_cast<__half>(e[4 * q + 1] / sum));
          __half2 b = __halves2half2(static_cast<__half>(e[4 * q + 2] / sum),
                                     static_cast<__half>(e[4 * q + 3] / sum));
          uint2 raw;
          raw.x = *reinterpret_cast<unsigned*>(&a);
          raw.y = *reinterpret_cast<unsigned*>(&b);
          reinterpret_cast<uint2*>(p)[q] = raw;
        }
      }
    }

    __global__ void __launch_bounds__(rows1024_per_block * C10_WARP_SIZE)
    softmax_rows1024(__half* output, const __half* input, const unsigned rows, const unsigned classes)
    {
      const unsigned lane = threadIdx.x % C10_WARP_SIZE;
      const unsigned row = blockIdx.x * rows1024_per_block + threadIdx.x / C10_WARP_SIZE;
      if (row >= rows)
        return;
      const size_t first = size_t(row) * classes + C10_WARP_SIZE * lane;
      const unsigned tail = classes - 1024;                 // values past the first 1024
      const unsigned n1 = tail > C10_WARP_SIZE * lane
        ? min(tail - C10_WARP_SIZE * lane, unsigned(C10_WARP_SIZE)) : 0;
      float e0[C10_WARP_SIZE], e1[C10_WARP_SIZE];
      rows1024_load(input + first, C10_WARP_SIZE, e0);
      rows1024_load(input + first + 1024, n1, e1);

      float lane_max = -max_float;
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
        lane_max = MaxFloat<float, float>()(lane_max, e0[i]);
        if (i < n1)
          lane_max = MaxFloat<float, float>()(lane_max, e1[i]);
      }
      #pragma unroll
      for (unsigned offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2)
        lane_max = Max<float>()(lane_max, __shfl_xor_sync(0xffffffff, lane_max, offset));
      const float max_k = lane_max;

      float warp_sum = 0.f;
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {       // legacy thread 32 * lane + i
        float thread_sum = 0.f;
        e0[i] = std::exp(e0[i] - max_k);
        thread_sum = thread_sum + e0[i];
        if (i < n1) {
          e1[i] = std::exp(e1[i] - max_k);
          thread_sum = thread_sum + e1[i];
        }
        warp_sum = Add<float>()(warp_sum, thread_sum);
      }
      float sum = 0.f;
      for (unsigned g = 0; g < C10_WARP_SIZE; ++g)
        sum = Add<float>()(sum, __shfl_sync(0xffffffff, warp_sum, g));

      rows1024_store(output + first, C10_WARP_SIZE, e0, sum);
      rows1024_store(output + first + 1024, n1, e1, sum);
    }

  }
}
