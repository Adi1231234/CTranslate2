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

    // Values 4q..4q+3 of a lane's chunk are the 8-byte slot q, slot_stride slots after slot q - 1: 1 for a
    // contiguous row, more where lanes' chunks are interleaved slot by slot (no shared-memory bank conflicts).
    __device__ __forceinline__ void rows1024_load(const __half* p, unsigned n, float* v, unsigned slot_stride = 1) {
      #pragma unroll
      for (unsigned q = 0; q < 8; ++q) {
        if (4 * q < n) {
          const uint2 raw = reinterpret_cast<const uint2*>(p)[q * slot_stride];
          const __half2 a = *reinterpret_cast<const __half2*>(&raw.x);
          const __half2 b = *reinterpret_cast<const __half2*>(&raw.y);
          v[4 * q] = __low2float(a); v[4 * q + 1] = __high2float(a);
          v[4 * q + 2] = __low2float(b); v[4 * q + 3] = __high2float(b);
        }
      }
    }

    __device__ __forceinline__ void rows1024_store(__half* p, unsigned n, const float* e, float sum,
                                                   unsigned slot_stride = 1) {
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
          reinterpret_cast<uint2*>(p)[q * slot_stride] = raw;
        }
      }
    }

    // e / sum as the IEEE division rounds it, from y = RN(1 / sum) computed once per row: q = RN(e y), then two
    // fma corrections q + (e - sum q) y (Markstein), exact whenever nothing underflows, i.e. for e >= 2^-60 with
    // 1 <= sum <= 2048 (a softmax row's sum). tools/turing/kernels/quotient_check.c: every e in [2^-60, 1]
    // against sums at and next to powers of two, and 2e9 random pairs, all equal to the division.
    __device__ __forceinline__ float rows1024_quotient(float e, float sum, float y) {
      float q = __fmul_rn(e, y);
      q = __fmaf_rn(__fmaf_rn(-sum, q, e), y, q);
      return __fmaf_rn(__fmaf_rn(-sum, q, e), y, q);
    }

    // rows1024_store with the quotients of rows1024_quotient; returns whether a value was outside its range
    // (below 2^-60, or NaN), in which case the caller stores the lane's values again with rows1024_store.
    __device__ __forceinline__ bool rows1024_store_fast(__half* p, unsigned n, const float* e, float sum, float y,
                                                        unsigned slot_stride = 1) {
      bool outside = !(sum >= 1.f && sum <= 2048.f);
      #pragma unroll
      for (unsigned q = 0; q < 8; ++q) {
        if (4 * q < n) {
          float v[4];
          #pragma unroll
          for (unsigned k = 0; k < 4; ++k) {
            outside |= !(e[4 * q + k] >= 0x1p-60f);
            v[k] = rows1024_quotient(e[4 * q + k], sum, y);
          }
          __half2 a = __halves2half2(static_cast<__half>(v[0]), static_cast<__half>(v[1]));
          __half2 b = __halves2half2(static_cast<__half>(v[2]), static_cast<__half>(v[3]));
          uint2 raw;
          raw.x = *reinterpret_cast<unsigned*>(&a);
          raw.y = *reinterpret_cast<unsigned*>(&b);
          reinterpret_cast<uint2*>(p)[q * slot_stride] = raw;
        }
      }
      return outside;
    }

    // One row by one warp: the softmax of a row, 1024 < classes <= 2048. The lane's chunks of the row (values
    // 32L.. and 1024 + 32L..) are read at in0 and in1 and written at out0 and out1 (in place is fine), their
    // 8-byte slots stride0 and stride1 slots apart (rows1024_load): a contiguous row (strides 1), or
    // exact_attention's rows interleaved slot by slot in shared memory.
    __device__ __forceinline__ void rows1024_row(__half* out0, __half* out1, const __half* in0,
                                                 const __half* in1, const unsigned stride0,
                                                 const unsigned stride1, const unsigned classes,
                                                 const unsigned lane)
    {
      const unsigned tail = classes - 1024;                 // values past the first 1024
      const unsigned n1 = tail > C10_WARP_SIZE * lane
        ? min(tail - C10_WARP_SIZE * lane, unsigned(C10_WARP_SIZE)) : 0;
      float e0[C10_WARP_SIZE], e1[C10_WARP_SIZE];
      rows1024_load(in0, C10_WARP_SIZE, e0, stride0);
      rows1024_load(in1, n1, e1, stride1);

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

      const float y = __frcp_rn(sum);                       // the divisions, as fma steps (rows1024_quotient)
      bool outside = rows1024_store_fast(out0, C10_WARP_SIZE, e0, sum, y, stride0);
      outside |= rows1024_store_fast(out1, n1, e1, sum, y, stride1);
      if (__any_sync(0xffffffff, outside)) {                // rare: tiny values take the division itself
        rows1024_store(out0, C10_WARP_SIZE, e0, sum, stride0);
        rows1024_store(out1, n1, e1, sum, stride1);
      }
    }

    static __global__ void __launch_bounds__(rows1024_per_block * C10_WARP_SIZE)
    softmax_rows1024(__half* output, const __half* input, const unsigned rows, const unsigned classes)
    {
      const unsigned row = blockIdx.x * rows1024_per_block + threadIdx.x / C10_WARP_SIZE;
      const unsigned lane = threadIdx.x % C10_WARP_SIZE;
      const size_t first = size_t(row) * classes + C10_WARP_SIZE * lane;
      if (row < rows)
        rows1024_row(output + first, output + first + 1024, input + first, input + first + 1024, 1, 1,
                     classes, lane);
    }

  }
}
