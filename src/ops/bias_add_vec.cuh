#pragma once

// fp16 BiasAdd on 16-byte vectors (8 halves per thread) with the per-element arithmetic of the
// thrust paths in bias_add_gpu.cu, where half math is available (CUDA_CAN_USE_HALF, so
// cuda::plus<__half> and __half + __half are __hadd):
//   no activation, no residual:  __hadd(bias, x)                  (add_block_broadcast)
//   residual:                    __hadd(__hadd(bias, x), residual) (trinary_add, plus3)
//   GELU:                        half(gelu_func(float(__hadd(bias, x)))) (bias_add, op_epilogue)
// For a bias along the last dimension (width 1) of a multiple of 8, 16-byte aligned buffers; with width > 1
// (a bias per block of width values, e.g. per channel of a Conv1D output) the 8 values' biases are gathered
// one by one, plain and GELU only.

#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    enum class BiasAddVecMode { plain, residual, gelu };

    // The per-element arithmetic of all modes on 8 values, their 8 biases and (residual mode) 8 residuals.
    template <BiasAddVecMode Mode>
    __device__ __forceinline__ uint4 bias_add_vec_apply(uint4 value, const uint4 b, const uint4 r) {
      __half2* vs = reinterpret_cast<__half2*>(&value);
      const __half2* bs = reinterpret_cast<const __half2*>(&b);
      const __half2* rs = reinterpret_cast<const __half2*>(&r);
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        __half2 s = __hadd2(bs[k], vs[k]);
        if (Mode == BiasAddVecMode::residual)
          s = __hadd2(s, rs[k]);
        if (Mode == BiasAddVecMode::gelu) {
          const float2 f = __half22float2(s);
          const cuda::gelu_func<__half> gelu;
          s = __floats2half2_rn(gelu(f.x), gelu(f.y));
        }
        vs[k] = s;
      }
      return value;
    }

    template <BiasAddVecMode Mode>
    __global__ void bias_add_vec_kernel(const uint4* x, const uint4* bias, const uint4* residual,
                                        uint4* y, unsigned depth_vecs, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      uint4 r;
      if (Mode == BiasAddVecMode::residual)
        r = residual[v];
      y[v] = bias_add_vec_apply<Mode>(x[v], bias[v % depth_vecs], r);
    }

    // Bias per block of width values (value e takes bias[(e / width) % depth]).
    template <BiasAddVecMode Mode>
    __global__ void bias_add_block_kernel(const uint4* x, const __half* bias, uint4* y, size_t width,
                                          size_t depth, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      uint4 b;
      __half* bs = reinterpret_cast<__half*>(&b);
      #pragma unroll
      for (int k = 0; k < 8; ++k)
        bs[k] = bias[((8 * v + k) / width) % depth];
      y[v] = bias_add_vec_apply<Mode>(x[v], b, b);
    }

    // Returns false (nothing launched) when the vector path does not apply.
    inline bool bias_add_vec(BiasAddVecMode mode, const __half* x, const __half* bias,
                             const __half* residual, __half* y, size_t numel, size_t depth,
                             size_t width) {
      auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
      const bool block = width != 1;
      if (cuda::use_stock_kernels() || (block && mode == BiasAddVecMode::residual)
          || (!block && (depth % 8 != 0 || !aligned(bias))) || numel % 8 != 0 || !aligned(x) || !aligned(y)
          || (mode == BiasAddVecMode::residual && !aligned(residual)))
        return false;
      const size_t total = numel / 8;
      if (total == 0)
        return true;
      constexpr unsigned threads = 256;
      const size_t blocks = (total + threads - 1) / threads;
      const auto* xv = reinterpret_cast<const uint4*>(x);
      const auto* bv = reinterpret_cast<const uint4*>(bias);
      const auto* rv = reinterpret_cast<const uint4*>(residual);
      auto* yv = reinterpret_cast<uint4*>(y);
      const unsigned dv = unsigned(depth / 8);
      cudaStream_t stream = cuda::get_cuda_stream();
      if (block && mode == BiasAddVecMode::plain)
        bias_add_block_kernel<BiasAddVecMode::plain><<<blocks, threads, 0, stream>>>(xv, bias, yv, width, depth, total);
      else if (block)
        bias_add_block_kernel<BiasAddVecMode::gelu><<<blocks, threads, 0, stream>>>(xv, bias, yv, width, depth, total);
      else if (mode == BiasAddVecMode::plain)
        bias_add_vec_kernel<BiasAddVecMode::plain><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      else if (mode == BiasAddVecMode::residual)
        bias_add_vec_kernel<BiasAddVecMode::residual><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      else
        bias_add_vec_kernel<BiasAddVecMode::gelu><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      return true;
    }

  }
}
