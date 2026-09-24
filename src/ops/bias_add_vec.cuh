#pragma once

// fp16 BiasAdd on 16-byte vectors (8 halves per thread) with the per-element arithmetic of the
// thrust paths in bias_add_gpu.cu, where half math is available (CUDA_CAN_USE_HALF, so
// cuda::plus<__half> and __half + __half are __hadd):
//   no activation, no residual:  __hadd(bias, x)                  (add_block_broadcast)
//   residual:                    __hadd(__hadd(bias, x), residual) (trinary_add, plus3)
//   GELU:                        half(gelu_func(float(__hadd(bias, x)))) (bias_add, op_epilogue)
// Only for a bias along the last dimension (width 1) of a multiple of 8, 16-byte aligned buffers.

#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    enum class BiasAddVecMode { plain, residual, gelu };

    template <BiasAddVecMode Mode>
    __global__ void bias_add_vec_kernel(const uint4* x, const uint4* bias, const uint4* residual,
                                        uint4* y, unsigned depth_vecs, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      uint4 value = x[v];
      const uint4 b = bias[v % depth_vecs];
      uint4 r;
      if (Mode == BiasAddVecMode::residual)
        r = residual[v];
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
      y[v] = value;
    }

    // Returns false (nothing launched) when the vector path does not apply.
    inline bool bias_add_vec(BiasAddVecMode mode, const __half* x, const __half* bias,
                             const __half* residual, __half* y, size_t numel, size_t depth,
                             size_t width) {
      auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
      if (cuda::use_stock_kernels() || width != 1 || depth % 8 != 0
          || numel % 8 != 0 || !aligned(x) || !aligned(bias) || !aligned(y)
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
      if (mode == BiasAddVecMode::plain)
        bias_add_vec_kernel<BiasAddVecMode::plain><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      else if (mode == BiasAddVecMode::residual)
        bias_add_vec_kernel<BiasAddVecMode::residual><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      else
        bias_add_vec_kernel<BiasAddVecMode::gelu><<<blocks, threads, 0, stream>>>(xv, bv, rv, yv, dv, total);
      return true;
    }

  }
}
