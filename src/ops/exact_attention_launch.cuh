#pragma once

// Host side of exact_attention.cuh, shared by CTranslate2 (ops/exact_attention_gpu.cu) and the bit-for-bit check
// (tools/turing/kernels/exact_attention_check.cu).

#include "exact_attention.cuh"

namespace at {
  namespace native {

    // Bytes of the kf and vf workspace for batch entries of n keys.
    inline size_t exact_attention_workspace(int batch, int n) {
      return sizeof (uint2) * batch * (eal_key_tiles(n) * 4 + (ea_depth / 8) * eal_groups(n)) * eal_lanes;
    }

    // o = SoftMax(q k^T * alpha) v for q [batch, m, 64], k and v [batch, n, 64] with batch = clips x heads, o
    // [clips, m, heads, 64] (the heads combined, as MultiHeadAttention::combine_heads would lay them out; heads 1
    // gives [batch, m, 64]); workspace: exact_attention_workspace(batch, n) bytes. m = n = 1500 (the kernel is
    // compiled for that shape; exact_attention_applies checks it), all 4-byte aligned. With a work counter
    // (zero, cuda/persistent.h), `blocks` blocks in total take the work items; else a block per item.
    inline void exact_attention(const __half* q, const __half* k, const __half* v, void* workspace, __half* o,
                                int batch, int heads, int m, int n, float alpha, cudaStream_t stream,
                                unsigned* counter = nullptr, int blocks = 0) {
      (void)m;
      constexpr int N = 1500;
      using S = ea_shape<N>;
      const int smem = int(ea_rows * S::pitch * sizeof (__half));
      static const bool configured = cudaFuncSetAttribute(exact_attention_kernel<N>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      uint2* kf = static_cast<uint2*>(workspace);
      uint2* vf = kf + (size_t)batch * eal_key_tiles(n) * 4 * eal_lanes;
      exact_attention_layout<<<1024, 256, 0, stream>>>(k, v, kf, vf, batch, n, ea_depth);
      if (counter)
        exact_attention_kernel<N><<<blocks, ea_warps * C10_WARP_SIZE, smem, stream>>>(q, kf, vf, o, heads, alpha,
                                                                                     counter, batch);
      else
        exact_attention_kernel<N><<<dim3(S::row_tiles, batch), ea_warps * C10_WARP_SIZE, smem, stream>>>(
          q, kf, vf, o, heads, alpha, nullptr, batch);
    }

  }
}
