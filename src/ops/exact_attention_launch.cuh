#pragma once

// Host side of exact_attention.cuh, shared by CTranslate2 (ops/exact_attention_gpu.cu) and the bit-for-bit check
// (tools/turing/kernels/exact_attention_check.cu).

#include "exact_attention.cuh"

namespace at {
  namespace native {

    // Bytes of the workspace for batch entries of n keys: kf and vf, and with `queries` the head-split queries
    // that exact_attention_qkv writes there.
    inline size_t exact_attention_workspace(int batch, int n, bool queries = false) {
      return sizeof (uint2) * batch * (eal_key_tiles(n) * 4 + (ea_depth / 8) * eal_groups(n)) * eal_lanes
        + (queries ? sizeof (__half) * batch * n * ea_depth : 0);
    }

    // o = SoftMax(q k^T * alpha) v, batch = clips x heads entries of m = n = 1500 rows of 64 dims (the kernel is
    // compiled for that shape; exact_attention_applies checks it); keys and values read where their EalSource
    // says; the queries head-split at q, or (q null) made from qs by the layout kernel into the workspace. o is
    // [clips, m, heads, 64] (the heads combined, as MultiHeadAttention::combine_heads would lay them out; heads 1
    // gives [batch, m, 64]); workspace: exact_attention_workspace(batch, n, !q) bytes. With a work counter (zero,
    // cuda/persistent.h), `blocks` blocks in total take the work items; else a block per item.
    template <int N, int RT>
    inline void exact_attention_items(const __half* queries, const uint2* kf, const uint2* vf, __half* o, int batch,
                                      int heads, float alpha, cudaStream_t stream, unsigned* counter, int blocks) {
      const int smem = int(RT * ea_rows * ea_shape<N>::pitch * sizeof (__half));
      static const bool configured = cudaFuncSetAttribute(exact_attention_kernel<N, RT>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      if (counter)
        exact_attention_kernel<N, RT><<<blocks, ea_warps * C10_WARP_SIZE, smem, stream>>>(queries, kf, vf, o, heads,
                                                                                         alpha, counter, batch);
      else
        exact_attention_kernel<N, RT><<<dim3(ea_items_per_entry<N, RT>, batch), ea_warps * C10_WARP_SIZE, smem,
                                        stream>>>(queries, kf, vf, o, heads, alpha, nullptr, batch);
    }

    // rows: queries a work item takes, 16 or 32 (exact_attention.cuh's RT x 16; the same bits either way).
    inline void exact_attention(const __half* q, const EalSource& qs, const EalSource& k, const EalSource& v,
                                void* workspace, __half* o, int batch, int heads, int n, float alpha,
                                cudaStream_t stream, unsigned* counter, int blocks, int rows = 16) {
      constexpr int N = 1500;
      uint2* kf = static_cast<uint2*>(workspace);
      uint2* vf = kf + (size_t)batch * eal_key_tiles(n) * 4 * eal_lanes;
      __half* qw = q ? nullptr : reinterpret_cast<__half*>(vf + (size_t)batch * (ea_depth / 8) * eal_groups(n) * eal_lanes);
      exact_attention_layout<<<1024, 256, 0, stream>>>(qs, qw, k, v, heads, kf, vf, batch, n, ea_depth);
      const __half* queries = q ? q : qw;
      if (rows == 32)
        exact_attention_items<N, 2>(queries, kf, vf, o, batch, heads, alpha, stream, counter, blocks);
      else
        exact_attention_items<N, 1>(queries, kf, vf, o, batch, heads, alpha, stream, counter, blocks);
    }

    // For head-split q [batch, m, 64] and k, v [batch, n, 64].
    inline void exact_attention(const __half* q, const __half* k, const __half* v, void* workspace, __half* o,
                                int batch, int heads, int m, int n, float alpha, cudaStream_t stream,
                                unsigned* counter = nullptr, int blocks = 0, int rows = 16) {
      exact_attention(q, eal_split(q, heads, m, ea_depth), eal_split(k, heads, n, ea_depth),
                      eal_split(v, heads, n, ea_depth), workspace, o, batch, heads, n, alpha, stream, counter, blocks,
                      rows);
    }

    // From the fused projection x [clips, n, 3 * heads * 64] without its bias, and that bias (or null): q, k and v
    // are its three parts plus their bias; workspace: exact_attention_workspace(clips * heads, n, true) bytes.
    inline void exact_attention_qkv(const __half* x, const __half* bias, void* workspace, __half* o, int clips,
                                    int heads, int n, float alpha, cudaStream_t stream, unsigned* counter = nullptr,
                                    int blocks = 0, int rows = 16) {
      const long long part = (long long)heads * ea_depth, row = 3 * part, clip = row * n;
      auto source = [&](int p) {
        return EalSource{x + p * part, bias ? bias + p * part : nullptr, clip, ea_depth, row};
      };
      exact_attention(nullptr, source(0), source(1), source(2), workspace, o, clips * heads, heads, n, alpha,
                      stream, counter, blocks, rows);
    }

  }
}
