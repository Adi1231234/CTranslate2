#ifndef CT2_USE_HIP

#include "cuda/split_heads.h"

#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int split_heads_vec = split_heads_bias_granule;   // halves per 16-byte vector

    struct SplitOutputs {
      uint4* p[3];
    };

    // bias + x on 8 packed halves with __hadd, which is cuda::plus<__half> per element.
    __device__ __forceinline__ uint4 add_bias8(uint4 x, uint4 b) {
      __half2* xs = reinterpret_cast<__half2*>(&x);
      const __half2* bs = reinterpret_cast<const __half2*>(&b);
      #pragma unroll
      for (int k = 0; k < 4; ++k)
        xs[k] = __hadd2(bs[k], xs[k]);
      return x;
    }

    // One thread per 16-byte vector of x, in x order (coalesced reads); each head's vectors land
    // contiguously in its output row. Index: the vector counts' type, 32 bits whenever they fit (a 64-bit
    // division per vector costs more issue slots than the copy).
    template <typename Index>
    __global__ void split_heads_bias_kernel(const uint4* x, const uint4* bias, SplitOutputs out,
                                            unsigned time, unsigned heads, unsigned head_vecs,
                                            unsigned row_vecs, Index total) {
      const Index v = Index(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      const Index rt = v / row_vecs;                // r * time + t
      const unsigned c = unsigned(v - rt * row_vecs);
      const unsigned part_vecs = heads * head_vecs;
      const unsigned part = c / part_vecs;
      const unsigned h = (c - part * part_vecs) / head_vecs;
      const unsigned i = c - part * part_vecs - h * head_vecs;
      const Index r = rt / time, t = rt - r * time;
      const uint4 value = bias ? add_bias8(x[v], bias[c]) : x[v];
      out.p[part][((size_t(r) * heads + h) * time + t) * head_vecs + i] = value;
    }

    bool split_heads_bias_aligned(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 16 == 0;
    }

    void split_heads_bias(const float16_t* x, const float16_t* bias, float16_t* const* out, int parts,
                          dim_t rows, dim_t time, dim_t heads, dim_t head_dim) {
      SplitOutputs outputs{};
      for (int p = 0; p < parts; ++p)
        outputs.p[p] = reinterpret_cast<uint4*>(out[p]);
      const unsigned head_vecs = head_dim / split_heads_vec;
      const unsigned row_vecs = parts * heads * head_vecs;
      const size_t total = size_t(rows) * time * row_vecs;
      if (total == 0)
        return;
      constexpr unsigned threads = 256;
      const size_t blocks = (total + threads - 1) / threads;
      const auto* xv = reinterpret_cast<const uint4*>(x);
      const auto* bv = reinterpret_cast<const uint4*>(bias);
      if (total <= UINT32_MAX - threads)
        split_heads_bias_kernel<unsigned><<<blocks, threads, 0, get_cuda_stream()>>>(
          xv, bv, outputs, time, heads, head_vecs, row_vecs, unsigned(total));
      else
        split_heads_bias_kernel<size_t><<<blocks, threads, 0, get_cuda_stream()>>>(
          xv, bv, outputs, time, heads, head_vecs, row_vecs, total);
    }

  }
}

#endif
