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

    // float(bias) + float(x) rounded to half, on 8 packed halves (cuda::plus<__half> per element).
    __device__ __forceinline__ uint4 add_bias8(uint4 x, uint4 b) {
      unsigned* xs = reinterpret_cast<unsigned*>(&x);
      const unsigned* bs = reinterpret_cast<const unsigned*>(&b);
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 xf = __half22float2(*reinterpret_cast<const __half2*>(xs + k));
        const float2 bf = __half22float2(*reinterpret_cast<const __half2*>(bs + k));
        const __half2 s = __floats2half2_rn(bf.x + xf.x, bf.y + xf.y);
        xs[k] = *reinterpret_cast<const unsigned*>(&s);
      }
      return x;
    }

    // One thread per 16-byte vector of x, in x order (coalesced reads); each head's vectors land
    // contiguously in its output row.
    __global__ void split_heads_bias_kernel(const uint4* x, const uint4* bias, SplitOutputs out,
                                            unsigned time, unsigned heads, unsigned head_vecs,
                                            unsigned row_vecs, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      const size_t rt = v / row_vecs;               // r * time + t
      const unsigned c = unsigned(v - rt * row_vecs);
      const unsigned part_vecs = heads * head_vecs;
      const unsigned part = c / part_vecs;
      const unsigned h = (c - part * part_vecs) / head_vecs;
      const unsigned i = c - part * part_vecs - h * head_vecs;
      const size_t r = rt / time, t = rt - r * time;
      const uint4 value = bias ? add_bias8(x[v], bias[c]) : x[v];
      out.p[part][((r * heads + h) * time + t) * head_vecs + i] = value;
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
      split_heads_bias_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(bias), outputs,
        time, heads, head_vecs, row_vecs, total);
    }

  }
}

#endif
