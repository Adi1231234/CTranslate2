#ifndef CT2_USE_HIP

#include "cuda/cross_attention.h"

#include <algorithm>

#include "cross_attention.cuh"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    bool cross_attention_applies(dim_t keys, dim_t depth) {
      static const bool enabled = read_bool_from_env("CT2_CROSS_ATTN", true);
      return enabled && keys == at::native::ca_keys && depth == at::native::ca_depth && hmma_replicas_verified();
    }

    void cross_attention_layout(const float16_t* k, const float16_t* v, float16_t* kf, float16_t* vf,
                                dim_t entries) {
      at::native::cross_attention_layout<<<1024, 256, 0, get_cuda_stream()>>>(
        reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(v),
        reinterpret_cast<uint4*>(kf), reinterpret_cast<uint4*>(vf), static_cast<size_t>(entries));
    }

    void cross_attention(const float16_t* q, const float16_t* kf, const float16_t* vf, float16_t* o,
                         dim_t clips, dim_t heads, dim_t m, float alpha) {
      const int rows = static_cast<int>(std::min<dim_t>(m, 8));   // queries per pass (the mma's n)
      const int smem = rows * at::native::ca_pitch * static_cast<int>(sizeof (__half));
      at::native::cross_attention_kernel<<<static_cast<unsigned>(clips * heads), at::native::ca_warps * 32, smem,
                                           get_cuda_stream()>>>(
        reinterpret_cast<const __half*>(q), reinterpret_cast<const uint4*>(kf), reinterpret_cast<const uint4*>(vf),
        reinterpret_cast<__half*>(o), static_cast<int>(heads), static_cast<int>(m), rows, alpha);
    }

  }
}

#endif
