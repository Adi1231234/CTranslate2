#ifndef CT2_USE_HIP

#include "cuda/exact_attention.h"

#include <cstdint>

#include "exact_attention.cuh"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    static bool aligned4(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 4 == 0;
    }

    bool exact_attention_applies(dim_t m, dim_t n, dim_t depth, const void* q, const void* k, const void* v) {
      return depth == at::native::ea_depth && m == 1500 && n == 1500
        && aligned4(q) && aligned4(k) && aligned4(v) && hmma_replicas_verified();
    }

    size_t exact_attention_workspace_bytes(dim_t batch, dim_t n) {
      return sizeof (float16_t) * batch * at::native::ea_depth * n;
    }

    void exact_attention(const float16_t* q, const float16_t* k, const float16_t* v, void* workspace,
                         float16_t* o, dim_t batch, dim_t m, dim_t n, float alpha) {
      at::native::exact_attention(reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
                                  reinterpret_cast<const __half*>(v), static_cast<__half*>(workspace),
                                  reinterpret_cast<__half*>(o), static_cast<int>(batch), static_cast<int>(m),
                                  static_cast<int>(n), alpha, get_cuda_stream());
    }

  }
}

#endif
