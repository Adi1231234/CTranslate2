#ifndef CT2_USE_HIP

#include "cuda/scores_softmax.h"

#include <cstdint>

#include "attention_scores_softmax.cuh"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    bool attention_scores_softmax_applies(dim_t m, dim_t n, dim_t depth, const void* q, const void* k) {
      return depth == at::native::ass_depth && m == 1500 && n == 1500
        && reinterpret_cast<uintptr_t>(q) % 4 == 0 && reinterpret_cast<uintptr_t>(k) % 4 == 0
        && hmma_replicas_verified();
    }

    void attention_scores_softmax(const float16_t* q, const float16_t* k, float16_t* p,
                                  dim_t batch, dim_t m, dim_t n, float alpha) {
      at::native::attention_scores_softmax(reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
                                           reinterpret_cast<__half*>(p), static_cast<int>(batch),
                                           static_cast<int>(m), static_cast<int>(n), alpha, get_cuda_stream());
    }

  }
}

#endif
