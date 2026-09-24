#include "scores_softmax_fused.h"

#include <stdexcept>

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/scores_softmax.h"
#endif

namespace ctranslate2 {
  namespace layers {

    bool scores_softmax_fusable(const StorageView& queries, const StorageView& keys) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      return queries.device() == Device::CUDA && queries.dtype() == DataType::FLOAT16
        && keys.dtype() == DataType::FLOAT16 && queries.rank() == 4 && keys.rank() == 4
        && queries.dim(0) == keys.dim(0) && queries.dim(1) == keys.dim(1) && queries.dim(3) == keys.dim(3)
        && cuda::attention_scores_softmax_applies(queries.dim(2), keys.dim(2), queries.dim(3),
                                                  queries.buffer(), keys.buffer());
#else
      (void)queries; (void)keys;
      return false;
#endif
    }

    void scores_softmax_fused(const StorageView& queries, const StorageView& keys, float scale,
                              StorageView& attention) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      attention.resize({queries.dim(0), queries.dim(1), queries.dim(2), keys.dim(2)});
      cuda::attention_scores_softmax(queries.data<float16_t>(), keys.data<float16_t>(),
                                     attention.data<float16_t>(), queries.dim(0) * queries.dim(1),
                                     queries.dim(2), keys.dim(2), scale);
#else
      (void)queries; (void)keys; (void)scale; (void)attention;
      throw std::logic_error("scores_softmax_fused requires CUDA");
#endif
    }

  }
}
