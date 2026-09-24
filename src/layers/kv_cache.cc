#include "kv_cache.h"

#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/cache_reorder.h"
#endif

namespace ctranslate2 {
  namespace layers {

    void reorder_and_append(StorageView& cache, const StorageView& order, const StorageView& fresh) {
      const dim_t rows = order.size();
      Shape shape = fresh.shape();
      shape[2] += cache.dim(2);
      StorageView out(std::move(shape), cache.dtype(), cache.device());
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      if (cache.device() == Device::CUDA && cache.dtype() == DataType::FLOAT16
          && order.device() == Device::CUDA && order.dtype() == DataType::INT32 && fresh.dim(0) == rows
          && cuda::cache_reorder_supported(cache.buffer(), fresh.buffer(), out.buffer(), cache.dim(3),
                                           cache.item_size())) {
        cuda::reorder_append(cache.data<float16_t>(), order.data<int32_t>(), fresh.data<float16_t>(),
                             out.data<float16_t>(), rows, cache.dim(1), cache.dim(2), fresh.dim(2),
                             cache.dim(3));
        cache = std::move(out);
        return;
      }
#endif
      ops::Gather()(cache, order);
      ops::Concat(2)({&cache, &fresh}, out);
      cache = std::move(out);
    }

  }
}
