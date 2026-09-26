#include "kv_cache.h"

#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/cache_reorder.h"
#  include "cuda/graph.h"
#endif

namespace ctranslate2 {
  namespace layers {

    static StorageView appended_like(const StorageView& cache, const StorageView& fresh) {
      Shape shape = fresh.shape();
      shape[2] += cache.dim(2);
      return StorageView(std::move(shape), cache.dtype(), cache.device());
    }

    void reorder_and_append(StorageView& keys, StorageView& values, const StorageView& order,
                            const StorageView& fresh_keys, const StorageView& fresh_values) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      // In a captured step, the caches grow here: launched as is, their memory from the pool (cuda/graph.h).
      const cuda::CaptureBreak capture_break;
#endif
      StorageView out_keys = appended_like(keys, fresh_keys);
      StorageView out_values = appended_like(values, fresh_values);
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const dim_t d = keys.dim(3);
      auto usable = [d](const StorageView& x) {
        return cuda::cache_reorder_supported(x.buffer(), d, x.item_size());
      };
      if (keys.device() == Device::CUDA && keys.dtype() == DataType::FLOAT16
          && order.device() == Device::CUDA && order.dtype() == DataType::INT32
          && fresh_keys.dim(0) == order.size() && values.shape() == keys.shape()
          && fresh_values.shape() == fresh_keys.shape()
          && usable(keys) && usable(values) && usable(fresh_keys) && usable(fresh_values)
          && usable(out_keys) && usable(out_values)) {
        const float16_t* cache[2] = {keys.data<float16_t>(), values.data<float16_t>()};
        const float16_t* fresh[2] = {fresh_keys.data<float16_t>(), fresh_values.data<float16_t>()};
        float16_t* out[2] = {out_keys.data<float16_t>(), out_values.data<float16_t>()};
        cuda::reorder_append(cache, fresh, out, 2, order.data<int32_t>(), order.size(), keys.dim(1),
                             keys.dim(2), fresh_keys.dim(2), d);
        keys = std::move(out_keys);
        values = std::move(out_values);
        return;
      }
#endif
      ops::Gather()(keys, order);
      ops::Gather()(values, order);
      ops::Concat(2)({&keys, &fresh_keys}, out_keys);
      ops::Concat(2)({&values, &fresh_values}, out_values);
      keys = std::move(out_keys);
      values = std::move(out_values);
    }

  }
}
