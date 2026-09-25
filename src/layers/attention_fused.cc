#include "attention_fused.h"

#include <stdexcept>

#include "ctranslate2/allocator.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/exact_attention.h"
#endif

namespace ctranslate2 {
  namespace layers {

    bool attention_fusable(const StorageView& queries, const StorageView& keys, const StorageView& values) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      return queries.device() == Device::CUDA && queries.dtype() == DataType::FLOAT16
        && keys.dtype() == DataType::FLOAT16 && values.dtype() == DataType::FLOAT16
        && queries.rank() == 4 && keys.shape() == values.shape()
        && queries.dim(0) == keys.dim(0) && queries.dim(1) == keys.dim(1) && queries.dim(3) == keys.dim(3)
        && cuda::exact_attention_applies(queries.dim(2), keys.dim(2), queries.dim(3),
                                         queries.buffer(), keys.buffer(), values.buffer());
#else
      (void)queries; (void)keys; (void)values;
      return false;
#endif
    }

    void attention_fused(const StorageView& queries, const StorageView& keys, const StorageView& values,
                         float scale, StorageView& output) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const dim_t batch = queries.dim(0) * queries.dim(1);
      output.resize(queries.shape());
      Allocator& allocator = get_allocator<Device::CUDA>();
      void* workspace = allocator.allocate(cuda::exact_attention_workspace_bytes(batch, keys.dim(2)));
      cuda::exact_attention(queries.data<float16_t>(), keys.data<float16_t>(), values.data<float16_t>(),
                            workspace, output.data<float16_t>(), batch, queries.dim(2), keys.dim(2), scale);
      allocator.free(workspace);                               // stream-ordered, after the kernels
#else
      (void)queries; (void)keys; (void)values; (void)scale; (void)output;
      throw std::logic_error("attention_fused requires CUDA");
#endif
    }

  }
}
