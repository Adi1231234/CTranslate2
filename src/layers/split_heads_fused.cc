#include "split_heads_fused.h"

#include <stdexcept>
#include <vector>

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/split_heads.h"
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace layers {

    bool split_heads_fusable(const StorageView& x, const Dense& linear, const Padder* padder,
                             dim_t d_head) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const StorageView* bias = linear.bias();
      return x.device() == Device::CUDA && x.dtype() == DataType::FLOAT16 && x.rank() == 3
        && !padder && linear.can_defer_bias() && !cuda::use_stock_kernels()
        && d_head % cuda::split_heads_bias_granule == 0
        && (!bias || (bias->dtype() == DataType::FLOAT16 && cuda::split_heads_bias_aligned(bias->buffer())));
#else
      (void)x; (void)linear; (void)padder; (void)d_head;
      return false;
#endif
    }

    void split_heads_with_bias(const StorageView& proj,
                               const StorageView* bias,
                               std::initializer_list<StorageView*> outs,
                               dim_t heads,
                               dim_t beam_size) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const dim_t parts = outs.size();
      const dim_t rows = beam_size > 1 ? proj.dim(0) / beam_size : proj.dim(0);
      const dim_t time = beam_size > 1 ? beam_size : proj.dim(1);
      const dim_t d_head = proj.dim(2) / (parts * heads);
      std::vector<float16_t*> ptrs;
      for (StorageView* out : outs) {
        out->resize({rows, heads, time, d_head});
        ptrs.push_back(out->data<float16_t>());
      }
      bool aligned = parts >= 1 && parts <= 3 && cuda::split_heads_bias_aligned(proj.buffer());
      for (const float16_t* p : ptrs)
        aligned = aligned && cuda::split_heads_bias_aligned(p);
      if (!aligned)
        throw std::runtime_error("split_heads_with_bias: unexpected layout");
      const float16_t* b = bias ? bias->data<float16_t>() : nullptr;
      cuda::split_heads_bias(proj.data<float16_t>(), b, ptrs.data(), parts, rows, time, heads, d_head);
#else
      (void)proj; (void)bias; (void)outs; (void)heads; (void)beam_size;
      throw std::logic_error("split_heads_with_bias requires CUDA");
#endif
    }

  }
}
