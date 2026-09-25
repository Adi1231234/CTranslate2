#include "cross_attention_fused.h"

#include <cstdint>
#include <stdexcept>

#include "env.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/cross_attention.h"
#endif

namespace ctranslate2 {
  namespace layers {

    bool cross_check_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CROSS_CHECK");
      return enabled;
    }

    int cross_kernel_residue(const StorageView& queries, const StorageView& keys, const StorageView& values) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      if (queries.device() != Device::CUDA || queries.dtype() != DataType::FLOAT16
          || keys.dtype() != DataType::FLOAT16 || values.dtype() != DataType::FLOAT16 || queries.rank() != 4
          || keys.rank() != 4 || keys.shape() != values.shape() || queries.dim(0) != keys.dim(0)
          || queries.dim(1) != keys.dim(1) || queries.dim(3) != keys.dim(3))
        return -1;
      return cuda::cross_attention_residue(queries.dim(2), queries.dim(0) * queries.dim(1), keys.dim(2),
                                           keys.dim(3));
#else
      (void)queries; (void)keys; (void)values;
      return -1;
#endif
    }

    bool cross_attention_fusable(const StorageView& queries, const StorageView& keys, const StorageView& values,
                                 int& residue) {
      residue = cross_check_enabled() ? -1 : cross_kernel_residue(queries, keys, values);
      return residue >= 0;
    }

    void cross_attention_fused(const StorageView& queries, const StorageView& keys, const StorageView& values,
                               float scale, int residue, StorageView& output) {
      output.resize(queries.shape());
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::cross_attention(queries.data<float16_t>(), keys.data<float16_t>(), values.data<float16_t>(),
                            output.data<float16_t>(), queries.dim(0), queries.dim(1), queries.dim(2), scale,
                            residue);
#else
      (void)keys; (void)values; (void)scale; (void)residue;
      throw std::logic_error("cross_attention_fused requires CUDA");
#endif
    }

    bool cross_q_kernel_applies(const StorageView& x, const Dense& linear, const StorageView& keys,
                                 const StorageView& values, dim_t& m, int& residue) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const StorageView* bias = linear.bias();
      const StorageView& w = linear.weight();
      if (x.device() != Device::CUDA || x.dtype() != DataType::FLOAT16 || x.rank() != 3
          || x.dim(1) != 1 || keys.rank() != 4 || keys.shape() != values.shape()
          || keys.dtype() != DataType::FLOAT16 || x.dim(0) % keys.dim(0) != 0 || !linear.can_defer_bias()
          || !bias || bias->dtype() != DataType::FLOAT16 || reinterpret_cast<uintptr_t>(bias->buffer()) % 16 != 0
          || w.dtype() != DataType::FLOAT16 || w.rank() != 2 || w.dim(0) != keys.dim(1) * keys.dim(3)
          || w.dim(1) != x.dim(2) || !cuda::cross_attention_projects(x.dim(0), w.dim(0), w.dim(1)))
        return false;
      m = x.dim(0) / keys.dim(0);
      residue = cuda::cross_attention_residue(m, keys.dim(0) * keys.dim(1), keys.dim(2), keys.dim(3));
      return residue >= 0;
#else
      (void)x; (void)linear; (void)keys; (void)values; (void)m; (void)residue;
      return false;
#endif
    }

    bool cross_attention_q_fusable(const StorageView& x, const Dense& linear, const StorageView& keys,
                                   const StorageView& values, dim_t& m, int& residue) {
      return !cross_check_enabled() && cross_q_kernel_applies(x, linear, keys, values, m, residue);
    }

    void cross_attention_fused_q(const StorageView& x, const Dense& linear, const StorageView& keys,
                                 const StorageView& values, float scale, dim_t m, int residue, StorageView& output) {
      output.resize({keys.dim(0), keys.dim(1), m, keys.dim(3)});
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::cross_attention(nullptr, keys.data<float16_t>(), values.data<float16_t>(), output.data<float16_t>(),
                            keys.dim(0), keys.dim(1), m, scale, residue, x.data<float16_t>(),
                            linear.weight().data<float16_t>(), linear.bias()->data<float16_t>(), x.dim(2));
#else
      (void)x; (void)linear; (void)values; (void)scale; (void)residue;
      throw std::logic_error("cross_attention_fused_q requires CUDA");
#endif
    }

  }
}
