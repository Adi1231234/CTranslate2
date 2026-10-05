#include "cuda/grouped_split_gemm.h"

#include "cuda/grouped_split_gemm.cuh"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    bool grouped_split_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t n, dim_t k,
                            const std::vector<dim_t>& group_rows, cudaStream_t stream) {
      static const bool verified = cublas_verified_on(8, 9);   // the splits are this device's (grouped_split_gemm.cuh)
      if (!verified || n != 1280 || k != 5120)
        return false;
      const std::vector<int64_t> rows(group_rows.begin(), group_rows.end());
      return gsg_run(reinterpret_cast<const __half*>(a), reinterpret_cast<const __half*>(w),
                     reinterpret_cast<__half*>(c), static_cast<int>(n), static_cast<int>(k), rows, stream);
    }

  }
}
