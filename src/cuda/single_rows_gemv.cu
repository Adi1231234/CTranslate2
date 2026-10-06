#include "cuda/single_rows_gemv.h"

#include <algorithm>

#include "cuda/single_rows_gemv.cuh"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static bool enabled() {
      static const bool on = read_bool_from_env("CT2_SINGLE_ROWS", true) && cublas_verified_on(8, 9);
      return on;
    }

    bool single_rows_gemv(const void* a, const void* w, void* c, dim_t n, dim_t k, const std::vector<dim_t>& rows) {
      const int T = single_rows_partials(n, k);
      if (T == 0 || rows.empty() || !enabled())
        return false;
      const auto* x = static_cast<const __half*>(a);
      auto* y = static_cast<__half*>(c);
      cudaStream_t stream = get_cuda_stream();
      for (size_t first = 0; first < rows.size(); first += sr_max_rows) {
        SingleRows chunk{};
        chunk.count = static_cast<int>(std::min<size_t>(sr_max_rows, rows.size() - first));
        for (int r = 0; r < sr_max_rows; ++r) {               // past the count: the first row's input, read unused
          const dim_t row = rows[first + (r < chunk.count ? r : 0)];
          chunk.x[r] = x + row * k;
          chunk.y[r] = r < chunk.count ? y + row * n : nullptr;
        }
        const int ni = static_cast<int>(n), ki = static_cast<int>(k);
        single_rows_launch(static_cast<const __half*>(w), chunk, ni, ki, T, single_rows_outputs_of(ki), stream);
      }
      return true;
    }

  }
}
