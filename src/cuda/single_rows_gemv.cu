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

    template <int T>
    static void launch(const __half* w, const SingleRows& rows, int n, int k, cudaStream_t stream) {
      constexpr int outputs = sr_threads / T;
      single_rows_gemv_kernel<T><<<(n + outputs - 1) / outputs, sr_threads, 0, stream>>>(w, rows, n, k);
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
        for (int r = 0; r < chunk.count; ++r) {
          chunk.x[r] = x + rows[first + r] * k;
          chunk.y[r] = y + rows[first + r] * n;
        }
        const auto* wh = static_cast<const __half*>(w);
        const int ni = static_cast<int>(n), ki = static_cast<int>(k);
        if (T == 32)
          launch<32>(wh, chunk, ni, ki, stream);
        else if (T == 16)
          launch<16>(wh, chunk, ni, ki, stream);
        else
          launch<8>(wh, chunk, ni, ki, stream);
      }
      return true;
    }

  }
}
