#include "cross_attention_fused.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
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

    static int kernel_residue(const StorageView& queries, const StorageView& keys, const StorageView& values) {
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
      residue = cross_check_enabled() ? -1 : kernel_residue(queries, keys, values);
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

    struct CheckTotals {
      std::atomic<size_t> calls{0}, bad_calls{0}, bad_values{0}, values{0};
      void print(const char* what) const {
        std::fprintf(stderr, "cross_check %s: %zu calls, %zu with differences, %zu of %zu outputs differ\n", what,
                     calls.load(), bad_calls.load(), bad_values.load(), values.load());
        std::fflush(stderr);
      }
      ~CheckTotals() {
        if (calls)
          print("total");
      }
    };

    // Bits of the kernel's [b, m, h, d] against the reference's [b, h, m, d].
    static size_t count_mismatches(const StorageView& fused, const StorageView& reference) {
      const StorageView f = fused.to(Device::CPU), r = reference.to(Device::CPU);
      const auto* fb = reinterpret_cast<const uint16_t*>(f.data<float16_t>());
      const auto* rb = reinterpret_cast<const uint16_t*>(r.data<float16_t>());
      const dim_t B = r.dim(0), H = r.dim(1), M = r.dim(2), D = r.dim(3);
      size_t bad = 0;
      for (dim_t b = 0; b < B; ++b)
        for (dim_t h = 0; h < H; ++h)
          for (dim_t j = 0; j < M; ++j)
            for (dim_t d = 0; d < D; ++d)
              bad += fb[((b * M + j) * H + h) * D + d] != rb[((b * H + h) * M + j) * D + d];
      return bad;
    }

    void cross_check(const StorageView& queries, const StorageView& keys, const StorageView& values, float scale,
                     const StorageView& reference) {
      const int residue = kernel_residue(queries, keys, values);
      if (residue < 0)
        return;
      static CheckTotals totals;
      StorageView fused(queries.dtype(), queries.device());
      cross_attention_fused(queries, keys, values, scale, residue, fused);
      const size_t bad = count_mismatches(fused, reference);
      const size_t n = ++totals.calls;
      totals.values += static_cast<size_t>(reference.size());
      if (bad) {
        ++totals.bad_calls;
        totals.bad_values += bad;
        std::fprintf(stderr, "cross_check: call %zu (m %lld, batch %lld, residue %d): %zu of %lld outputs differ\n",
                     n, static_cast<long long>(queries.dim(2)), static_cast<long long>(queries.dim(0) * queries.dim(1)),
                     residue, bad, static_cast<long long>(reference.size()));
      }
      if (n % 1000 == 0 || bad)
        totals.print("so far");
    }

  }
}
