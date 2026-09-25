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

    constexpr dim_t fragment_keys = 1504;

    bool cross_check_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CROSS_CHECK");
      return enabled;
    }

    static bool kernel_applies(const StorageView& keys) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      return keys.device() == Device::CUDA && keys.dtype() == DataType::FLOAT16 && keys.rank() == 4
        && cuda::cross_attention_applies(keys.dim(2), keys.dim(3));
#else
      (void)keys;
      return false;
#endif
    }

    bool cross_fragments_apply(const StorageView& keys) {
      return !cross_check_enabled() && kernel_applies(keys);
    }

    static void make_fragments(const StorageView& keys, const StorageView& values, StorageView& kf,
                               StorageView& vf) {
      Shape shape = keys.shape();
      shape[2] = fragment_keys;
      kf = StorageView(shape, keys.dtype(), keys.device());
      vf = StorageView(std::move(shape), values.dtype(), values.device());
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::cross_attention_layout(keys.data<float16_t>(), values.data<float16_t>(), kf.data<float16_t>(),
                                   vf.data<float16_t>(), keys.dim(0) * keys.dim(1));
#endif
    }

    void to_cross_fragments(StorageView& keys, StorageView& values) {
      StorageView kf(keys.dtype(), keys.device()), vf(values.dtype(), values.device());
      make_fragments(keys, values, kf, vf);
      keys = std::move(kf);
      values = std::move(vf);
    }

    bool are_cross_fragments(const StorageView& keys, dim_t memory_time) {
      return keys.rank() == 4 && keys.dim(2) == fragment_keys && memory_time != fragment_keys;
    }

    void cross_attention_fused(const StorageView& queries, const StorageView& kf, const StorageView& vf,
                               float scale, StorageView& output) {
      if (queries.dim(0) != kf.dim(0) || queries.dim(1) != kf.dim(1) || queries.dim(3) != kf.dim(3))
        throw std::invalid_argument("cross_attention_fused: queries do not match the cache");
      output.resize(queries.shape());
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::cross_attention(queries.data<float16_t>(), kf.data<float16_t>(), vf.data<float16_t>(),
                            output.data<float16_t>(), queries.dim(0), queries.dim(1), queries.dim(2), scale);
#else
      (void)scale;
      throw std::logic_error("cross_attention_fused requires CUDA");
#endif
    }

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

    void cross_check(const StorageView& queries, const StorageView& keys, const StorageView& values, float scale,
                     const StorageView& reference) {
      if (!kernel_applies(keys) || keys.dim(2) == fragment_keys)
        return;
      static CheckTotals totals;
      StorageView kf(keys.dtype(), keys.device()), vf(values.dtype(), values.device());
      StorageView fused(queries.dtype(), queries.device());
      make_fragments(keys, values, kf, vf);
      cross_attention_fused(queries, kf, vf, scale, fused);
      const size_t bad = count_mismatches(fused, reference);
      const size_t n = ++totals.calls;
      totals.values += static_cast<size_t>(reference.size());
      if (bad) {
        ++totals.bad_calls;
        totals.bad_values += bad;
        std::fprintf(stderr, "cross_check: call %zu (m %lld): %zu of %lld outputs differ\n", n,
                     static_cast<long long>(queries.dim(2)), bad, static_cast<long long>(reference.size()));
      }
      if (n % 1000 == 0 || bad)
        totals.print("so far");
    }

  }
}
