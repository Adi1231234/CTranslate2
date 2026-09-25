#include "cross_attention_fused.h"

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace ctranslate2 {
  namespace layers {

    struct CheckTotals {
      const char* name;
      std::atomic<size_t> calls{0}, bad_calls{0}, bad_values{0}, values{0};
      explicit CheckTotals(const char* n) : name(n) {}
      void print(const char* what) const {
        std::fprintf(stderr, "%s %s: %zu calls, %zu with differences, %zu of %zu outputs differ\n", name, what,
                     calls.load(), bad_calls.load(), bad_values.load(), values.load());
        std::fflush(stderr);
      }
      ~CheckTotals() {
        if (calls)
          print("total");
      }
      void add(size_t bad, size_t count, dim_t m, dim_t batch, int residue) {
        const size_t n = ++calls;
        values += count;
        if (bad) {
          ++bad_calls;
          bad_values += bad;
          std::fprintf(stderr, "%s: call %zu (m %lld, batch %lld, residue %d): %zu of %zu outputs differ\n", name,
                       n, static_cast<long long>(m), static_cast<long long>(batch), residue, bad, count);
        }
        if (n % 1000 == 0 || bad)
          print("so far");
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
      const int residue = cross_kernel_residue(queries, keys, values);
      if (residue < 0)
        return;
      static CheckTotals totals("cross_check");
      StorageView fused(queries.dtype(), queries.device());
      cross_attention_fused(queries, keys, values, scale, residue, fused);
      totals.add(count_mismatches(fused, reference), static_cast<size_t>(reference.size()), queries.dim(2),
                 queries.dim(0) * queries.dim(1), residue);
    }

    void cross_check_q(const StorageView& x, const Dense& linear, const StorageView& keys, const StorageView& values,
                       float scale, const StorageView& reference) {
      dim_t m = 0;
      int residue = -1;
      if (!cross_q_kernel_applies(x, linear, keys, values, m, residue))
        return;
      static CheckTotals totals("cross_check_q");
      StorageView fused(x.dtype(), x.device());
      cross_attention_fused_q(x, linear, keys, values, scale, m, residue, fused);
      totals.add(count_mismatches(fused, reference), static_cast<size_t>(reference.size()), m,
                 keys.dim(0) * keys.dim(1), residue);
    }

  }
}
