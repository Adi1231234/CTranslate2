#include "cuda/ladder_cross.h"

#include <cstdint>

#include "cuda/clip_groups.h"
#include "cuda/partial_sums.cuh"
#include "cuda/shared_memory_rows.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int lc_keys = 1500, lc_depth = 64, lc_max_rows = 32;

    struct LadderRows {
      int count;                             // rows listed
      int8_t row[lc_max_rows];               // the rows (indices into the queries and outputs)
      int8_t group[lc_max_rows];             // each one's group's rows: its cuBLAS call has group x heads entries
    };

    static bool enabled() {
      static const bool on = read_bool_from_env("CT2_LADDER_CROSS") && cublas_verified_on(8, 9);
      return on;
    }

    // Every row's group size as shared_memory_rows.cu's products split the rows (for_each_clip_group, else one call),
    // or false when a group has no recovered arithmetic (more than 5 rows) or the rows are too many.
    static bool row_groups(dim_t rows, LadderRows& all) {
      if (rows < 1 || rows > lc_max_rows)
        return false;
      bool known = true;
      all.count = static_cast<int>(rows);
      const auto set = [&](dim_t first, dim_t count) {
        known = known && count >= 1 && count <= 5;
        for (dim_t r = first; r < first + count; ++r) {
          all.row[r] = static_cast<int8_t>(r);
          all.group[r] = static_cast<int8_t>(count);
        }
      };
      if (!for_each_clip_group(rows, set))
        set(0, rows);
      return known;
    }

    static __device__ __forceinline__ float hf(__half x) {
      return __half2float(x);
    }

    static __device__ __forceinline__ unsigned pair(const __half* p) {   // p[0], p[1] (4-byte aligned)
      return *reinterpret_cast<const unsigned*>(p);
    }

    // Scores of rows whose groups are 1 or 2 rows (cuBLAS's gemv): a thread per (key, row), blockIdx.z the row.
    __global__ void lc_scores_gemv(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                   float alpha) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, r = rows.row[blockIdx.z];
      if (i >= lc_keys)
        return;
      const uint4* kv = reinterpret_cast<const uint4*>(k + (static_cast<size_t>(h) * lc_keys + i) * lc_depth);
      const uint4* qv = reinterpret_cast<const uint4*>(q + (static_cast<size_t>(r) * heads + h) * lc_depth);
      float part[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int v = 0; v < lc_depth / 8; ++v) {               // dims 8v .. 8v + 7 in order
        const uint4 a = __ldg(kv + v), b = __ldg(qv + v);
        const __half* ah = reinterpret_cast<const __half*>(&a);
        const __half* bh = reinterpret_cast<const __half*>(&b);
        #pragma unroll
        for (int u = 0; u < 8; ++u)
          part[u & 3] = fmaf(hf(ah[u]), hf(bh[u]), part[u & 3]);
      }
      part[0] += part[2];
      part[1] += part[3];
      part[0] += part[1];
      scores[(static_cast<size_t>(r) * heads + h) * lc_keys + i] = __float2half_rn(alpha * part[0]);
    }

    // Scores of rows whose groups are 3..5 rows (cuBLAS's tensor-core kernel): a warp per (16 keys, head), the keys
    // the rows of A and 8 rows' queries the columns of B, an mma.sync m16n8k16 chain over the dims.
    __global__ void lc_scores_mma(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                  float alpha) {
      const int lane = threadIdx.x, g = lane >> 2, t = lane & 3, i0 = blockIdx.x * 16, h = blockIdx.y;
      const __half* kb = k + static_cast<size_t>(h) * lc_keys * lc_depth;
      const auto key_pair = [&](int i, int d) { return i < lc_keys ? pair(kb + static_cast<size_t>(i) * lc_depth + d)
                                                                   : 0u; };
      unsigned a[4][4];
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int d0 = 16 * s;
        a[s][0] = key_pair(i0 + g, d0 + 2 * t);
        a[s][1] = key_pair(i0 + g + 8, d0 + 2 * t);
        a[s][2] = key_pair(i0 + g, d0 + 8 + 2 * t);
        a[s][3] = key_pair(i0 + g + 8, d0 + 8 + 2 * t);
      }
      for (int c0 = 0; c0 < rows.count; c0 += 8) {
        const int col = c0 + g;
        const __half* qc = col < rows.count
          ? q + (static_cast<size_t>(rows.row[col]) * heads + h) * lc_depth : nullptr;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int d0 = 16 * s;
          const unsigned b0 = qc ? pair(qc + d0 + 2 * t) : 0u, b1 = qc ? pair(qc + d0 + 8 + 2 * t) : 0u;
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                       "{%0,%1,%2,%3};"
                       : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                       : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]), "r"(b0), "r"(b1));
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {                        // C[key g (+8)][column 2t (+1)]
          const int i = i0 + g + 8 * (e / 2), c = c0 + 2 * t + e % 2;
          if (i < lc_keys && c < rows.count)
            scores[(static_cast<size_t>(rows.row[c]) * heads + h) * lc_keys + i] = __float2half_rn(alpha * acc[e]);
        }
      }
    }

    // Output: a block per (32 dims, head, row), its threads 32 dims x split_lanes lanes sharing each output's
    // partials (partial_sums.cuh). A row's group size sets its arithmetic, the same for the whole block.
    __global__ void lc_output(const __half* p, const __half* v, __half* out, LadderRows rows, int heads) {
      const int x = threadIdx.x, y = threadIdx.y, d = blockIdx.x * 32 + x, h = blockIdx.y;
      const int r = rows.row[blockIdx.z], group = rows.group[blockIdx.z];
      __shared__ float sm[32][32];
      const __half* pr = p + (static_cast<size_t>(r) * heads + h) * lc_keys;
      const __half* vd = v + static_cast<size_t>(h) * lc_keys * lc_depth + d;
      const auto pa = [&](int i) { return hf(pr[i]); };
      const auto vb = [&](int i) { return hf(vd[static_cast<size_t>(i) * lc_depth]); };
      const float sum = group == 1 ? split_partials<32, 1, true, 1>(sm, x, y, lc_keys, pa, vb)    // 32 chunks of 47
                      : group == 5 ? split_partials<4, 1, false, 1>(sm, x, y, lc_keys, pa, vb)    // key i in i % 4
                                   : split_partials<16, 1, false, 1>(sm, x, y, lc_keys, pa, vb);  // key i in i % 16
      if (y == 0)
        out[(static_cast<size_t>(r) * heads + h) * lc_depth + d] = __float2half_rn(sum);
    }

    bool ladder_cross_scores(const SharedMemoryRows& rows, const __half* q, const __half* k, __half* scores,
                             dim_t heads, dim_t keys, dim_t depth, float alpha) {
      LadderRows all{};
      if (!enabled() || rows.clips != 1 || keys != lc_keys || depth != lc_depth || !row_groups(rows.rows, all))
        return false;
      LadderRows gemv{}, mma{};
      for (int y = 0; y < all.count; ++y) {
        LadderRows& list = all.group[y] <= 2 ? gemv : mma;
        list.row[list.count] = all.row[y];
        list.group[list.count] = all.group[y];
        ++list.count;
      }
      cudaStream_t stream = get_cuda_stream();
      const int h = static_cast<int>(heads);
      if (gemv.count > 0)
        lc_scores_gemv<<<dim3((lc_keys + 127) / 128, h, gemv.count), 128, 0, stream>>>(q, k, scores, gemv, h, alpha);
      if (mma.count > 0)
        lc_scores_mma<<<dim3((lc_keys + 15) / 16, h), 32, 0, stream>>>(q, k, scores, mma, h, alpha);
      return true;
    }

    bool ladder_cross_output(const SharedMemoryRows& rows, const __half* p, const __half* v, __half* out, dim_t heads,
                             dim_t keys, dim_t depth) {
      LadderRows all{};
      if (!enabled() || rows.clips != 1 || keys != lc_keys || depth != lc_depth || !row_groups(rows.rows, all))
        return false;
      lc_output<<<dim3(lc_depth / 32, static_cast<unsigned>(heads), all.count), dim3(32, split_lanes), 0,
                  get_cuda_stream()>>>(p, v, out, all, static_cast<int>(heads));
      return true;
    }

  }
}
