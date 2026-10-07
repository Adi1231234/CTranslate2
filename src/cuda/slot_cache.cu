#include "cuda/slot_cache.h"

#include <algorithm>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int max_slot_rows = 64;

    __global__ void slot_plan_kernel(const SlotPlan* plans) {
      const SlotPlan p = plans[blockIdx.x];
      int old_slot[max_slot_rows], next[max_slot_rows];
      bool kept[max_slot_rows];
      for (int i = 0; i < p.old_rows; ++i)
        old_slot[i] = p.slot_of_row[i];
      for (int s = 0; s < p.rows; ++s)
        kept[s] = false;
      for (int r = 0; r < p.rows; ++r) {                    // a parent's first child keeps its slot
        const int s = old_slot[p.order ? p.order[r] : r];
        next[r] = kept[s] ? -1 - s : s;
        kept[s] = true;
      }
      int free_slot = 0, forks = 0;
      for (int r = 0; r < p.rows; ++r) {                    // its other children take slots no row keeps
        if (next[r] >= 0)
          continue;
        while (kept[free_slot])
          ++free_slot;
        kept[free_slot] = true;
        p.fork_src[forks] = -1 - next[r];
        p.fork_dst[forks++] = free_slot;
        next[r] = free_slot;
      }
      for (int r = 0; r < p.rows; ++r) {
        p.slot_of_row[r] = next[r];
        p.row_of_slot[next[r]] = r;
      }
      *p.fork_count = forks;
    }

    void slot_plan(const SlotPlan* plans, int parts) {
      if (parts > 0)
        slot_plan_kernel<<<parts, 1, 0, get_cuda_stream()>>>(plans);
    }

    // A part a blockIdx.y; its work (the forks' copies of both caches, then the step's vectors) over the x threads.
    __global__ void slot_append_kernel(const SlotAppend* parts, const uint4* fresh_keys, const uint4* fresh_values,
                                       unsigned heads, unsigned head_vecs, unsigned capacity) {
      const SlotAppend a = parts[blockIdx.y];
      const size_t forks = static_cast<size_t>(*a.fork_count);
      const size_t span = size_t(a.time - a.shared) * head_vecs;     // a head's positions [shared, time)
      const size_t per_fork = size_t(heads) * span;                    // one cache of one fork
      const size_t copies = 2 * forks * per_fork;
      const size_t per_cache = size_t(a.rows) * heads * head_vecs;     // the step's vectors of one cache
      const size_t total = copies + 2 * per_cache;
      for (size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x; v < total; v += size_t(gridDim.x) * blockDim.x) {
        if (v < copies) {
          const size_t c = v / (forks * per_fork), w = v - c * forks * per_fork;
          const size_t f = w / per_fork, x = w - f * per_fork;
          const size_t h = x / span, y = size_t(a.shared) * head_vecs + (x - h * span);
          uint4* cache = static_cast<uint4*>(c ? a.values : a.keys);
          const size_t src = (size_t(a.fork_src[f]) * heads + h) * capacity * head_vecs + y;
          const size_t dst = (size_t(a.fork_dst[f]) * heads + h) * capacity * head_vecs + y;
          cache[dst] = cache[src];
        } else {
          const size_t w0 = v - copies, c = w0 / per_cache, w = w0 - c * per_cache;
          const size_t r = w / (size_t(heads) * head_vecs), x = w - r * heads * head_vecs;
          const size_t h = x / head_vecs, i = x - h * head_vecs;
          uint4* cache = static_cast<uint4*>(c ? a.values : a.keys);
          const uint4* fresh = c ? fresh_values : fresh_keys;
          cache[((size_t(a.slot_of_row[r]) * heads + h) * capacity + a.time) * head_vecs + i]
            = fresh[((size_t(a.row_begin) + r) * heads + h) * head_vecs + i];
        }
      }
    }

    void slot_append(const SlotAppend* parts, int count, const void* fresh_keys, const void* fresh_values,
                     int heads, int depth, int capacity, int max_rows, int max_time) {
      if (count == 0)
        return;
      const unsigned head_vecs = depth * 2 / 16;                         // fp16
      const size_t most = 2 * (size_t(max_rows) * heads * max_time + size_t(max_rows) * heads) * head_vecs;
      constexpr unsigned threads = 256;
      const unsigned blocks = static_cast<unsigned>(std::min<size_t>((most + threads - 1) / threads, 1024));
      slot_append_kernel<<<dim3(std::max(blocks, 1u), count), threads, 0, get_cuda_stream()>>>(
        parts, static_cast<const uint4*>(fresh_keys), static_cast<const uint4*>(fresh_values), heads, head_vecs,
        capacity);
    }

    __global__ void slot_permute_kernel(const SlotPermute* parts, const uint4* src, uint4* dst, unsigned row_vecs) {
      const SlotPermute p = parts[blockIdx.y];
      const size_t total = size_t(p.rows) * row_vecs;
      for (size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x; v < total; v += size_t(gridDim.x) * blockDim.x) {
        const size_t i = v / row_vecs, j = v - i * row_vecs;
        dst[(size_t(p.row_begin) + i) * row_vecs + j] = src[(size_t(p.row_begin) + p.map[i]) * row_vecs + j];
      }
    }

    void slot_permute(const SlotPermute* parts, int count, const void* src, void* dst, int row_bytes, int max_rows) {
      if (count == 0)
        return;
      const unsigned row_vecs = row_bytes / 16;
      constexpr unsigned threads = 256;
      const unsigned blocks = std::max(1u, std::min((unsigned(max_rows) * row_vecs + threads - 1) / threads, 64u));
      slot_permute_kernel<<<dim3(blocks, count), threads, 0, get_cuda_stream()>>>(
        parts, static_cast<const uint4*>(src), static_cast<uint4*>(dst), row_vecs);
    }

  }
}
