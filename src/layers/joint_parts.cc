#include "joint_parts.h"

#include <algorithm>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "kv_cache.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/cache_reorder.h"
#  include "cuda/softmax_parts.h"
#endif

namespace ctranslate2 {
  namespace layers {

    // Parts [first, last) by reorder_and_append or Concat, one by one.
    static void append_each(const JointStep& joint, StorageView& keys, StorageView& values, size_t first,
                            size_t last) {
      for (size_t p = first; p < last; ++p) {
        const auto& part = joint.parts[p];
        StorageView& cached_keys = *part.self_keys[joint.layer];
        StorageView& cached_values = *part.self_values[joint.layer];
        StorageView part_keys = rows_view(keys, part.row_begin, part.rows);
        StorageView part_values = rows_view(values, part.row_begin, part.rows);
        if (part.cache_reorder) {
          reorder_and_append(cached_keys, cached_values, *part.cache_reorder, part_keys, part_values);
        } else {
          StorageView tmp(keys.dtype(), keys.device());
          tmp = std::move(cached_keys);
          ops::Concat(2)({&tmp, &part_keys}, cached_keys);
          tmp = std::move(cached_values);
          ops::Concat(2)({&tmp, &part_values}, cached_values);
        }
      }
    }

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    // Parts [first, last) in one reorder_append_parts launch, or false where it does not apply.
    static bool append_fused(const JointStep& joint, StorageView& keys, StorageView& values, size_t first,
                             size_t last) {
      const dim_t heads = keys.dim(1), depth = keys.dim(3);
      bool fused = keys.device() == Device::CUDA && keys.dtype() == DataType::FLOAT16 && keys.dim(2) == 1
        && cuda::cache_reorder_supported(keys.buffer(), depth, keys.item_size())
        && cuda::cache_reorder_supported(values.buffer(), depth, values.item_size());
      for (size_t p = first; p < last && fused; ++p) {
        const auto& part = joint.parts[p];
        const StorageView& cache = *part.self_keys[joint.layer];
        fused = cache.rank() == 4 && cache.dim(1) == heads && cache.dim(3) == depth
          && cache.dim(0) == (part.cache_reorder ? cache.dim(0) : part.rows)
          && (!part.cache_reorder || (part.cache_reorder->device() == Device::CUDA
                                      && part.cache_reorder->size() == part.rows))
          && (part.row_begin * heads * depth * keys.item_size()) % 16 == 0;
      }
      if (!fused)
        return false;
      cuda::CacheParts parts;
      std::vector<StorageView> out;                          // each part's new keys and values
      out.reserve(2 * (last - first));
      for (size_t i = first; i < last; ++i) {
        const auto& part = joint.parts[i];
        StorageView& cache_keys = *part.self_keys[joint.layer];
        StorageView& cache_values = *part.self_values[joint.layer];
        const dim_t time = cache_keys.dim(2);
        out.emplace_back(Shape{part.rows, heads, time + 1, depth}, keys.dtype(), keys.device());
        out.emplace_back(Shape{part.rows, heads, time + 1, depth}, keys.dtype(), keys.device());
        const int p = parts.count++;
        parts.cache[p][0] = cache_keys.buffer();
        parts.cache[p][1] = cache_values.buffer();
        parts.fresh[p][0] = keys.data<float16_t>() + part.row_begin * heads * depth;
        parts.fresh[p][1] = values.data<float16_t>() + part.row_begin * heads * depth;
        parts.out[p][0] = out[2 * p].buffer();
        parts.out[p][1] = out[2 * p + 1].buffer();
        parts.order[p] = part.cache_reorder ? part.cache_reorder->data<int32_t>() : nullptr;
        parts.rows[p] = static_cast<int>(part.rows);
        parts.time[p] = static_cast<int>(time);
      }
      cuda::reorder_append_parts(parts, heads, depth);
      for (size_t p = first; p < last; ++p) {
        *joint.parts[p].self_keys[joint.layer] = std::move(out[2 * (p - first)]);
        *joint.parts[p].self_values[joint.layer] = std::move(out[2 * (p - first) + 1]);
      }
      return true;
    }

    // Parts [first, last)'s scores in one softmax_parts launch, or false where it does not apply.
    static bool softmax_fused(std::vector<StorageView>& scores, size_t first, size_t last) {
      cuda::SoftmaxParts parts;
      unsigned rows = 0;
      for (size_t i = first; i < last; ++i) {
        StorageView& s = scores[i];
        if (s.device() != Device::CUDA || s.dtype() != DataType::FLOAT16 || !cuda::softmax_parts_supported(s.dim(-1)))
          return false;
        rows += static_cast<unsigned>(s.size() / s.dim(-1));
        parts.data[parts.count] = s.buffer();
        parts.rows_end[parts.count] = rows;
        parts.cols[parts.count++] = static_cast<unsigned>(s.dim(-1));
      }
      cuda::softmax_parts(parts);
      return true;
    }
#endif

    void append_parts(const JointStep& joint, StorageView& keys, StorageView& values) {
      for (const auto& part : joint.parts)
        if (part.self_keys[joint.layer]->empty())
          throw std::logic_error("A joint decoding step needs every part's self-attention cache");
      for (size_t first = 0; first < joint.parts.size();) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
        const size_t last = std::min(joint.parts.size(), first + cuda::CacheParts::max_parts);
        if (!append_fused(joint, keys, values, first, last))
#else
        const size_t last = joint.parts.size();
#endif
          append_each(joint, keys, values, first, last);
        first = last;
      }
    }

    void softmax_parts(std::vector<StorageView>& scores) {
      for (size_t first = 0; first < scores.size();) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
        const size_t last = std::min(scores.size(), first + cuda::SoftmaxParts::max_parts);
        if (!softmax_fused(scores, first, last))
#else
        const size_t last = scores.size();
#endif
          for (size_t i = first; i < last; ++i)
            ops::SoftMax()(scores[i], nullptr, scores[i]);
        first = last;
      }
    }

  }
}
