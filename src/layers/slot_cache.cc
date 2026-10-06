#include "slot_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/primitives.h"
#include "env.h"
#include "joint_step.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/slot_cache.h"
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace layers {

    bool joint_slots() {
      static const bool on = read_bool_from_env("CT2_JOINT_SLOTS");
      return on;
    }

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    template <typename T>
    static StorageView device_table(const std::vector<T>& entries) {    // the structs as int32 words, uploaded
      std::vector<int32_t> words(entries.size() * sizeof (T) / sizeof (int32_t));
      if (!entries.empty())
        std::memcpy(words.data(), entries.data(), entries.size() * sizeof (T));
      return StorageView({static_cast<dim_t>(words.size())}, words).to(Device::CUDA);
    }

    static int32_t* map(SlotCache& s, int which) {               // 0 slot_of_row, 1 row_of_slot, 2/3 forks, 4 count
      return s.maps.data<int32_t>() + which * s.rows;
    }

    // The part's first slot step: its caches' rows (fewer at the beams' expansion) into slots 0.. of new buffers.
    static void make_slots(const JointStep::Part& part, SlotCache& s) {
      const StorageView& first = *part.self_keys[0];
      const dim_t old_rows = first.dim(0), heads = first.dim(1), time = first.dim(2), depth = first.dim(3);
      if (part.rows > 64 || old_rows > part.rows || time >= SlotCache::capacity)
        throw std::logic_error("A slot part needs at most 64 rows and room for its next position");
      s.rows = part.rows;
      s.time = time;
      for (size_t l = 0; l < part.self_keys.size(); ++l)
        for (int c = 0; c < 2; ++c) {
          const StorageView& cache = c ? *part.self_values[l] : *part.self_keys[l];
          auto& slots = c ? s.values : s.keys;
          slots.emplace_back(Shape{s.rows, heads, SlotCache::capacity, depth}, cache.dtype(), cache.device());
          if (time > 0)
            CUDA_CHECK(cudaMemcpy2DAsync(slots.back().buffer(), SlotCache::capacity * depth * cache.item_size(),
                                         cache.buffer(), time * depth * cache.item_size(),
                                         time * depth * cache.item_size(), old_rows * heads,
                                         cudaMemcpyDeviceToDevice, cuda::get_cuda_stream()));
        }
      std::vector<int32_t> maps(4 * s.rows + 1, 0);
      for (dim_t r = 0; r < s.rows; ++r)
        maps[r] = maps[s.rows + r] = static_cast<int32_t>(r);        // old row i in slot i
      s.maps = StorageView({4 * s.rows + 1}, maps).to(Device::CUDA);
    }
#endif

    void prepare_slots(JointStep& joint) {
      joint.slot_parts = 0;
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      std::vector<cuda::SlotPlan> plans;
      std::vector<cuda::SlotPermute> queries, outputs;
      std::vector<cuda::SlotAppend> appends;
      std::vector<JointStep::Part*> parts;
      for (auto& part : joint.parts) {
        if (!part.slots)
          continue;
        SlotCache& s = *part.slots;
        if (part.clips != 1 || (s.time < 0 && part.rows == part.clips)   // not yet expanded: this step as before
            || part.self_keys[0]->dtype() != DataType::FLOAT16 || part.self_keys[0]->device() != Device::CUDA) {
          part.slots = nullptr;
          continue;
        }
        const dim_t old_rows = s.time < 0 ? part.self_keys[0]->dim(0) : s.rows;
        if (s.time < 0)
          make_slots(part, s);
        else if (part.rows != s.rows || s.time + 1 >= SlotCache::capacity)
          throw std::logic_error("A slot part's rows changed, or it has no room for its next position");
        const int32_t* order = part.cache_reorder ? part.cache_reorder->data<int32_t>() : nullptr;
        plans.push_back({order, map(s, 0), map(s, 1), map(s, 2), map(s, 3), map(s, 4),
                         static_cast<int32_t>(s.rows), static_cast<int32_t>(old_rows)});
        queries.push_back({map(s, 1), static_cast<int32_t>(s.rows), static_cast<int32_t>(part.row_begin)});
        outputs.push_back({map(s, 0), static_cast<int32_t>(s.rows), static_cast<int32_t>(part.row_begin)});
        joint.slot_max_rows = std::max(joint.slot_max_rows, static_cast<int>(s.rows));
        joint.slot_max_time = std::max(joint.slot_max_time, static_cast<int>(s.time));
        parts.push_back(&part);
      }
      joint.slot_parts = static_cast<int>(parts.size());
      if (parts.empty())
        return;
      const StorageView plan_table = device_table(plans);
      cuda::slot_plan(reinterpret_cast<const cuda::SlotPlan*>(plan_table.data<int32_t>()), joint.slot_parts);
      const size_t layers = parts[0]->self_keys.size();
      for (size_t l = 0; l < layers; ++l)
        for (auto* part : parts) {
          SlotCache& s = *part->slots;
          appends.push_back({s.keys[l].buffer(), s.values[l].buffer(), map(s, 0), map(s, 2), map(s, 3), map(s, 4),
                             static_cast<int32_t>(s.rows), static_cast<int32_t>(s.time),
                             static_cast<int32_t>(part->row_begin), 0});
        }
      joint.slot_appends = device_table(appends);
      joint.slot_queries = device_table(queries);
      joint.slot_outputs = device_table(outputs);
#endif
    }

    void finish_slots(JointStep& joint) {
      for (auto& part : joint.parts)
        if (part.slots)
          ++part.slots->time;
    }

    void slot_append(const JointStep& joint, const StorageView& keys, const StorageView& values) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      if (joint.slot_parts == 0)
        return;
      const auto* table = reinterpret_cast<const cuda::SlotAppend*>(joint.slot_appends.data<int32_t>())
        + joint.layer * joint.slot_parts;
      cuda::slot_append(table, joint.slot_parts, keys.buffer(), values.buffer(), keys.dim(1), keys.dim(3),
                        SlotCache::capacity, joint.slot_max_rows, joint.slot_max_time);
#endif
    }

    void slot_queries(const JointStep& joint, const StorageView& queries, StorageView& slot_queries) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::slot_permute(reinterpret_cast<const cuda::SlotPermute*>(joint.slot_queries.data<int32_t>()),
                         joint.slot_parts, queries.buffer(), slot_queries.buffer(),
                         queries.dim(1) * queries.dim(3) * queries.item_size(), joint.slot_max_rows);
#endif
    }

    void slot_context(const JointStep& joint, const StorageView& slot_context, StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::slot_permute(reinterpret_cast<const cuda::SlotPermute*>(joint.slot_outputs.data<int32_t>()),
                         joint.slot_parts, slot_context.buffer(), context.buffer(),
                         context.dim(1) * context.dim(3) * context.item_size(), joint.slot_max_rows);
#endif
    }

    dim_t slot_time(const JointStep& joint, size_t part) {
      return joint.parts[part].slots->time + 1;
    }

    // MatMul(trans_b) of the part's queries (slot order) and its keys' slots, as ops::MatMul calls cuBLAS for
    // [rows, heads, 1, depth] x [rows, heads, t, depth]^T but with the slots' stride (capacity x depth).
    void slot_scores(const JointStep& joint, size_t p, const StorageView& slot_queries, float scale,
                     StorageView& scores) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const auto& part = joint.parts[p];
      const StorageView& keys = part.slots->keys[joint.layer];
      const dim_t heads = keys.dim(1), depth = keys.dim(3), t = slot_time(joint, p);
      const float16_t* q = slot_queries.data<float16_t>() + part.row_begin * heads * depth;
      primitives<Device::CUDA>::gemm_batch_strided(false, true, 1, t, depth, scale, q, depth, depth,
                                                   keys.data<float16_t>(), depth, SlotCache::capacity * depth,
                                                   0.f, scores.data<float16_t>(), t, t, part.rows * heads);
#endif
    }

    // MatMul of the part's probabilities and its values' slots, into its rows of the slot-order context.
    void slot_values(const JointStep& joint, size_t p, const StorageView& probs, StorageView& slot_context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const auto& part = joint.parts[p];
      const StorageView& values = part.slots->values[joint.layer];
      const dim_t heads = values.dim(1), depth = values.dim(3), t = slot_time(joint, p);
      float16_t* out = slot_context.data<float16_t>() + part.row_begin * heads * depth;
      primitives<Device::CUDA>::gemm_batch_strided(false, false, 1, depth, t, 1.f, probs.data<float16_t>(), t, t,
                                                   values.data<float16_t>(), depth, SlotCache::capacity * depth,
                                                   0.f, out, depth, depth, part.rows * heads);
#endif
    }

  }
}
