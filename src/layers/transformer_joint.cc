#include "ctranslate2/layers/transformer.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "joint_step.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/clip_groups.h"
#endif

namespace ctranslate2 {
  namespace layers {

    // Each clip's memory keys and values per layer, as joint_attention's cross-attention reads them
    // (JointStep::memory_table): two pointers a clip, packed in int32 words.
    static StorageView memory_pointers(const std::vector<TransformerDecoder::JointPart>& parts, size_t layers,
                                       Device device) {
      std::vector<uintptr_t> table;
      for (size_t l = 0; l < layers; ++l) {
        const std::string l_str = std::to_string(l);
        for (const auto& part : parts) {
          StorageView& keys = part.state->at("memory_keys_" + l_str);
          StorageView& values = part.state->at("memory_values_" + l_str);
          const dim_t clip_bytes = keys.size() / keys.dim(0) * keys.item_size();
          for (const dim_t entry : part.memory_entries) {
            if (entry < 0 || entry >= keys.dim(0) || values.shape() != keys.shape())
              throw std::out_of_range("A joint decoding step's memory entry is out of its cache");
            table.push_back(reinterpret_cast<uintptr_t>(static_cast<char*>(keys.buffer()) + entry * clip_bytes));
            table.push_back(reinterpret_cast<uintptr_t>(static_cast<char*>(values.buffer()) + entry * clip_bytes));
          }
        }
      }
      std::vector<int32_t> words(table.size() * sizeof (uintptr_t) / sizeof (int32_t));
      std::memcpy(words.data(), table.data(), table.size() * sizeof (uintptr_t));
      return StorageView({static_cast<dim_t>(words.size())}, words).to(device);
    }

    void TransformerDecoder::decode_joint(const std::vector<JointPart>& parts, StorageView& logits) {
      PROFILE("TransformerDecoder::decode_joint");
      if (parts.empty())
        throw std::invalid_argument("A joint decoding step needs parts");
      if (!_with_encoder_attention || _sliding_window > 0 || _project_in || _project_out || _layernorm_embedding
          || _start_from_zero_embedding || _outputs_scale || _final_logit_softcapping != 0.f || _tensor_parallel
          || _use_flash_attention || _alibi)
        throw std::logic_error("A joint decoding step supports the Whisper decoder's layout only");
      const DataType dtype = output_type();
      const Device device = _device;

      JointStep joint;
      std::vector<const StorageView*> ids;
      dim_t rows = 0;
      for (const auto& part : parts) {
        JointStep::Part p;
        p.row_begin = rows;
        p.rows = part.ids->size();
        p.clips = part.memory_entries.size();
        if (p.clips == 0 || p.rows % p.clips != 0)
          throw std::invalid_argument("A joint decoding step's part has no whole beams of its clips");
        // The beam order its last update_state left for this step (defers_state_reorder).
        if (auto it = part.state->find(pending_reorder_key); it != part.state->end()) {
          p.cache_reorder = std::make_unique<StorageView>(std::move(it->second));
          part.state->erase(it);
        }
        for (size_t l = 0; l < _layers.size(); ++l) {
          const std::string l_str = std::to_string(l);
          p.self_keys.push_back(&part.state->at("self_keys_" + l_str));
          p.self_values.push_back(&part.state->at("self_values_" + l_str));
        }
        rows += p.rows;
        joint.clips += p.clips;
        ids.push_back(part.ids);
        joint.parts.push_back(std::move(p));
      }
      joint.memory_table = memory_pointers(parts, _layers.size(), device);

      StorageView all_ids(DataType::INT32, device);
      ops::Concat(0)(ids, all_ids);
      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);
      _embeddings(all_ids, layer_in);
      if (_embeddings_scale)
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        for (size_t i = 0; i < parts.size(); ++i) {
          StorageView part_in = rows_view(layer_in, joint.parts[i].row_begin, joint.parts[i].rows);
          (*_position_encoder)(part_in, parts[i].step);
        }

      // Every product as each part's own batch would run it (cuda/clip_groups.h, a group per part); the
      // attention layers take each part's own caches (joint_step.h).
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::ClipGroups groups;
      for (const auto& p : joint.parts)
        groups.clips.push_back(p.clips);
      groups.total = joint.clips;
      const cuda::ClipGroupsScope clip_groups(std::move(groups));
#endif
      const JointStepScope joint_scope(joint);
      StorageView memory(dtype, device);                     // unused: every part's memory projection is cached
      StorageView position_bias(dtype, device);
      // A layer's last residual add also makes the next layer's first pre-norm, or the output norm (NormHandoff).
      StorageView normed(dtype, device);
      StorageView next_normed(dtype, device);
      bool have_normed = false;
      for (size_t l = 0; l < _layers.size(); ++l) {
        joint.layer = l;
        const LayerNorm* next_norm = l + 1 < _layers.size() ? _layers[l + 1]->input_norm() : _output_norm.get();
        const NormHandoff next{next_norm, &next_normed};
        (*_layers[l])(layer_in, nullptr, &memory, nullptr, nullptr, nullptr, nullptr, nullptr, layer_out, nullptr,
                      nullptr, nullptr, return_normalized_attention(), &position_bias, 0, nullptr,
                      have_normed ? &normed : nullptr, next_norm ? &next : nullptr);
        layer_in = std::move(layer_out);
        have_normed = next_norm != nullptr;
        if (have_normed)
          normed = std::move(next_normed);
      }

      if (have_normed)                                       // the output norm, made by the last layer
        layer_in = std::move(normed);
      else if (_output_norm)
        (*_output_norm)(layer_in, layer_in);
      _proj(layer_in, logits);
      logits.squeeze(1);
    }

  }
}
