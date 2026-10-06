#include "ctranslate2/models/whisper_stream.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "layers/joint_step.h"
#include "layers/slot_cache.h"
#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    WhisperStream::WhisperStream(WhisperOptions options, WhisperStreamLimits limits)
      : _options(std::move(options))
      , _limits(limits) {
      if (_options.beam_size < 2 || _options.group_size != 0 || _limits.max_batches == 0)
        throw std::invalid_argument("A Whisper stream decodes whole batches with a beam search");
    }

    void WhisperStream::submit(uint64_t tag, StorageView encoder_output, std::vector<std::vector<size_t>> prompts) {
      std::unique_lock lock(_mutex);
      _changed.wait(lock, [&] { return _pending.size() < _limits.max_pending || _error; });
      if (_error)
        std::rethrow_exception(_error);
      if (_closed)
        throw std::logic_error("A batch submitted to a closed Whisper stream");
      _pending.push_back(Batch{tag, std::move(encoder_output), std::move(prompts)});
      ++_submitted;
      _changed.notify_all();
    }

    void WhisperStream::close() {
      const std::lock_guard lock(_mutex);
      _closed = true;
      _changed.notify_all();
    }

    bool WhisperStream::next(uint64_t& tag, std::vector<WhisperGenerationResult>& results) {
      std::unique_lock lock(_mutex);
      _changed.wait(lock, [&] { return !_done.empty() || _error || (_closed && _returned == _submitted); });
      if (!_done.empty()) {
        tag = _done.front().first;
        results = std::move(_done.front().second);
        _done.pop_front();
        ++_returned;
        return true;
      }
      if (_error)
        std::rethrow_exception(_error);
      return false;
    }

    bool WhisperStream::take(Batch& batch, bool wait) {
      std::unique_lock lock(_mutex);
      if (wait)
        _changed.wait(lock, [&] { return !_pending.empty() || _closed; });
      if (_pending.empty())
        return false;
      batch = std::move(_pending.front());
      _pending.pop_front();
      _changed.notify_all();
      return true;
    }

    void WhisperStream::finished(uint64_t tag, std::vector<WhisperGenerationResult> results) {
      const std::lock_guard lock(_mutex);
      _done.emplace_back(tag, std::move(results));
      _changed.notify_all();
    }

    void WhisperStream::failed(std::exception_ptr error) {
      const std::lock_guard lock(_mutex);
      _error = error;
      _changed.notify_all();
    }


    void WhisperReplica::decode_stream(WhisperStream& stream) {
      PROFILE("WhisperReplica::decode_stream");
#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const WhisperOptions& options = stream.options();
      const WhisperStreamLimits& limits = stream.limits();
      const dim_t beams = options.beam_size;

      // A batch decoding: generate()'s state for it and its beam search between steps.
      struct Active {
        uint64_t tag = 0;
        Prepared prepared;
        std::unique_ptr<DecodeRun> decode;
        StorageView ids{DataType::INT32};
        std::unique_ptr<layers::SlotCache> slots;            // a batch of one input's caches in slots (CT2_JOINT_SLOTS)
      };
      std::vector<std::unique_ptr<Active>> active;
      std::optional<WhisperStream::Batch> held;              // taken, waiting for room

      try {
        while (true) {
          dim_t rows = 0;
          for (const auto& a : active)
            rows += static_cast<dim_t>(a->decode->search().alive_inputs().size()) * beams;
          while (active.size() < limits.max_batches) {
            if (!held) {
              WhisperStream::Batch batch;
              if (!stream.take(batch, /*wait=*/active.empty()))
                break;
              held = std::move(batch);
            }
            const dim_t batch_rows = static_cast<dim_t>(held->prompts.size()) * beams;
            if (!active.empty() && rows + batch_rows > static_cast<dim_t>(limits.max_rows))
              break;
            if (held->prompts.empty()) {
              stream.finished(held->tag, {});
              held.reset();
              continue;
            }
            auto a = std::make_unique<Active>();
            a->tag = held->tag;
            a->prepared = prepare_generation(std::move(held->encoder_output), held->prompts, options);
            a->decode = start_decode(*_decoder, a->prepared.state, a->prepared.start_tokens, {_eot_id},
                                     a->prepared.decoding_options);
            if (layers::joint_slots() && held->prompts.size() == 1)
              a->slots = std::make_unique<layers::SlotCache>();
            held.reset();
            rows += batch_rows;
            active.push_back(std::move(a));
          }
          if (active.empty())
            break;                                           // closed, and every batch decoded

          // One decoder step for every batch, each with its own position, caches and memory.
          std::vector<layers::TransformerDecoder::JointPart> parts;
          parts.reserve(active.size());
          for (auto& a : active) {
            BeamSearchRun& run = a->decode->search();
            if (run.with_attention() || !run.next_ids(a->ids))
              throw std::logic_error("A Whisper stream's search has no plain step to decode");
            std::vector<dim_t> entries = run.alive_inputs();
            if (!run.keeps_memory_in_place())               // the memory was compacted with the inputs
              std::iota(entries.begin(), entries.end(), dim_t(0));
            parts.push_back({run.decoder_step(), &a->ids, &a->prepared.state, std::move(entries), a->slots.get()});
          }
          StorageView logits(_decoder->output_type(), _decoder->device());
          _decoder->decode_joint(parts, logits);

          // Each batch's rows of the logits, then its search's own step.
          dim_t row = 0;
          for (auto& a : active) {
            BeamSearchRun& run = a->decode->search();
            const dim_t batch_rows = a->ids.size();
            run.logits() = layers::rows_view(logits, row, batch_rows);
            row += batch_rows;
            if (!run.advance()) {
              stream.finished(a->tag, finish_generation(a->decode->finish(), a->prepared, options));
              a.reset();
            }
          }
          active.erase(std::remove(active.begin(), active.end(), nullptr), active.end());
        }
      } catch (...) {
        stream.failed(std::current_exception());
      }
    }


    std::shared_ptr<WhisperStream> Whisper::open_stream(WhisperOptions options, WhisperStreamLimits limits) {
      auto stream = std::make_shared<WhisperStream>(std::move(options), limits);
      post<int>([stream](WhisperReplica& replica) {
        replica.decode_stream(*stream);
        return 0;
      });
      return stream;
    }

  }
}
