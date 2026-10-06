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
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/pinned_buffer.h"
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

    void WhisperStream::submit_sampled(uint64_t tag, StorageView encoder_output,
                                       std::vector<std::vector<size_t>> prompts, WhisperOptions options) {
      if (options.beam_size != 1)
        throw std::invalid_argument("A sampled batch of a Whisper stream decodes with beam_size 1");
      std::unique_lock lock(_mutex);
      _changed.wait(lock, [&] { return _pending.size() < _limits.max_pending || _error; });
      if (_error)
        std::rethrow_exception(_error);
      if (_closed)
        throw std::logic_error("A batch submitted to a closed Whisper stream");
      _pending.push_back(Batch{tag, std::move(encoder_output), std::move(prompts),
                               std::make_shared<const WhisperOptions>(std::move(options))});
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
      const Device device = _decoder->device();
      const auto synchronize = [device] {                    // the searches' device results read next
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
        if (device == Device::CUDA)
          cuda::synchronize_stream();
#else
        (void)device;
#endif
      };

      // A batch decoding: generate()'s state for it and its search between steps (the stream's beam search, or a
      // sampled batch's greedy search with its own options).
      struct Active {
        uint64_t tag = 0;
        std::shared_ptr<const WhisperOptions> sampled;
        Prepared prepared;
        std::unique_ptr<DecodeRun> decode;
        StorageView ids{DataType::INT32};
        std::unique_ptr<layers::SlotCache> slots;            // a batch of one input's caches in slots (CT2_JOINT_SLOTS)
      };
      std::vector<std::unique_ptr<Active>> active;
      std::optional<WhisperStream::Batch> held;              // taken, waiting for room
      const auto rows_of = [&](const Active& a) {
        if (GreedySearchRun* greedy = a.decode->greedy())
          return greedy->rows();
        return static_cast<dim_t>(a.decode->beam()->alive_inputs().size()) * beams;
      };
      const auto batch_rows_of = [&](const WhisperStream::Batch& batch) {
        if (!batch.sampled)
          return static_cast<dim_t>(batch.prompts.size()) * beams;
        const size_t variants = std::max<size_t>(batch.sampled->sampling_temperatures.size(), 1);
        return static_cast<dim_t>(batch.prompts.size() * batch.sampled->num_hypotheses * variants);
      };

      try {
        while (true) {
          dim_t rows = 0;
          for (const auto& a : active)
            rows += rows_of(*a);
          while (active.size() < limits.max_batches) {
            if (!held) {
              WhisperStream::Batch batch;
              if (!stream.take(batch, /*wait=*/active.empty()))
                break;
              held = std::move(batch);
            }
            const dim_t batch_rows = batch_rows_of(*held);
            if (!active.empty() && rows + batch_rows > static_cast<dim_t>(limits.max_rows))
              break;
            if (held->prompts.empty()) {
              stream.finished(held->tag, {});
              held.reset();
              continue;
            }
            auto a = std::make_unique<Active>();
            a->tag = held->tag;
            a->sampled = held->sampled;
            const WhisperOptions& batch_options = a->sampled ? *a->sampled : options;
            a->prepared = prepare_generation(std::move(held->encoder_output), held->prompts, batch_options);
            a->decode = start_decode(*_decoder, a->prepared.state, a->prepared.start_tokens, {_eot_id},
                                     a->prepared.decoding_options);
            if (!a->sampled && layers::joint_slots() && held->prompts.size() == 1)
              a->slots = std::make_unique<layers::SlotCache>();
            held.reset();
            rows += batch_rows;
            active.push_back(std::move(a));
          }
          if (active.empty())
            break;                                           // closed, and every batch decoded

          // One decoder step for every batch, each with its own position, caches and memory: the beam searches'
          // parts first, then the greedy searches' (decode_joint's order), their logits' rows in that order.
          std::vector<Active*> order;
          for (const auto& a : active)
            if (a->decode->beam())
              order.push_back(a.get());
          const size_t beam_parts = order.size();
          for (const auto& a : active)
            if (a->decode->greedy())
              order.push_back(a.get());
          std::vector<layers::TransformerDecoder::JointPart> parts;
          parts.reserve(order.size());
          for (Active* a : order) {
            if (BeamSearchRun* run = a->decode->beam()) {
              if (run->with_attention() || !run->next_ids(a->ids))
                throw std::logic_error("A Whisper stream's search has no plain step to decode");
              std::vector<dim_t> entries = run->alive_inputs();
              if (!run->keeps_memory_in_place())             // the memory was compacted with the inputs
                std::iota(entries.begin(), entries.end(), dim_t(0));
              parts.push_back({run->decoder_step(), &a->ids, &a->prepared.state, std::move(entries), a->slots.get()});
            } else {
              GreedySearchRun& greedy = *a->decode->greedy();
              if (greedy.with_attention() || !greedy.next_ids(a->ids))
                throw std::logic_error("A Whisper stream's sampled search has no plain step to decode");
              std::vector<dim_t> entries(greedy.memory_inputs());   // its inputs' entries, compacted in order
              std::iota(entries.begin(), entries.end(), dim_t(0));
              parts.push_back({greedy.decoder_step(), &a->ids, &a->prepared.state, std::move(entries), nullptr,
                               &greedy.joint_rows()});
            }
          }
          StorageView logits(_decoder->output_type(), _decoder->device());
          _decoder->decode_joint(parts, logits);

          // Each batch's rows of the logits, then its search's own step in three phases over all the batches, the
          // device waited for once between phases (twice a batch with advance(); each search's work is the same).
          dim_t row = 0, beam_rows = 0;
          bool pending = false;
          for (Active* a : order) {
            const dim_t batch_rows = a->ids.size();
            StorageView view = layers::rows_view(logits, row, batch_rows);
            row += batch_rows;
            if (BeamSearchRun* run = a->decode->beam()) {
              run->logits() = std::move(view);
              pending = run->queue_processors() || pending;
              beam_rows = row;
            } else {
              GreedySearchRun& greedy = *a->decode->greedy();
              greedy.logits() = std::move(view);
              pending = greedy.queue_processors() || pending;
            }
          }
          if (pending)
            synchronize();
          std::vector<BeamSearchRun*> runs;
          bool joint = true;
          for (size_t i = 0; i < beam_parts; ++i) {
            runs.push_back(order[i]->decode->beam());
            joint = runs.back()->prepare_candidates() && joint;
          }
          if (!runs.empty()) {
            if (joint) {                                     // every beam search's candidates in one launch each
              StorageView beam_logits = layers::rows_view(logits, 0, beam_rows);
              BeamSearchRun::joint_candidates(runs, beam_logits);
            } else {
              for (BeamSearchRun* run : runs)
                run->own_candidates();
            }
          }
          for (size_t i = beam_parts; i < order.size(); ++i)
            order[i]->decode->greedy()->queue_candidates();
          synchronize();
          for (Active* a : order) {
            const bool going = a->decode->beam() ? a->decode->beam()->take_candidates()
                                                 : a->decode->greedy()->take_candidates();
            if (!going) {
              const WhisperOptions& batch_options = a->sampled ? *a->sampled : options;
              stream.finished(a->tag, finish_generation(a->decode->finish(), a->prepared, batch_options));
              for (auto& owned : active)
                if (owned.get() == a)
                  owned.reset();
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
