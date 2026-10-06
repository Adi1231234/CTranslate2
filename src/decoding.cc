#include "ctranslate2/decoding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>

#include "ctranslate2/ops/ops.h"
#include "dispatch.h"
#include "layers/capacity_cache.h"
#include "layers/joint_step.h"
#ifdef CT2_WITH_CUDA
#  include "cuda/clip_groups.h"
#  include "cuda/graph.h"
#  include "cuda/memory_slots.h"
#  include "cuda/row_random.h"
#  include "cuda/shared_memory_rows.h"
#endif
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/pinned_buffer.h"
#  define CT2_HOST_COPIES 1                             // device results copied to the host without waiting
#else
#  define CT2_HOST_COPIES 0
#endif

namespace ctranslate2 {

  static const ops::Gather gather;

  // Each row's sampling stream: its input's seed and the hypothesis as subsequence (cuda/row_random.h).
  using RowSeeds = std::vector<std::pair<uint64_t, uint64_t>>;

  static void gather_beam_flat(StorageView& data, const StorageView& indices, dim_t beam_size) {
    merge_batch_beam(data);
    gather(data, indices);
    split_batch_beam(data, beam_size);
  }

  static void update_sample_with_prefix(const size_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const std::vector<size_t>& end_ids,
                                        const std::vector<dim_t>& batch_offset,
                                        const dim_t beam_size = 1,
                                        StorageView* beam_origins = nullptr,
                                        const bool is_expanded = true) {
    const dim_t batch_size = sampled_scores.dim(0);
    for (dim_t i = 0; i < batch_size; ++i) {
      const auto& prefix = prefix_ids[batch_offset[i]];
      if (step > prefix.size())
        continue;

      const dim_t num_samples = sampled_scores.dim(1);
      for (dim_t k = 0; k < num_samples; ++k) {
        const dim_t flat_index = i * num_samples + k;
        auto& sampled_id = sampled_ids.at<int32_t>(flat_index);
        int32_t new_id = -1;
        float new_score = 0;

        // When step < prefix_length, we override the sampled ids with the prefix ids
        // and set the highest probability to the first beam.
        if (step < prefix.size()) {
          new_id = prefix[step];
          new_score = (k == 0 ? 0.f : float(-1e10));

        // When step == prefix_length (the first unconstrained decoding step),
        // only the first beam is expanded. It happens that </s> appears in the topk,
        // especially when k is large. This can produce incorrect and short predictions
        // that dominate others when no length normalization is used (see issue #277).
        // To mitigate this issue, we penalize </s> in secondary beams.
        } else if (k > 0 && is_eos(sampled_id, end_ids)) {
          new_id = 0;
          new_score = -1e10;
        }

        if (new_id >= 0) {
          sampled_id = new_id;
          TYPE_DISPATCH(sampled_scores.dtype(), sampled_scores.at<T>(flat_index) = T(new_score));
          if (beam_origins)
            beam_origins->at<int32_t>(flat_index) = (is_expanded ? i * beam_size : i);
        }
      }
    }
  }

  static inline void convert_to_original_word_ids(const layers::Decoder& decoder,
                                                  StorageView& ids) {
    if (!decoder.output_layer_is_updated())
      return;
    ctranslate2::Device device = ids.device();
    if (device != Device::CPU)
      ids = ids.to(Device::CPU);
    auto* ids_data = ids.data<int32_t>();
    for (dim_t i = 0; i < ids.size(); ++i)
      ids_data[i] = decoder.to_original_word_id(ids_data[i]);
    if (ids.device() != device)
      ids = ids.to(device);
  }

  // A decoding step's decoder work, from the second step on (the first makes the caches) as one CUDA graph where
  // cuda::graphs_enabled() (cuda/graph.h); `capturable` is false where the step also returns attention.
  template <typename Run>
  static void run_decoder_step(Device device, dim_t step, bool capturable, Run&& run) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA && step > 0 && capturable) {
      cuda::StepGraph graph(step);
      try {
        run();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "decoding step %lld under CUDA graph capture failed: %s\n",
                     static_cast<long long>(step), e.what());
        std::fflush(stderr);
        throw;
      }
      graph.launch();
      return;
    }
#else
    (void)device; (void)step; (void)capturable;
#endif
    run();
  }

  template <typename T>
  static void initialize_beam_scores(StorageView& scores,
                                     const dim_t batch_size,
                                     const dim_t beam_size) {
    const dim_t size = batch_size * beam_size;
    scores.resize({size});
    auto* data = scores.data<T>();
    for (dim_t i = 0; i < size; ++i) {
      data[i] = (i % beam_size == 0 ? T(0) : std::numeric_limits<T>::lowest());
    }
  }

  static StorageView unflatten_ids(StorageView& ids,
                                   const dim_t beam_size,
                                   const dim_t vocabulary_size,
                                   const bool is_expanded) {
    const dim_t num_ids = ids.size();
    StorageView beam_origins({num_ids}, DataType::INT32);

    auto* ids_data = ids.data<int32_t>();
    auto* origins_data = beam_origins.data<int32_t>();

    for (dim_t i = 0; i < num_ids; ++i) {
      const auto flat_id = ids_data[i];
      const auto beam_id = flat_id / vocabulary_size;
      const auto word_id = flat_id % vocabulary_size;
      const auto batch_id = i / ids.dim(-1);
      ids_data[i] = word_id;
      origins_data[i] = is_expanded ? batch_id * beam_size + beam_id : batch_id;
    }

    return beam_origins;
  }

  static void append_step_output(StorageView& history,    // [batch, beam, time, ...]
                                 StorageView step_output,  // [batch, beam, ...]
                                 const StorageView* beam_origins = nullptr) {
    step_output.expand_dims(2);  // Insert time dimension.

    if (history) {
      if (beam_origins)
        gather_beam_flat(history, *beam_origins, step_output.dim(1));
      const StorageView cur_history(std::move(history));
      ops::Concat(2)({&cur_history, &step_output}, history);
    } else {
      history = std::move(step_output);
    }
  }

  static std::vector<size_t> build_hypothesis(const StorageView& history,
                                              const dim_t batch,
                                              const dim_t beam,
                                              const dim_t start,
                                              const dim_t end) {
    const auto* ids = history.index<int32_t>({batch, beam, 0});
    return std::vector<size_t>(ids + start, ids + end);
  }

  static std::vector<std::vector<float>> build_attention(const StorageView& history,
                                                         const dim_t batch,
                                                         const dim_t beam,
                                                         const dim_t start,
                                                         const dim_t end) {
    if (!history)
      return {};

    const auto source_length = history.dim(-1);

    std::vector<std::vector<float>> attention;
    attention.reserve(end - start);
    for (dim_t t = start; t < end; ++t) {
      const auto* vector = history.index<float>({batch, beam, t, 0});
      attention.emplace_back(vector, vector + source_length);
    }
    return attention;
  }

  static std::vector<StorageView> build_logits(const StorageView& history,
                                                  const dim_t batch) {
    if (!history)
      return {};
    std::vector<StorageView> logits;
    logits.reserve(batch);
    for (dim_t t = 0; t < batch; ++t) {
      ops::Slide slide(0, t, 1);
      StorageView tmp(history.dtype(), history.device());
      slide(history, tmp);
      logits.emplace_back(std::move(tmp.squeeze(0)));
    }

    return logits;
  }

  static float compute_coverage_penalty(const std::vector<std::vector<float>>& attention,
                                        const float beta) {
    float penalty = 0;
    for (size_t column = 0; column < attention[0].size(); column++) {
      float coverage = 0;
      for (size_t row = 0; row < attention.size(); row++)
        coverage += attention[row][column];
      if (coverage > 0)
        penalty += std::log(std::min(coverage, 1.f));
    }
    return beta * penalty;
  }

  static float finalize_hypothesis_score(float score,
                                         const float length,
                                         const float length_penalty,
                                         const float coverage_penalty,
                                         const std::vector<std::vector<float>>* attention) {
    score /= std::pow(length, length_penalty);

    if (coverage_penalty != 0) {
      if (!attention)
        throw std::runtime_error("The attention weights are required to apply the coverage penalty");
      score += compute_coverage_penalty(*attention, coverage_penalty);
    }

    return score;
  }

  // Sort hypotheses from best to worst score, in the limit of max_hypotheses.
  static inline void sort_hypotheses(DecodingResult& result,
                                     size_t max_hypotheses,
                                     bool keep_scores,
                                     bool keep_attention,
                                     bool keep_logits_vocab) {
    std::vector<size_t> idx(result.hypotheses.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&result](size_t i1, size_t i2) { return result.scores[i1] > result.scores[i2]; });

    if (max_hypotheses < idx.size())
      idx.resize(max_hypotheses);

    result.hypotheses = index_vector(result.hypotheses, idx);

    if (keep_scores)
      result.scores = index_vector(result.scores, idx);
    else
      result.scores.clear();

    if (keep_attention)
      result.attention = index_vector(result.attention, idx);
    else
      result.attention.clear();

    if (keep_logits_vocab)
      result.logits_vocab = index_vector(result.logits_vocab, idx);
    else
      result.logits_vocab.clear();
  }

  static inline void finalize_result(DecodingResult& result,
                                     const size_t max_hypotheses,
                                     const float length_penalty,
                                     const float coverage_penalty,
                                     const bool keep_scores,
                                     const bool keep_attention,
                                     const bool keep_logits_vocab) {
    for (size_t i = 0; i < result.scores.size(); ++i) {
      const auto* attention = result.attention.empty() ? nullptr : &result.attention[i];
      result.scores[i] = finalize_hypothesis_score(result.scores[i],
                                                   result.hypotheses[i].size(),
                                                   length_penalty,
                                                   coverage_penalty,
                                                   attention);
    }

    sort_hypotheses(result, max_hypotheses, keep_scores, keep_attention, keep_logits_vocab);
  }

  BiasedDecoder::BiasedDecoder(const float prefix_bias_beta,
                               const std::vector<std::vector<size_t>>& prefix_ids)
    : _prefix_bias_beta(prefix_bias_beta)
    , _prefix_ids(prefix_ids)
  {
  }

  void BiasedDecoder::decode(const dim_t cur_batch_size,
                             const size_t step,
                             const std::vector<dim_t>& batch_offset,
                             const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
                             const StorageView& logits,
                             StorageView& log_probs) {
    const dim_t num_beams = logits.dim(0);
    const Device device = logits.device();
    const DataType dtype = logits.dtype();

    if (_spare_beam.dtype() != dtype || _spare_beam.device() != device) {
      _spare_beam = StorageView(device, dtype);
    }

    std::vector<StorageView> logit_beam_view_storage(num_beams, StorageView(device, dtype));
    std::vector<StorageView*> logit_beam_views(num_beams);
    std::vector<StorageView> log_prob_beam_view_storage(num_beams, StorageView(device, dtype));
    std::vector<StorageView*> log_prob_beam_views(num_beams);
    for (dim_t i = 0; i < num_beams; ++i) {
      logit_beam_views[i] = &(logit_beam_view_storage[i]);
      log_prob_beam_views[i] = &(log_prob_beam_view_storage[i]);
    }
    ops::Split(0, /*no_copy=*/true)(logits, logit_beam_views);
    log_probs.resize_as(logits);
    log_probs.reshape(logits.shape());
    ops::Split(0, /*no_copy=*/true)(log_probs, log_prob_beam_views);

    // Scalar's need to be allocated on CPUs.
    StorageView scalar_discount(1 - _prefix_bias_beta, Device::CPU);
    assert (num_beams % cur_batch_size == 0);
    const dim_t cur_beam_size = num_beams / cur_batch_size;
    for (dim_t b = 0; b < num_beams; ++b) {
      StorageView &logit_beam = *(logit_beam_views[b]);
      StorageView &log_prob_beam = *(log_prob_beam_views[b]);
      const dim_t index_batch = b / cur_beam_size;
      const dim_t index_beam = b % cur_beam_size;
      const auto& prefix = _prefix_ids[batch_offset[index_batch]];
      if (static_cast<size_t>(step) < prefix.size()
          && !beams_diverged_from_prefix[index_batch][index_beam]) {
        ops::SoftMax()(logit_beam, log_prob_beam);
        ops::Mul()(log_prob_beam,
                   scalar_discount.to(log_prob_beam.dtype()),
                   _spare_beam);
        const size_t biased_word_id = prefix[step];
        StorageView spare_scalar_view;
        TYPE_DISPATCH(
          _spare_beam.dtype(),
          spare_scalar_view = StorageView({1}, _spare_beam.data<T>() + biased_word_id, device));
        const StorageView spare_scalar_copy(spare_scalar_view);
        StorageView beta_scalar;
        TYPE_DISPATCH(
          _spare_beam.dtype(),
          // Scalar's need to be allocated on CPUs.
          beta_scalar = StorageView(static_cast<T>(_prefix_bias_beta), Device::CPU));
        ops::Add()(spare_scalar_copy, beta_scalar, spare_scalar_view);
        ops::Log()(_spare_beam, log_prob_beam);
      } else {
        ops::LogSoftMax()(logit_beam, log_prob_beam);
      }
    }
  }

  static inline std::vector<std::vector<bool>>
  get_beams_divergence_from_prefix(const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
                                   const size_t step,
                                   const StorageView& sampled_ids,
                                   const std::vector<std::vector<size_t>>& prefix_ids,
                                   const std::vector<dim_t>& batch_offset) {
    auto updated = beams_diverged_from_prefix;
    for (dim_t i = 0; i < dim_t(updated.size()); ++i) {
      for (dim_t k = 0; k < dim_t(updated[i].size()); ++k) {
        const size_t word_id = sampled_ids.at<int32_t>({i, k});
        const auto& prefix = prefix_ids[batch_offset[i]];
        updated[i][k] = (step >= prefix.size()
                         || beams_diverged_from_prefix[i][k]
                         || word_id != prefix[step]);
      }
    }
    return updated;
  }

  static inline bool
  all_beams_diverged_from_prefix(const std::vector<std::vector<bool>>& beams_diverged_from_prefix) {
    for (const auto& batch : beams_diverged_from_prefix) {
      for (const bool beam_diverged : batch) {
        if (!beam_diverged)
          return false;
      }
    }
    return true;
  }

  static inline size_t get_max_candidates(const dim_t beam_size, const float patience) {
    return std::round(float(beam_size) * patience);
  }

  static dim_t get_max_step(const dim_t max_length,
                            const bool return_prefix,
                            const std::vector<std::vector<size_t>>* prefix_ids) {
    dim_t max_step = 0;

    if (prefix_ids && !return_prefix) {
      for (const auto& ids : *prefix_ids) {
        const dim_t prefix_length = ids.size();
        max_step = std::max(max_step, prefix_length + max_length);
      }

    } else {
      max_step = max_length;
    }

    return max_step;
  }

  static inline bool is_last_step(const dim_t step,
                                  const dim_t max_length,
                                  const dim_t prefix_length,
                                  const bool return_prefix) {
    return step + 1 == max_length + (return_prefix ? 0 : prefix_length);
  }

  static void apply_min_length(const dim_t step,
                               const dim_t min_length,
                               const std::vector<size_t>& end_ids,
                               DisableTokens& disable_tokens,
                               const std::vector<dim_t>& batch_offset,
                               const bool return_prefix,
                               const std::vector<std::vector<size_t>>* prefix_ids) {
    if (prefix_ids && !return_prefix) {
      const size_t batch_size = batch_offset.size();

      for (size_t i = 0; i < batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        const dim_t prefix_length = prefix_ids->at(batch_id).size();

        if (step < prefix_length + min_length) {
          for (const size_t end_id : end_ids)
            disable_tokens.add(i, end_id);
        }
      }

    } else if (step < min_length) {
      for (const size_t end_id : end_ids)
        disable_tokens.add(end_id);
    }
  }


  BeamSearch::BeamSearch(const dim_t beam_size,
                         const float length_penalty,
                         const float coverage_penalty,
                         const float prefix_bias_beta,
                         const float patience,
                         const dim_t group_size)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
    , _prefix_bias_beta(prefix_bias_beta)
    , _max_candidates(get_max_candidates(beam_size, patience))
    , _group_size(group_size)
  {
  }

  // The loop state of one beam search between its steps (BeamSearchRun).
#if CT2_HOST_COPIES
  // The top candidates of several searches (BeamSearchRun::joint_candidates) and their copy on the host.
  struct JointCandidates {
    dim_t k = 0;
    StorageView ids{DataType::INT32};
    StorageView scores;
    cuda::PinnedBuffer host_ids;
    cuda::PinnedBuffer host_scores;
  };
#endif

  struct BeamSearchRun::Impl {
    // The search's parameters.
    dim_t beam_size;
    float length_penalty;
    float coverage_penalty;
    float prefix_bias_beta;
    size_t max_candidates;
    dim_t group_size;
    // Its arguments.
    layers::Decoder& decoder;
    layers::DecoderState& state;
    const Sampler& sampler;
    const std::vector<size_t> end_ids;
    const dim_t start_step;
    const dim_t max_length;
    const dim_t min_length;
    const bool return_scores;
    const bool return_attention;
    const bool return_logits_vocab;
    const bool return_prefix;
    const size_t num_hypotheses;
    const bool include_eos_in_hypotheses;
    const std::vector<std::shared_ptr<LogitsProcessor>> logits_processors;
    const std::vector<std::vector<size_t>>* prefix_ids;

    Device device;
    DataType dtype;
    dim_t vocabulary_size;
    dim_t batch_size;
    dim_t num_candidates;
    bool expand_after_first_step;
    bool allow_early_exit;
    StorageView topk_ids;
    StorageView topk_scores;
    std::vector<bool> top_beam_finished;
    std::vector<dim_t> batch_offset;
    std::vector<DecodingResult> results;
    std::unique_ptr<BiasedDecoder> biased_decoder;
    std::vector<std::vector<bool>> beams_diverged_from_prefix;
    bool bias_towards_prefix;
    bool use_hard_prefix;
    StorageView logits;
    StorageView alive_seq;
    StorageView alive_attention;
    StorageView attention_step;
    dim_t max_step;
    bool memory_slots = false;
    StorageView slots{DataType::INT32};
    dim_t step = 0;
    bool done = false;
    // A step between its phases (queue_processors, queue_candidates, take_candidates).
    std::unique_ptr<DisableTokens> disable_tokens;
    LogitsProcessor::Rest processors_rest;
    std::vector<StorageView> logits_vec;
    StorageView device_ids{DataType::INT32};
    StorageView device_scores;
#if CT2_HOST_COPIES
    cuda::PinnedBuffer host_ids;
    cuda::PinnedBuffer host_scores;
    std::shared_ptr<const JointCandidates> joint;            // this step's, from joint_candidates
    dim_t joint_batch = 0;                                   // this search's first batch in them
#endif

    void upload_slots() {
      std::vector<int32_t> ids(batch_offset.begin(), batch_offset.end());
      slots = StorageView({static_cast<dim_t>(ids.size())}, ids).to(device);
    }
  };

#ifdef CT2_WITH_CUDA
  // A run's clip groups (the groups' clips still decoding: each group's products as a batch of its own would run
  // them) and memory slots, for its decoder step and its own update.
  struct BeamSearchRunScopes {
    explicit BeamSearchRunScopes(const BeamSearchRun::Impl& r)
      : groups(cuda::make_clip_groups(r.batch_offset, r.group_size))
      , view{r.memory_slots ? r.slots.data<int32_t>() : nullptr, static_cast<dim_t>(r.batch_offset.size())}
    {
      if (r.memory_slots)
        slots = std::make_unique<cuda::MemorySlotsScope>(view);
    }
    const cuda::ClipGroupsScope groups;
    const cuda::MemorySlots view;
    std::unique_ptr<cuda::MemorySlotsScope> slots;
  };
#endif

  std::unique_ptr<BeamSearchRun>
  BeamSearch::start(layers::Decoder& decoder,
                    layers::DecoderState& state,
                    const Sampler& sampler,
                    const std::vector<size_t>& start_ids,
                    const std::vector<size_t>& end_ids,
                    const dim_t start_step,
                    const dim_t max_length,
                    const dim_t min_length,
                    const bool return_scores,
                    const bool return_attention,
                    const bool return_logits_vocab,
                    const bool return_prefix,
                    const size_t num_hypotheses,
                    const bool include_eos_in_hypotheses,
                    const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                    const std::vector<std::vector<size_t>>* prefix_ids) const {
    auto run = std::unique_ptr<BeamSearchRun>(new BeamSearchRun());
    run->_impl = std::unique_ptr<BeamSearchRun::Impl>(new BeamSearchRun::Impl{
        _beam_size, _length_penalty, _coverage_penalty, _prefix_bias_beta, _max_candidates, _group_size,
        decoder, state, sampler, end_ids, start_step, max_length, min_length, return_scores, return_attention,
        return_logits_vocab, return_prefix, num_hypotheses, include_eos_in_hypotheses, logits_processors,
        prefix_ids});
    BeamSearchRun::Impl& r = *run->_impl;

    r.device = decoder.device();
    r.dtype = decoder.output_type();
    r.vocabulary_size = decoder.output_size();
    r.batch_size = start_ids.size();

    // We get more candidates than the beam size so that if half the candidates are EOS,
    // we can replace finished hypotheses with active beams.
    r.num_candidates = _beam_size * 2;

    // Only the first beam is considered in the first step. As an additional optimization
    // we try to run the first step without expanding the batch size.
    r.expand_after_first_step = (r.device == Device::CPU && r.num_candidates <= r.vocabulary_size);

    // We can exit early when the first beam finishes and no penalties are used.
    r.allow_early_exit = (_length_penalty == 0 && _coverage_penalty == 0);

    r.topk_ids = StorageView({r.batch_size}, DataType::INT32);
    r.topk_scores = StorageView(r.dtype);

    r.top_beam_finished.assign(r.batch_size, false);
    r.batch_offset.resize(r.batch_size);
    r.results.resize(r.batch_size);
    for (dim_t i = 0; i < r.batch_size; ++i) {
      r.batch_offset[i] = i;
      r.topk_ids.at<int32_t>(i) = start_ids[i];
    }

    if (!r.expand_after_first_step) {
      decoder.replicate_state(state, _beam_size);
      repeat_batch(r.topk_ids, _beam_size);
      TYPE_DISPATCH(r.dtype, initialize_beam_scores<T>(r.topk_scores, r.batch_size, _beam_size));
    }

    r.bias_towards_prefix = prefix_ids && _prefix_bias_beta > 0;
    if (r.bias_towards_prefix) {
      r.biased_decoder = std::make_unique<BiasedDecoder>(_prefix_bias_beta, *prefix_ids);
      r.beams_diverged_from_prefix.resize(r.batch_size, std::vector<bool>(_beam_size, false));
    }
    r.use_hard_prefix = prefix_ids && !r.bias_towards_prefix;

    r.logits = StorageView(r.dtype, r.device);
    r.alive_seq = StorageView(r.topk_ids.dtype());

    r.max_step = get_max_step(max_length, return_prefix, r.use_hard_prefix ? prefix_ids : nullptr);

#ifdef CT2_WITH_CUDA
    // Finished inputs leave the memory keys and values where they are (cuda/memory_slots.h): the fused
    // cross-attention reads each input's at its slot, its original index, for groups (or a batch) of at most 8
    // inputs, where that kernel runs every step.
    r.memory_slots = r.device == Device::CUDA && r.dtype == DataType::FLOAT16 && cuda::memory_slots_enabled()
      && (_group_size > 0 ? _group_size <= 8 : r.batch_size <= 8);
    if (r.memory_slots)
      r.upload_slots();
#endif
    return run;
  }

  BeamSearchRun::~BeamSearchRun() = default;

  bool BeamSearchRun::next_ids(StorageView& step_ids) {
    Impl& r = *_impl;
    if (r.done || r.step >= r.max_step) {
      r.done = true;
      return false;
    }
    r.attention_step = StorageView(r.dtype, r.device);
    convert_to_original_word_ids(r.decoder, r.topk_ids);
    step_ids = r.topk_ids.to(r.device);
    return true;
  }

  dim_t BeamSearchRun::step() const {
    return _impl->step;
  }

  dim_t BeamSearchRun::decoder_step() const {
    return _impl->start_step + _impl->step;
  }

  bool BeamSearchRun::with_attention() const {
    return _impl->return_attention || _impl->coverage_penalty != 0;
  }

  StorageView* BeamSearchRun::attention_output() {
    return with_attention() ? &_impl->attention_step : nullptr;
  }

  StorageView& BeamSearchRun::logits() {
    return _impl->logits;
  }

  std::shared_ptr<void> BeamSearchRun::own_scopes() {
#ifdef CT2_WITH_CUDA
    return std::make_shared<BeamSearchRunScopes>(*_impl);
#else
    return nullptr;
#endif
  }

  const std::vector<dim_t>& BeamSearchRun::alive_inputs() const {
    return _impl->batch_offset;
  }

  bool BeamSearchRun::keeps_memory_in_place() const {
    return _impl->memory_slots;
  }

  // Waits for the work queued on the thread's stream of `device` (the host reads its results next).
  static void synchronize(Device device) {
#if CT2_HOST_COPIES
    if (device == Device::CUDA)
      cuda::synchronize_stream();
#else
    (void)device;
#endif
  }

  bool BeamSearchRun::advance() {
    if (queue_processors())
      synchronize(_impl->device);
    queue_candidates();
    synchronize(_impl->device);
    return take_candidates();
  }

  bool BeamSearchRun::queue_processors() {
    Impl& r = *_impl;
#ifdef CT2_WITH_CUDA
    const BeamSearchRunScopes scopes(r);
#endif
    r.disable_tokens = std::make_unique<DisableTokens>(r.logits);
    r.processors_rest = nullptr;

    // Prevent the generation of end_ids until the minimum length is reached.
    apply_min_length(r.step,
                     r.min_length,
                     r.end_ids,
                     *r.disable_tokens,
                     r.batch_offset,
                     r.return_prefix,
                     r.prefix_ids);

    if (!r.logits_processors.empty()) {
      if (r.alive_seq)
        merge_batch_beam(r.alive_seq);
      for (size_t i = 0; i < r.logits_processors.size(); ++i) {
        LogitsProcessor::Rest rest = r.logits_processors[i]->apply_queued(r.step, r.logits, *r.disable_tokens,
                                                                          r.alive_seq, r.batch_offset,
                                                                          r.prefix_ids);
        if (rest && i + 1 < r.logits_processors.size()) {   // the next processors see what it disables
          synchronize(r.device);
          rest(*r.disable_tokens);
        } else {
          r.processors_rest = std::move(rest);
        }
      }
      if (r.alive_seq)
        split_batch_beam(r.alive_seq, r.beam_size);
    }
    return bool(r.processors_rest);
  }

  void BeamSearchRun::queue_candidates() {
    prepare_candidates();
    own_candidates();
  }

  bool BeamSearchRun::prepare_candidates() {
    Impl& r = *_impl;
#ifdef CT2_WITH_CUDA
    const BeamSearchRunScopes scopes(r);
#endif
    const bool is_expanded = (!r.expand_after_first_step || r.step > 0);
    StorageView& logits = r.logits;
    const dim_t cur_batch_size = is_expanded ? logits.dim(0) / r.beam_size : logits.dim(0);

    if (r.processors_rest) {
      r.processors_rest(*r.disable_tokens);
      r.processors_rest = nullptr;
    }
    r.disable_tokens->apply();
    r.disable_tokens.reset();
    r.logits_vec.clear();
    if (r.return_logits_vocab) {
      if (is_expanded)
        r.logits_vec = build_logits(logits, cur_batch_size * r.beam_size);
      else
        r.logits_vec = build_logits(logits, cur_batch_size);
    }
#if CT2_HOST_COPIES
    return r.device == Device::CUDA && is_expanded && !r.bias_towards_prefix && r.topk_scores
      && r.topk_scores.size() == logits.dim(0) && dynamic_cast<const BestSampler*>(&r.sampler) != nullptr;
#else
    return false;
#endif
  }

  void BeamSearchRun::own_candidates() {
    Impl& r = *_impl;
#ifdef CT2_WITH_CUDA
    const BeamSearchRunScopes scopes(r);
#endif
    const bool is_expanded = (!r.expand_after_first_step || r.step > 0);
    StorageView& logits = r.logits;
    const dim_t cur_batch_size = is_expanded ? logits.dim(0) / r.beam_size : logits.dim(0);

    StorageView log_probs(r.dtype, r.device);
    if (r.bias_towards_prefix) {
      r.biased_decoder->decode(cur_batch_size,
                               r.step,
                               r.batch_offset,
                               r.beams_diverged_from_prefix,
                               logits,
                               log_probs);
    } else {
      ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);
    }

    // Multiply by the current beam log probs.
    if (r.topk_scores) {
      DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                               primitives<D>::add_depth_broadcast(r.topk_scores.to(r.device).data<T>(),
                                                                  log_probs.data<T>(),
                                                                  r.topk_scores.size(),
                                                                  log_probs.size()));
    }

    // Flatten the probs into a list of candidates.
    log_probs.reshape({cur_batch_size, -1});

    // TopK candidates: on the GPU, copied to the host without waiting (take_candidates reads them).
#if CT2_HOST_COPIES
    if (r.device == Device::CUDA) {
      r.device_ids = StorageView(DataType::INT32, r.device);
      r.device_scores = StorageView(log_probs.dtype(), r.device);
      r.sampler.sample_on_device(log_probs, r.device_ids, r.device_scores, r.num_candidates);
      r.host_ids.copy_from_device(r.device_ids.buffer(), r.device_ids.size() * r.device_ids.item_size());
      r.host_scores.copy_from_device(r.device_scores.buffer(),
                                     r.device_scores.size() * r.device_scores.item_size());
      return;
    }
#endif
    r.sampler(log_probs, r.topk_ids, r.topk_scores, r.num_candidates);
  }

  void BeamSearchRun::joint_candidates(const std::vector<BeamSearchRun*>& runs, StorageView& logits) {
#if CT2_HOST_COPIES
    if (runs.empty())
      return;
    const Impl& first = *runs[0]->_impl;
    const dim_t beam_size = first.beam_size, k = first.num_candidates, rows = logits.dim(0);
    // The searches' log probs: one LogSoftMax over all their rows (a row's arithmetic is its own).
    ops::LogSoftMax()(logits);
    // Their beams' scores added to their rows, uploaded together.
    StorageView scores({rows}, first.dtype);
    dim_t row = 0;
    for (BeamSearchRun* run : runs) {
      const Impl& r = *run->_impl;
      std::memcpy(static_cast<char*>(scores.buffer()) + row * scores.item_size(), r.topk_scores.buffer(),
                  r.topk_scores.size() * scores.item_size());
      row += r.topk_scores.size();
    }
    if (row != rows)
      throw std::logic_error("The joint searches' rows are not the logits' rows");
    DEVICE_AND_TYPE_DISPATCH(logits.device(), logits.dtype(),
                             primitives<D>::add_depth_broadcast(scores.to(logits.device()).data<T>(),
                                                                logits.data<T>(), rows, logits.size()));
    // Each batch's candidates over its beams' rows: one TopK over all the batches (a row's search is its own).
    auto joint = std::make_shared<JointCandidates>();
    joint->k = k;
    StorageView flat(logits.dtype(), logits.device());
    flat.shallow_copy(logits);
    flat.reshape({rows / beam_size, beam_size * logits.dim(1)});
    joint->ids = StorageView(DataType::INT32, logits.device());
    joint->scores = StorageView(logits.dtype(), logits.device());
    first.sampler.sample_on_device(flat, joint->ids, joint->scores, k);
    joint->host_ids.copy_from_device(joint->ids.buffer(), joint->ids.size() * joint->ids.item_size());
    joint->host_scores.copy_from_device(joint->scores.buffer(), joint->scores.size() * joint->scores.item_size());
    dim_t batch = 0;
    for (BeamSearchRun* run : runs) {
      Impl& r = *run->_impl;
      r.joint = joint;
      r.joint_batch = batch;
      batch += r.logits.dim(0) / beam_size;
    }
#else
    (void)runs; (void)logits;
    throw std::logic_error("Joint candidates need CUDA");
#endif
  }

  bool BeamSearchRun::take_candidates() {
    Impl& r = *_impl;
#ifdef CT2_WITH_CUDA
    const BeamSearchRunScopes scopes(r);
#endif
#if CT2_HOST_COPIES
    if (r.joint) {                                           // joint_candidates' copies, the stream synchronized
      const JointCandidates& joint = *r.joint;
      const dim_t batches = r.logits.dim(0) / r.beam_size, k = joint.k;
      r.topk_ids.resize({batches, k});
      std::memcpy(r.topk_ids.buffer(), static_cast<const int32_t*>(joint.host_ids.data()) + r.joint_batch * k,
                  batches * k * sizeof (int32_t));
      r.topk_scores.resize({batches, k});
      std::memcpy(r.topk_scores.buffer(),
                  static_cast<const char*>(joint.host_scores.data()) + r.joint_batch * k * r.topk_scores.item_size(),
                  batches * k * r.topk_scores.item_size());
      r.joint.reset();
    } else if (r.device == Device::CUDA) {                   // queue_candidates' copies, the stream synchronized
      r.topk_ids.resize(r.device_ids.shape());
      std::memcpy(r.topk_ids.buffer(), r.host_ids.data(), r.topk_ids.size() * r.topk_ids.item_size());
      r.topk_scores.resize(r.device_scores.shape());
      std::memcpy(r.topk_scores.buffer(), r.host_scores.data(), r.topk_scores.size() * r.topk_scores.item_size());
    }
#endif
    const dim_t step = r.step;
    const dim_t beam_size = r.beam_size;
    const dim_t num_candidates = r.num_candidates;
    const bool is_expanded = (!r.expand_after_first_step || step > 0);
    StorageView& logits = r.logits;
    StorageView& topk_ids = r.topk_ids;
    StorageView& topk_scores = r.topk_scores;
    StorageView& alive_seq = r.alive_seq;
    StorageView& alive_attention = r.alive_attention;
    std::vector<dim_t>& batch_offset = r.batch_offset;
    std::vector<StorageView>& logits_vec = r.logits_vec;
    const auto* prefix_ids = r.prefix_ids;

    const dim_t cur_batch_size = is_expanded ? logits.dim(0) / beam_size : logits.dim(0);

    // Unflatten the ids.
    StorageView gather_indices = unflatten_ids(topk_ids, beam_size, r.vocabulary_size, is_expanded);

    if (prefix_ids) {
      if (r.use_hard_prefix) {
        update_sample_with_prefix(step,
                                  topk_ids,
                                  topk_scores,
                                  *prefix_ids,
                                  r.end_ids,
                                  batch_offset,
                                  beam_size,
                                  &gather_indices,
                                  is_expanded);
      } else if (r.bias_towards_prefix) {
        r.beams_diverged_from_prefix = get_beams_divergence_from_prefix(r.beams_diverged_from_prefix,
                                                                        step,
                                                                        topk_ids,
                                                                        *prefix_ids,
                                                                        batch_offset);
      }
    }

    // Append last prediction.
    append_step_output(alive_seq, topk_ids, &gather_indices);

    if (r.attention_step) {
      if (!is_expanded)
        repeat_batch(r.attention_step, beam_size);
      split_batch_beam(r.attention_step, beam_size);
      append_step_output(alive_attention, r.attention_step.to_float32().to(Device::CPU));
      gather_beam_flat(alive_attention, gather_indices, num_candidates);
    }

    // Check if some hypotheses are finished.
    std::vector<int32_t> non_finished_index;
    non_finished_index.reserve(cur_batch_size);

    // Only keep the first beam_size candidates.
    StorageView active_beams({cur_batch_size * beam_size}, DataType::INT32);

    for (dim_t i = 0; i < cur_batch_size; ++i) {
      const dim_t batch_id = batch_offset[i];
      const dim_t prefix_length = r.use_hard_prefix ? prefix_ids->at(batch_id).size() : 0;
      const bool is_last_step_for_batch = is_last_step(step,
                                                       r.max_length,
                                                       prefix_length,
                                                       r.return_prefix);

      auto& result = r.results[batch_id];
      dim_t secondary_candidates_offset = beam_size;

      for (dim_t k = 0; k < beam_size; ++k) {
        const size_t last_id = topk_ids.at<int32_t>({i, k});
        dim_t next_beam_id = k;

        if ((is_eos(last_id, r.end_ids) && step >= prefix_length) || is_last_step_for_batch) {
          if (k == 0)
            r.top_beam_finished[i] = true;

          const bool ignore_last_token = is_eos(last_id, r.end_ids) && !r.include_eos_in_hypotheses;
          const dim_t start = r.return_prefix ? 0 : prefix_length;
          const dim_t end = ignore_last_token ? step : step + 1;

          // Register this hypothesis.
          result.scores.emplace_back(topk_scores.scalar_at<float>({i, k}));
          result.hypotheses.emplace_back(build_hypothesis(alive_seq, i, k, start, end));
          if (alive_attention)
            result.attention.emplace_back(build_attention(alive_attention, i, k, start, end));
          if (r.return_logits_vocab) {
            result.logits_vocab.emplace_back(std::move(logits_vec[i * k]));
          }

          // Move another active beam to this position.
          for (dim_t j = secondary_candidates_offset; j < num_candidates; ++j) {
            const auto candidate = topk_ids.at<int32_t>({i, j});
            if (!is_eos(candidate, r.end_ids)) {
              next_beam_id = j;
              secondary_candidates_offset = j + 1;
              break;
            }
          }
        }

        active_beams.at<int32_t>(i * beam_size + k) = i * num_candidates + next_beam_id;
      }

      bool is_finished = false;
      if (is_last_step_for_batch)
        is_finished = true;
      else if (r.allow_early_exit)
        is_finished = r.top_beam_finished[i] && result.hypotheses.size() >= r.num_hypotheses;
      else
        is_finished = result.hypotheses.size() >= r.max_candidates;

      if (is_finished) {
        finalize_result(result,
                        r.num_hypotheses,
                        r.length_penalty,
                        r.coverage_penalty,
                        r.return_scores,
                        r.return_attention,
                        r.return_logits_vocab);
      } else {
        non_finished_index.emplace_back(i);
      }
    }

    const dim_t next_batch_size = non_finished_index.size();

    // If all remaining sentences are finished, no need to go further.
    if (next_batch_size == 0) {
      if (!is_expanded) {
        // We should ensure that states are replicated before exiting this function.
        r.decoder.replicate_state(r.state, beam_size);
      }
      r.done = true;
      return false;
    }

    gather(gather_indices, active_beams);
    gather_beam_flat(topk_ids, active_beams, beam_size);
    gather_beam_flat(topk_scores, active_beams, beam_size);
    gather_beam_flat(alive_seq, active_beams, beam_size);
    if (alive_attention)
      gather_beam_flat(alive_attention, active_beams, beam_size);

    // If some sentences finished on this step, ignore them for the next step.
    std::unique_ptr<StorageView> keep_batches;
    if (next_batch_size != cur_batch_size) {
      batch_offset = index_vector(batch_offset, non_finished_index);
      r.top_beam_finished = index_vector(r.top_beam_finished, non_finished_index);
      if (r.bias_towards_prefix)
        r.beams_diverged_from_prefix = index_vector(r.beams_diverged_from_prefix, non_finished_index);

      keep_batches = std::make_unique<StorageView>(Shape{next_batch_size}, non_finished_index);
      gather(topk_ids, *keep_batches);
      gather(topk_scores, *keep_batches);
      gather(alive_seq, *keep_batches);
      if (alive_attention)
        gather(alive_attention, *keep_batches);
      // Left on the host: update_state then compacts the decoder state in place.
    }

    if (gather_indices.device() != r.device)
      gather_indices = gather_indices.to(r.device);
    r.decoder.update_state(r.state, gather_indices, beam_size, keep_batches.get(), r.memory_slots);
#ifdef CT2_WITH_CUDA
    if (r.memory_slots && keep_batches)
      r.upload_slots();                                      // batch_offset: the inputs still decoding
#endif

    topk_ids.reshape({next_batch_size * beam_size});
    topk_scores.reshape({next_batch_size * beam_size});

    if (r.bias_towards_prefix)
      r.bias_towards_prefix = !all_beams_diverged_from_prefix(r.beams_diverged_from_prefix);

    if (++r.step >= r.max_step) {
      r.done = true;
      return false;
    }
    return true;
  }

  std::vector<DecodingResult> BeamSearchRun::finish() {
    _impl->decoder.flush_state_reorder(_impl->state);  // callers may reuse the state
    return std::move(_impl->results);
  }

  std::vector<DecodingResult>
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const std::vector<size_t>& end_ids,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const bool return_scores,
                     const bool return_attention,
                     const bool return_logits_vocab,
                     const bool return_prefix,
                     const size_t num_hypotheses,
                     const bool include_eos_in_hypotheses,
                     const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
#ifdef CT2_WITH_CUDA
    const cuda::StepGraphScope step_graphs;           // releases the step graph and arenas at the end
#endif
    const auto run = start(decoder, state, sampler, start_ids, end_ids, start_step, max_length, min_length,
                           return_scores, return_attention, return_logits_vocab, return_prefix, num_hypotheses,
                           include_eos_in_hypotheses, logits_processors, prefix_ids);
    StorageView step_ids(DataType::INT32);
    while (run->next_ids(step_ids)) {
      {
        const auto scopes = run->own_scopes();
        run_decoder_step(decoder.device(), run->step(), !run->with_attention(), [&] {
          decoder(run->decoder_step(),
                  step_ids,
                  state,
                  &run->logits(),  // output shape: (cur_batch_size*beam_size x vocab_size), if not expanded beam_size is 1
                  run->attention_output());
        });
      }
      if (!run->advance())
        break;
    }
    return run->finish();
  }


  GreedySearch::GreedySearch(const float length_penalty,
                             const float coverage_penalty,
                             std::function<bool(DecodingStepResult)> callback,
                             const dim_t group_size,
                             std::vector<uint64_t> seeds,
                             std::vector<float> temperatures)
    : _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
    , _callback(std::move(callback))
    , _group_size(group_size)
    , _seeds(std::move(seeds))
    , _temperatures(std::move(temperatures))
  {
  }

  // The loop state of one greedy search between its steps (GreedySearchRun): search()'s, its inputs' hypotheses and
  // temperature variants expanded into rows (each row then a search of one hypothesis at its own temperature).
  struct GreedySearchRun::Impl {
    Impl(layers::Decoder& decoder_, layers::DecoderState& state_, const Sampler& sampler_)
      : decoder(decoder_), state(state_), sampler(sampler_) {
    }
    layers::Decoder& decoder;
    layers::DecoderState& state;
    const Sampler& sampler;
    std::vector<size_t> end_ids;
    dim_t start_step = 0;
    dim_t max_length = 0;
    dim_t min_length = 0;
    bool return_scores = false;                              // the rows' (with hypotheses: always)
    bool return_attention = false;
    bool return_logits_vocab = false;
    bool return_prefix = true;
    bool include_eos_in_hypotheses = true;
    float length_penalty = 0;
    float coverage_penalty = 0;
    std::function<bool(DecodingStepResult)> callback;
    std::vector<std::shared_ptr<LogitsProcessor>> logits_processors;
    std::vector<std::vector<size_t>> prefix_storage;
    const std::vector<std::vector<size_t>>* prefix_ids = nullptr;
    // The expansion, undone by finish(): `expanded` with hypotheses or variants, the caller's return_scores.
    bool expanded = false;
    size_t inputs = 0;
    size_t variants = 1;
    size_t num_hypotheses = 1;
    bool caller_return_scores = false;

    Device device;
    DataType dtype;
    bool gather_attention = false;
    dim_t max_step = 0;
    StorageView sample_from{DataType::INT32};
    StorageView logits;
    std::vector<dim_t> batch_offset;
    std::vector<DecodingResult> results;
    StorageView alive_seq{DataType::INT32};
    StorageView attention_step;
    StorageView attention_step_device;
    // Hypotheses sharing their input's memory entries (share rows an input): each row's input among those
    // entries, which hold the inputs still decoding, in order.
    dim_t share = 0;
    std::vector<dim_t> memory_inputs;
    StorageView row_input{DataType::INT32};
    dim_t group_rows = 0;
    RowSeeds row_seeds;                                      // each row's stream where seeds were given
    std::vector<float> row_temperatures;                     // each row's (variants), at its original index
#ifdef CT2_WITH_CUDA
    std::unique_ptr<cuda::RowStates> row_states;
#endif
    layers::CapacityCaches capacity;
    bool capacity_on = false;
    layers::SampledRows joint;
    dim_t step = 0;
    bool done = false;
    // A step between its phases (queue_processors, queue_candidates, take_candidates).
    std::unique_ptr<DisableTokens> disable_tokens;
    LogitsProcessor::Rest processors_rest;
    std::vector<StorageView> logits_vec;
    StorageView logits_orig;
    StorageView best_ids{DataType::INT32};
    StorageView best_probs;
    StorageView device_ids{DataType::INT32};
    StorageView device_probs;
#if CT2_HOST_COPIES
    cuda::PinnedBuffer host_ids;
    cuda::PinnedBuffer host_probs;
#endif

    void map_rows() {
      std::vector<int32_t> rows(batch_offset.size());
      for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<int32_t>(std::find(memory_inputs.begin(), memory_inputs.end(), batch_offset[i] / share)
                                       - memory_inputs.begin());
      row_input = StorageView({static_cast<dim_t>(rows.size())}, rows).to(device);
    }
  };

#ifdef CT2_WITH_CUDA
  // A run's clip groups (each group's products as a batch of its own), shared memory rows and capacity caches for
  // its own decoder step.
  struct GreedySearchRunScopes {
    explicit GreedySearchRunScopes(GreedySearchRun::Impl& r)
      : groups(cuda::make_clip_groups(r.batch_offset, r.group_rows))
      , shared_rows{r.share ? r.row_input.data<int32_t>() : nullptr, static_cast<dim_t>(r.batch_offset.size()),
                    static_cast<dim_t>(r.memory_inputs.size())}
    {
      if (r.share)
        shared = std::make_unique<cuda::SharedMemoryRowsScope>(shared_rows);
      if (r.capacity_on)
        capacity = std::make_unique<layers::CapacityCacheScope>(&r.capacity);
    }
    const cuda::ClipGroupsScope groups;
    const cuda::SharedMemoryRows shared_rows;
    std::unique_ptr<cuda::SharedMemoryRowsScope> shared;
    std::unique_ptr<layers::CapacityCacheScope> capacity;
  };
#endif

  std::unique_ptr<GreedySearchRun>
  GreedySearch::start(layers::Decoder& decoder,
                      layers::DecoderState& state,
                      const Sampler& sampler,
                      const std::vector<size_t>& start_ids,
                      const std::vector<size_t>& end_ids,
                      const dim_t start_step,
                      const dim_t max_length,
                      const dim_t min_length,
                      const bool return_scores,
                      const bool return_attention,
                      const bool return_logits_vocab,
                      const bool return_prefix,
                      const size_t num_hypotheses,
                      const bool include_eos_in_hypotheses,
                      const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                      const std::vector<std::vector<size_t>>* prefix_ids) const {
    auto run = std::unique_ptr<GreedySearchRun>(new GreedySearchRun());
    run->_impl = std::make_unique<GreedySearchRun::Impl>(decoder, state, sampler);
    GreedySearchRun::Impl& r = *run->_impl;
    r.end_ids = end_ids;
    r.start_step = start_step;
    r.max_length = max_length;
    r.min_length = min_length;
    r.return_attention = return_attention;
    r.return_logits_vocab = return_logits_vocab;
    r.return_prefix = return_prefix;
    r.include_eos_in_hypotheses = include_eos_in_hypotheses;
    r.length_penalty = _length_penalty;
    r.coverage_penalty = _coverage_penalty;
    r.logits_processors = logits_processors;
    r.device = decoder.device();
    r.dtype = decoder.output_type();
    r.inputs = start_ids.size();
    r.num_hypotheses = num_hypotheses;
    r.caller_return_scores = return_scores;

    // We can return multiple hypotheses from greedy search when random sampling is enabled.
    // In that case we replicate the batches and then merge the hypotheses in a single result.
    // Temperature variants likewise: an input's rows are its variants' hypotheses, variant-major.
    std::vector<size_t> row_start_ids = start_ids;
    const bool variants_here = !_temperatures.empty();
    r.expanded = num_hypotheses > 1 || variants_here;
    if (r.expanded) {
      r.variants = variants_here ? _temperatures.size() : 1;
      const size_t per_input = num_hypotheses * r.variants;
#ifdef CT2_WITH_CUDA
      // On CUDA the hypotheses read one copy of their input's memory keys and values (cuda/shared_memory_rows.h):
      // the entries the decoder does not replicate for beams stay one per input.
      const bool share = decoder.device() == Device::CUDA && decoder.output_type() == DataType::FLOAT16
        && cuda::shared_memory_rows_enabled();
#else
      const bool share = false;
#endif
      for (auto& [name, value] : state) {
        if (value && !(share && !decoder.replicate_state(name)))
          repeat_batch(value, per_input);
      }
      r.share = share ? static_cast<dim_t>(per_input) : 0;
      // A variant's hypotheses decode as the one group a search of their input alone would (cuda/clip_groups.h).
      r.group_rows = static_cast<dim_t>(num_hypotheses) * (variants_here ? 1 : _group_size);
      for (size_t s = 0; s < _seeds.size(); ++s)               // seed s (input i's variant v), hypothesis j: (s, j)
        for (size_t j = 0; j < num_hypotheses; ++j)
          r.row_seeds.emplace_back(_seeds[s], j);
      if (variants_here)
        for (size_t i = 0; i < r.inputs; ++i)
          for (const float temperature : _temperatures)
            r.row_temperatures.insert(r.row_temperatures.end(), num_hypotheses, temperature);
      row_start_ids = repeat_vector(start_ids, per_input);
      if (prefix_ids) {
        r.prefix_storage = repeat_vector(*prefix_ids, per_input);
        r.prefix_ids = &r.prefix_storage;
      }
      r.return_scores = num_hypotheses > 1 || return_scores;   // as a variant's search alone samples
      if (_callback)
        r.callback = [callback = _callback, num_hypotheses](DecodingStepResult result) {
          result.hypothesis_id = result.batch_id % num_hypotheses;
          result.batch_id /= num_hypotheses;
          return callback(std::move(result));
        };
    } else {
      r.group_rows = _group_size;
      for (const uint64_t seed : _seeds)
        r.row_seeds.emplace_back(seed, 0);
      r.prefix_ids = prefix_ids;
      r.return_scores = return_scores;
      r.callback = _callback;
    }

    const dim_t batch_size = row_start_ids.size();
    r.gather_attention = (return_attention || (r.return_scores && _coverage_penalty != 0));
    r.sample_from = StorageView({batch_size}, DataType::INT32);
    r.logits = StorageView(r.dtype, r.device);
    r.batch_offset.resize(batch_size);
    r.results.resize(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      r.batch_offset[i] = i;
      r.sample_from.at<int32_t>(i) = row_start_ids[i];
      r.results[i].hypotheses.resize(1);
      if (r.return_scores)
        r.results[i].scores.resize(1, 0.f);
      if (return_attention)
        r.results[i].attention.resize(1);
    }
    r.best_probs = StorageView(r.dtype);
    r.attention_step_device = StorageView(r.dtype, r.device);
    r.max_step = get_max_step(max_length, return_prefix, r.prefix_ids);

    if (r.share) {
      for (dim_t i = 0; i < batch_size / r.share; ++i)
        r.memory_inputs.push_back(i);
      r.map_rows();
    }
    // Each row's own sampling stream where seeds were given (cuda/row_random.h), at its original index.
    if (!r.row_seeds.empty() && static_cast<dim_t>(r.row_seeds.size()) != batch_size)
      throw std::invalid_argument("sampling_seeds needs one seed per input (and temperature variant)");
#ifdef CT2_WITH_CUDA
    if (!r.row_seeds.empty() && r.device == Device::CUDA)
      r.row_states = std::make_unique<cuda::RowStates>(r.row_seeds);
#endif
    // The self-attention caches in place, with room for the search's steps (layers/capacity_cache.h).
    r.capacity.steps = r.max_step;
    r.capacity_on = layers::capacity_caches_enabled() && r.device == Device::CUDA && r.dtype == DataType::FLOAT16;
    return run;
  }

  GreedySearchRun::~GreedySearchRun() = default;

  bool GreedySearchRun::next_ids(StorageView& step_ids) {
    Impl& r = *_impl;
    if (r.done || r.step >= r.max_step) {
      r.done = true;
      return false;
    }
    convert_to_original_word_ids(r.decoder, r.sample_from);
    step_ids = r.sample_from.to(r.device);
    return true;
  }

  dim_t GreedySearchRun::step() const {
    return _impl->step;
  }

  dim_t GreedySearchRun::decoder_step() const {
    return _impl->start_step + _impl->step;
  }

  bool GreedySearchRun::with_attention() const {
    return _impl->gather_attention;
  }

  StorageView* GreedySearchRun::attention_output() {
    return _impl->gather_attention ? &_impl->attention_step_device : nullptr;
  }

  StorageView& GreedySearchRun::logits() {
    return _impl->logits;
  }

  std::shared_ptr<void> GreedySearchRun::own_scopes() {
#ifdef CT2_WITH_CUDA
    return std::make_shared<GreedySearchRunScopes>(*_impl);
#else
    return nullptr;
#endif
  }

  const layers::SampledRows& GreedySearchRun::joint_rows() {
    Impl& r = *_impl;
    if (!r.share || !r.capacity_on)
      throw std::logic_error("A greedy search joins a joint step with shared memory rows and capacity caches only");
#ifdef CT2_WITH_CUDA
    r.joint.group_rows = cuda::make_clip_groups(r.batch_offset, r.group_rows).clips;
#endif
    r.joint.rows = static_cast<dim_t>(r.batch_offset.size());
    r.joint.row_input = r.row_input.data<int32_t>();
    r.joint.inputs = static_cast<dim_t>(r.memory_inputs.size());
    r.joint.capacity = &r.capacity;
    return r.joint;
  }

  dim_t GreedySearchRun::memory_inputs() const {
    return _impl->share ? static_cast<dim_t>(_impl->memory_inputs.size())
                        : static_cast<dim_t>(_impl->batch_offset.size());
  }

  dim_t GreedySearchRun::rows() const {
    return static_cast<dim_t>(_impl->batch_offset.size());
  }

  bool GreedySearchRun::advance() {
    if (queue_processors())
      synchronize(_impl->device);
    queue_candidates();
    synchronize(_impl->device);
    return take_candidates();
  }

  bool GreedySearchRun::queue_processors() {
    Impl& r = *_impl;
    r.capacity.advance();                                    // the decoder step has run
    r.disable_tokens = std::make_unique<DisableTokens>(r.logits);
    r.processors_rest = nullptr;

    // Prevent the generation of end_id until the minimum length is reached.
    apply_min_length(r.step,
                     r.min_length,
                     r.end_ids,
                     *r.disable_tokens,
                     r.batch_offset,
                     r.return_prefix,
                     r.prefix_ids);

    for (size_t i = 0; i < r.logits_processors.size(); ++i) {
      LogitsProcessor::Rest rest = r.logits_processors[i]->apply_queued(r.step, r.logits, *r.disable_tokens,
                                                                        r.alive_seq, r.batch_offset, r.prefix_ids);
      if (rest && i + 1 < r.logits_processors.size()) {     // the next processors see what it disables
        synchronize(r.device);
        rest(*r.disable_tokens);
      } else {
        r.processors_rest = std::move(rest);
      }
    }
    return bool(r.processors_rest);
  }

  void GreedySearchRun::queue_candidates() {
    Impl& r = *_impl;
    if (r.processors_rest) {
      r.processors_rest(*r.disable_tokens);
      r.processors_rest = nullptr;
    }
    r.disable_tokens->apply();
    r.disable_tokens.reset();

    r.logits_vec.clear();
    r.logits_orig = StorageView(r.dtype, r.device);
    if (r.return_logits_vocab) {
      r.logits_vec = build_logits(r.logits, r.logits.dim(0));
      r.logits_orig.copy_from(r.logits);
    }
    // Compute log probs only if required.
    StorageView log_probs(r.dtype, r.device);
    if (r.return_scores)
      ops::LogSoftMax()(r.logits);
    log_probs.shallow_copy(r.logits);

#ifdef CT2_WITH_CUDA
    StorageView state_of_row(DataType::INT32);                // the rows still sampling: their original index
    cuda::RowRandom seeded;
    std::unique_ptr<cuda::RowRandomScope> row_scope;
    if (r.row_states) {
      const dim_t rows = static_cast<dim_t>(r.batch_offset.size());
      state_of_row = StorageView({rows}, std::vector<int32_t>(r.batch_offset.begin(), r.batch_offset.end()))
        .to(r.device);
      seeded = cuda::RowRandom{r.row_states->states(), state_of_row.data<int32_t>(), rows};
      row_scope = std::make_unique<cuda::RowRandomScope>(seeded);
    }
#endif
    // The rows' 1 / temperature, converted as RandomSampler converts its own (StorageView(1 / t).to(dtype)).
    StorageView row_scale(r.dtype, r.device);
    std::unique_ptr<RowScalesScope> scales_scope;
    if (!r.row_temperatures.empty()) {
      std::vector<float> inverse;
      inverse.reserve(r.batch_offset.size());
      for (const dim_t row : r.batch_offset)
        inverse.push_back(float(1) / r.row_temperatures[row]);
      row_scale = StorageView({static_cast<dim_t>(inverse.size())}, inverse).to(r.dtype).to(r.device);
      scales_scope = std::make_unique<RowScalesScope>(&row_scale);
    }
    // The samples on the GPU, copied to the host without waiting (take_candidates reads them).
#if CT2_HOST_COPIES
    if (r.device == Device::CUDA) {
      r.device_ids = StorageView(DataType::INT32, r.device);
      r.device_probs = StorageView(r.dtype, r.device);
      r.sampler.sample_on_device(log_probs, r.device_ids, r.device_probs, 1);
      r.host_ids.copy_from_device(r.device_ids.buffer(), r.device_ids.size() * r.device_ids.item_size());
      r.host_probs.copy_from_device(r.device_probs.buffer(), r.device_probs.size() * r.device_probs.item_size());
      return;
    }
#endif
    r.sampler(log_probs, r.best_ids, r.best_probs);
  }

  bool GreedySearchRun::take_candidates() {
    Impl& r = *_impl;
#if CT2_HOST_COPIES
    if (r.device == Device::CUDA) {                          // queue_candidates' copies, the stream synchronized
      r.best_ids = StorageView(r.device_ids.shape(), DataType::INT32);
      std::memcpy(r.best_ids.buffer(), r.host_ids.data(), r.best_ids.size() * r.best_ids.item_size());
      r.best_probs = StorageView(r.device_probs.shape(), r.dtype);
      std::memcpy(r.best_probs.buffer(), r.host_probs.data(), r.best_probs.size() * r.best_probs.item_size());
    }
#endif
    const dim_t step = r.step;
    StorageView& best_ids = r.best_ids;
    StorageView& best_probs = r.best_probs;
    std::vector<dim_t>& batch_offset = r.batch_offset;
    const auto* prefix_ids = r.prefix_ids;
    if (prefix_ids)
      update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, r.end_ids, batch_offset);
    if (r.attention_step_device)
      r.attention_step.copy_from(r.attention_step_device.to_float32());

    if (!r.logits_processors.empty()) {
      if (r.alive_seq) {
        const StorageView cur_alive_seq = std::move(r.alive_seq);
        ops::Concat(-1)({&cur_alive_seq, &best_ids}, r.alive_seq);
      } else {
        r.alive_seq = best_ids;
      }
    }

    const dim_t cur_batch_size = static_cast<dim_t>(batch_offset.size());
    std::vector<int32_t> non_finished_index;
    non_finished_index.reserve(cur_batch_size);

    for (dim_t i = 0; i < cur_batch_size; ++i) {
      const size_t word_id = best_ids.at<int32_t>(i);
      const size_t batch_id = batch_offset[i];
      const dim_t prefix_length = prefix_ids ? prefix_ids->at(batch_id).size() : 0;
      const float score = best_probs.scalar_at<float>({i, 0});
      DecodingResult& result = r.results[batch_id];

      if (r.return_logits_vocab) {
        result.logits_vocab.resize(1);
        result.logits_vocab[0].emplace_back(std::move(r.logits_vec[i]));
      }

      if ((!is_eos(word_id, r.end_ids) || r.include_eos_in_hypotheses)
          && (r.return_prefix || step >= prefix_length)) {
        result.hypotheses[0].push_back(word_id);
        if (r.attention_step) {
          const auto* attn = r.attention_step.index<float>({i, 0});
          result.attention[0].emplace_back(attn, attn + r.attention_step.dim(-1));
        }
      }

      if (r.return_scores)
        result.scores[0] += score;

      bool is_finished = ((is_eos(word_id, r.end_ids) && step >= prefix_length)
                          || (is_last_step(step, r.max_length, prefix_length, r.return_prefix)));

      if (r.callback && (r.return_prefix || step >= prefix_length)) {
        DecodingStepResult step_result;
        step_result.step = step;
        step_result.batch_id = batch_id;
        step_result.token_id = word_id;
        step_result.hypothesis_id = 0;
        step_result.is_last = is_finished;
        if (r.return_scores)
          step_result.score = score;
        if (r.return_logits_vocab)
          step_result.logits = std::move(r.logits_orig);
        if (r.callback(std::move(step_result))) {
          is_finished = true;
        }
      }

      if (is_finished) {
        finalize_result(result,
                        1,
                        r.length_penalty,
                        r.coverage_penalty,
                        r.return_scores,
                        r.return_attention,
                        r.return_logits_vocab);
      } else {
        non_finished_index.emplace_back(i);
        r.sample_from.at<int32_t>(i) = word_id;
      }
    }

    const dim_t count_alive = non_finished_index.size();

    // No more sentences are alive, stop here.
    if (count_alive == 0) {
      r.done = true;
      return false;
    }

    // Remove finished sentences from the execution.
    if (count_alive != cur_batch_size) {
      batch_offset = index_vector(batch_offset, non_finished_index);

      StorageView alive({count_alive}, non_finished_index);
      if (r.alive_seq)
        gather(r.alive_seq, alive);
      gather(r.sample_from, alive);
      if (r.share) {
        // The rows' entries by the alive rows; the shared memory entries by the inputs still decoding.
        std::vector<int32_t> keep;
        std::vector<dim_t> kept;
        for (size_t j = 0; j < r.memory_inputs.size(); ++j)
          if (std::any_of(batch_offset.begin(), batch_offset.end(),
                          [&](dim_t row) { return row / r.share == r.memory_inputs[j]; })) {
            keep.push_back(static_cast<int32_t>(j));
            kept.push_back(r.memory_inputs[j]);
          }
        r.decoder.flush_state_reorder(r.state);
        const StorageView alive_rows = alive.to(r.device);
        const StorageView keep_inputs = StorageView({static_cast<dim_t>(keep.size())}, keep).to(r.device);
        for (auto& [name, value] : r.state) {
          if (r.decoder.replicate_state(name))
            gather(value, alive_rows);
          else if (kept.size() != r.memory_inputs.size())
            gather(value, keep_inputs);
        }
        r.memory_inputs = std::move(kept);
        r.map_rows();
      } else {
        r.decoder.update_state(r.state, alive.to(r.device));
      }
    }

    if (++r.step >= r.max_step) {
      r.done = true;
      return false;
    }
    return true;
  }

  std::vector<DecodingResult> GreedySearchRun::finish() {
    Impl& r = *_impl;
    if (!r.expanded)
      return std::move(r.results);
    std::vector<DecodingResult> final_results(r.inputs * r.variants);   // input i's variant v: i * variants + v
    for (size_t i = 0; i < r.results.size(); ++i) {
      auto& result = r.results[i];
      auto& final_result = final_results[i / r.num_hypotheses];

      final_result.hypotheses.emplace_back(std::move(result.hypotheses[0]));
      if (!result.scores.empty())
        final_result.scores.emplace_back(result.scores[0]);
      if (r.return_attention)
        final_result.attention.emplace_back(std::move(result.attention[0]));
      if (r.return_logits_vocab)
        final_result.logits_vocab.emplace_back(std::move(result.logits_vocab[0]));
    }
    if (r.num_hypotheses > 1)
      for (auto& result : final_results)
        sort_hypotheses(result, r.num_hypotheses, r.caller_return_scores, r.return_attention, r.return_logits_vocab);
    return final_results;
  }

  std::vector<DecodingResult>
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const std::vector<size_t>& end_ids,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const bool return_scores,
                       const bool return_attention,
                       const bool return_logits_vocab,
                       const bool return_prefix,
                       const size_t num_hypotheses,
                       const bool include_eos_in_hypotheses,
                       const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                       const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("greedy_search");
#ifdef CT2_WITH_CUDA
    const cuda::StepGraphScope step_graphs;           // releases the step graph and arenas at the end
#endif
    const auto run = start(decoder, state, sampler, start_ids, end_ids, start_step, max_length, min_length,
                           return_scores, return_attention, return_logits_vocab, return_prefix, num_hypotheses,
                           include_eos_in_hypotheses, logits_processors, prefix_ids);
    StorageView step_ids(DataType::INT32);
    while (run->next_ids(step_ids)) {
      {
        const auto scopes = run->own_scopes();
        run_decoder_step(decoder.device(), run->step(), !run->with_attention(), [&] {
          decoder(run->decoder_step(), step_ids, state, &run->logits(), run->attention_output());
        });
      }
      if (!run->advance())
        break;
    }
    return run->finish();
  }

  static layers::DecoderState get_batch_state(const layers::DecoderState& state,
                                              const int32_t batch_id) {
    const Device device = state.begin()->second.device();
    const ops::Gather gather_op;

    StorageView indices(batch_id, device);
    indices.reshape({1});

    layers::DecoderState batch_state;
    batch_state.reserve(state.size());

    for (const auto& pair : state) {
      const auto& name = pair.first;
      const auto& value = pair.second;
      StorageView batch_value(value.dtype(), device);
      if (value)
        gather_op(value, indices, batch_value);
      batch_state.emplace(name, std::move(batch_value));
    }

    return batch_state;
  }

  static std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
  split_start_tokens(const std::vector<std::vector<size_t>>& start_tokens) {
    std::vector<size_t> start_ids;
    std::vector<std::vector<size_t>> prefix_ids;
    start_ids.reserve(start_tokens.size());
    prefix_ids.reserve(start_tokens.size());
    bool only_start_token = true;

    for (const auto& tokens : start_tokens) {
      if (tokens.empty())
        throw std::invalid_argument("One input has no decoder start token");
      if (tokens.size() > 1)
        only_start_token = false;

      start_ids.emplace_back(tokens.front());
      prefix_ids.emplace_back(tokens.begin() + 1, tokens.end());
    }

    if (only_start_token)
      prefix_ids.clear();

    return std::make_pair(std::move(start_ids), std::move(prefix_ids));
  }

  static void validate_decoding_options(const DecodingOptions& options, const Device device) {
    if (options.beam_size == 0)
      throw std::invalid_argument("The beam size must be > 0");
    if (options.patience <= 0)
      throw std::invalid_argument("The patience factor must be > 0");
    if (options.num_hypotheses == 0)
      throw std::invalid_argument("The number of hypotheses must be > 0");
    if (options.num_hypotheses > get_max_candidates(options.beam_size, options.patience)
        && !options.return_alternatives
        && !(options.beam_size == 1 && options.sampling_topk != 1))
      throw std::invalid_argument("The number of hypotheses cannot be greater than "
                                  "beam_size * patience");
    if (options.min_length > options.max_length)
      throw std::invalid_argument("The minimum decoding length is greater than "
                                  "the maximum decoding length");
    if (options.max_length == 0)
      throw std::invalid_argument("The maximum decoding length must be > 0");
    if (options.repetition_penalty <= 0)
      throw std::invalid_argument("The repetition penalty must be > 0");
    if (options.prefix_bias_beta >= 1)
      throw std::invalid_argument("The beta value in biased decoding must be < 1");
    if (options.prefix_bias_beta > 0 && options.return_alternatives)
      throw std::invalid_argument("Biased decoding is not compatible with the return_alternatives "
                                  "mode");
    if (options.return_alternatives
        && (options.min_alternative_expansion_prob < 0
            || options.min_alternative_expansion_prob > 1))
      throw std::invalid_argument("The minimum alternative expansion probability must be "
                                  "between 0 and 1");
    if (options.callback && (options.beam_size != 1 || options.prefix_bias_beta > 0))
      throw std::invalid_argument("The callback function is not compatible with "
                                  "beam_size > 1 or prefix_bias_beta > 0");

    if (options.sampling_topp <= 0 || options.sampling_topp > 1)
      throw std::invalid_argument("The sampling_topp parameter must be between 0 and 1");
    if (!options.sampling_temperatures.empty()) {
      // num_hypotheses > 1: a search alone then expands its input's rows the same way (GreedySearch::search).
      if (options.beam_size != 1 || options.sampling_topk == 1 || options.num_hypotheses < 2
          || options.prefix_bias_beta > 0 || options.return_alternatives || options.callback
          || options.group_size > 1)
        throw std::invalid_argument("Temperature variants are random sampling (beam_size 1, sampling_topk != 1, "
                                    "num_hypotheses > 1) without alternatives, prefix bias, callback or groups "
                                    "of several inputs");
      for (const float temperature : options.sampling_temperatures)
        if (!(temperature > 0))
          throw std::invalid_argument("Every temperature variant must be > 0");
    }
    if (options.sampling_topp < 1
        && options.sampling_topk > static_cast<size_t>(ops::TopPMask::max_num_classes(device)))
      throw std::invalid_argument(
        "The sampling_topp parameter currently requires sampling_topk <= "
        + std::to_string(ops::TopPMask::max_num_classes(device))
        + " when running on a " + device_to_str(device) + " device");
  }

  static std::unique_ptr<const Sampler>
  make_sampler(const DecodingOptions& options) {
    if (!options.sampling_temperatures.empty())   // each row at its variant's (GreedySearch's RowScalesScope)
      return std::make_unique<RandomSampler>(options.sampling_topk, options.sampling_topp, 1.f);
    if (options.sampling_topk == 1 || options.sampling_temperature == 0.0)
      return std::make_unique<BestSampler>();
    else
      return std::make_unique<RandomSampler>(options.sampling_topk,
                                             options.sampling_topp,
                                             options.sampling_temperature);
  }

  static std::unique_ptr<const SearchStrategy>
  make_search_strategy(const DecodingOptions& options) {
    if (options.beam_size == 1 && options.prefix_bias_beta == 0)
      return std::make_unique<GreedySearch>(options.length_penalty,
                                            options.coverage_penalty,
                                            options.callback,
                                            options.group_size,
                                            options.sampling_seeds,
                                            options.sampling_temperatures);
    else
      return std::make_unique<BeamSearch>(options.beam_size,
                                          options.length_penalty,
                                          options.coverage_penalty,
                                          options.prefix_bias_beta,
                                          options.patience,
                                          options.group_size);
  }

  static std::vector<std::shared_ptr<LogitsProcessor>>
  make_logits_processors(const DecodingOptions& options) {
    std::vector<std::shared_ptr<LogitsProcessor>> processors;

    for (const auto& processor : options.logits_processors) {
      if (processor->apply_first())
        processors.emplace_back(processor);
    }

    if (options.repetition_penalty != 1)
      processors.emplace_back(std::make_shared<RepetitionPenalty>(options.repetition_penalty));

    if (options.no_repeat_ngram_size > 0)
      processors.emplace_back(std::make_shared<NoRepeatNgram>(options.no_repeat_ngram_size));

    if (!options.disable_ids.empty())
      processors.emplace_back(std::make_shared<SuppressTokens>(options.disable_ids));

    if (!options.disable_ids_begin.empty())
      processors.emplace_back(std::make_shared<SuppressTokensBegin>(options.disable_ids_begin));

    if (!options.disable_sequences.empty())
      processors.emplace_back(std::make_shared<SuppressSequences>(options.disable_sequences));

    for (const auto& processor : options.logits_processors) {
      if (!processor->apply_first())
        processors.emplace_back(processor);
    }

    return processors;
  }

  static DecodingResult
  decode_alternatives(layers::Decoder& decoder,
                      layers::DecoderState& state,
                      std::vector<size_t> start_tokens,
                      const std::vector<size_t>& end_ids,
                      const DecodingOptions& options) {
    DecodingResult result;
    result.hypotheses.resize(options.num_hypotheses);
    if (options.return_scores)
      result.scores.resize(options.num_hypotheses, 0);
    if (options.return_attention)
      result.attention.resize(options.num_hypotheses);
    if (options.return_logits_vocab)
      result.logits_vocab.resize(options.num_hypotheses);

    if (start_tokens.empty())
      throw std::invalid_argument("One input has no decoder start token");
    if (start_tokens.size() > options.max_length + 1)
      start_tokens.resize(options.max_length + 1);

    const dim_t min_length = options.min_length;
    const dim_t max_length = options.max_length;
    const dim_t prefix_length = start_tokens.size() - 1;
    dim_t start_step = options.start_step;

    if (prefix_length > 0) {
      // Initialize the decoder state with the prefix.
      const Device device = decoder.device();
      StorageView attention(decoder.output_type(), device);
      StorageView input_ids({1, prefix_length},
                            std::vector<int32_t>(start_tokens.begin(),
                                                 start_tokens.begin() + prefix_length),
                            device);

      convert_to_original_word_ids(decoder, input_ids);
      decoder(start_step,
              input_ids,
              state,
              /*logits=*/nullptr,
              options.return_attention ? &attention : nullptr);

      for (size_t i = 0; i < options.num_hypotheses; ++i) {
        result.hypotheses[i] = std::vector<size_t>(start_tokens.begin() + 1, start_tokens.end());

        if (options.return_attention) {
          if (attention.device() != Device::CPU)
            attention = attention.to_float32().to(Device::CPU);
          for (dim_t t = 0; t < prefix_length; ++t) {
            const float* vector = attention.index<float>({0, t, 0});
            result.attention[i].emplace_back(vector, vector + attention.dim(-1));
          }
        }
      }

      if (prefix_length == max_length)
        return result;

      start_step += prefix_length;
    }

    std::vector<size_t> start_ids{start_tokens.back()};

    const auto logits_processors = make_logits_processors(options);

    // Expand the next "num_hypotheses" candidate words using the beam search.
    BeamSearch beam(options.num_hypotheses);
    DecodingResult expansion_result = beam.search(decoder,
                                                  state,
                                                  BestSampler(),
                                                  start_ids,
                                                  end_ids,
                                                  start_step,
                                                  /*max_length=*/1,
                                                  /*min_length=*/1,
                                                  /*return_scores=*/true,
                                                  options.return_attention,
                                                  options.return_logits_vocab,
                                                  options.return_prefix,
                                                  options.num_hypotheses,
                                                  options.include_eos_in_hypotheses,
                                                  logits_processors)[0];

    start_ids.clear();

    for (size_t i = 0; i < options.num_hypotheses; ++i) {
      const float prob = std::exp(expansion_result.scores[i]);
      if (prob < options.min_alternative_expansion_prob)
        break;

      // Add expanded word to the result.
      result.hypotheses[i].emplace_back(expansion_result.hypotheses[i].back());
      if (options.return_attention)
        result.attention[i].emplace_back(std::move(expansion_result.attention[i].back()));
      if (options.return_scores)
        result.scores[i] = expansion_result.scores[i];
      if (options.return_logits_vocab)
        result.logits_vocab[i].emplace_back(std::move(expansion_result.logits_vocab[i].back()));

      // The next input is the words we just expanded.
      start_ids.push_back(result.hypotheses[i].back());
    }

    const size_t num_alternatives = start_ids.size();

    for (auto& [name, value] : state) {
      if (decoder.replicate_state(name)) {
        // Reduce state to the effective number of alternatives.
        if (num_alternatives < options.num_hypotheses)
          value.resize(0, num_alternatives);
      } else {
        // The beam dimension becomes the batch so we need to replicate all states.
        repeat_batch(value, num_alternatives);
      }
    }

    if (num_alternatives < options.num_hypotheses) {
      result.hypotheses.resize(num_alternatives);
      if (options.return_scores)
        result.scores.resize(num_alternatives);
      if (options.return_attention)
        result.attention.resize(num_alternatives);
    }

    start_step += 1;
    if (start_step == max_length)
      return result;

    // Continue the decoding from each alternative words independently.
    const auto search_strategy = make_search_strategy(options);
    const auto sampler = make_sampler(options);
    auto suffix_results = search_strategy->search(decoder,
                                                  state,
                                                  *sampler,
                                                  start_ids,
                                                  end_ids,
                                                  start_step,
                                                  std::max(max_length - start_step, dim_t(0)),
                                                  std::max(min_length - start_step, dim_t(0)),
                                                  options.return_scores,
                                                  options.return_attention,
                                                  options.return_logits_vocab,
                                                  options.return_prefix,
                                                  /*num_hypotheses=*/1,
                                                  options.include_eos_in_hypotheses,
                                                  logits_processors);

    // Update the result with the suffix decoding.
    for (size_t i = 0; i < suffix_results.size(); ++i) {
      auto& suffix = suffix_results[i];

      if (options.return_scores) {
        result.scores[i] += suffix.scores[0];
      }

      if (options.return_logits_vocab) {
        result.logits_vocab[i].insert(result.logits_vocab[i].end(),
                                   std::make_move_iterator(suffix.logits_vocab[0].begin()),
                                   std::make_move_iterator(suffix.logits_vocab[0].end()));
      }

      if (options.return_attention)
        result.attention[i].insert(result.attention[i].end(),
                                   std::make_move_iterator(suffix.attention[0].begin()),
                                   std::make_move_iterator(suffix.attention[0].end()));

      result.hypotheses[i].insert(result.hypotheses[i].end(),
                                  std::make_move_iterator(suffix.hypotheses[0].begin()),
                                  std::make_move_iterator(suffix.hypotheses[0].end()));
    }

    return result;
  }

  static std::vector<size_t> map_to_output_word_ids(const layers::Decoder& decoder,
                                                    const std::vector<size_t>& ids) {
    std::vector<size_t> new_ids;
    new_ids.reserve(ids.size());
    for (const size_t id : ids) {
      if (decoder.is_in_output(id))
        new_ids.push_back(decoder.to_output_word_id(id));
    }
    return new_ids;
  }

  // decode()'s checks, and its ids mapped to the output layer's.
  static void prepare_decode(const layers::Decoder& decoder,
                             std::vector<std::vector<size_t>>& start_tokens,
                             std::vector<size_t>& end_ids,
                             DecodingOptions& options) {
    validate_decoding_options(options, decoder.device());
    if (start_tokens.empty())
      throw std::invalid_argument("No decoder start tokens are set");

    if (decoder.output_layer_is_updated()) {
      end_ids = map_to_output_word_ids(decoder, end_ids);

      for (auto& ids : start_tokens)
        ids = map_to_output_word_ids(decoder, ids);
      for (auto& ids : options.disable_sequences)
        ids = map_to_output_word_ids(decoder, ids);

      options.disable_ids = map_to_output_word_ids(decoder, options.disable_ids);
      options.disable_ids_begin = map_to_output_word_ids(decoder, options.disable_ids_begin);
    }
  }

  // The results' original word ids.
  static void restore_word_ids(const layers::Decoder& decoder, std::vector<DecodingResult>& results) {
    if (decoder.output_layer_is_updated()) {
      for (auto& result : results) {
        for (auto& hypothesis : result.hypotheses) {
          for (auto& id : hypothesis)
            id = decoder.to_original_word_id(id);
        }
      }
    }
  }

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         std::vector<size_t> end_ids,
         DecodingOptions options) {
    prepare_decode(decoder, start_tokens, end_ids, options);
    const size_t batch_size = start_tokens.size();

    std::vector<DecodingResult> results;

    if (options.return_alternatives) {
      results.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        layers::DecoderState batch_state = get_batch_state(state, i);
        results.emplace_back(decode_alternatives(decoder,
                                                 batch_state,
                                                 start_tokens[i],
                                                 end_ids,
                                                 options));
      }

    } else {
      std::vector<size_t> start_ids;
      std::vector<std::vector<size_t>> prefix_ids;
      std::tie(start_ids, prefix_ids) = split_start_tokens(start_tokens);

      const auto search_strategy = make_search_strategy(options);
      const auto sampler = make_sampler(options);
      const auto logits_processors = make_logits_processors(options);
      results = search_strategy->search(decoder,
                                        state,
                                        *sampler,
                                        start_ids,
                                        end_ids,
                                        options.start_step,
                                        options.max_length,
                                        options.min_length,
                                        options.return_scores,
                                        options.return_attention,
                                        options.return_logits_vocab,
                                        options.return_prefix,
                                        options.num_hypotheses,
                                        options.include_eos_in_hypotheses,
                                        logits_processors,
                                        prefix_ids.empty() ? nullptr : &prefix_ids);
    }

    restore_word_ids(decoder, results);
    return results;
  }

  std::unique_ptr<DecodeRun> start_decode(layers::Decoder& decoder,
                                          layers::DecoderState& state,
                                          std::vector<std::vector<size_t>> start_tokens,
                                          std::vector<size_t> end_ids,
                                          DecodingOptions options) {
    prepare_decode(decoder, start_tokens, end_ids, options);
    if (options.return_alternatives)
      throw std::invalid_argument("A decoding a step at a time is a search without alternatives");

    auto run = std::unique_ptr<DecodeRun>(new DecodeRun());
    run->_decoder = &decoder;
    run->_options = std::move(options);
    run->_end_ids = std::move(end_ids);
    std::tie(run->_start_ids, run->_prefix_ids) = split_start_tokens(start_tokens);
    const DecodingOptions& o = run->_options;
    run->_sampler = make_sampler(o);
    run->_processors = make_logits_processors(o);
    const auto* prefix_ids = run->_prefix_ids.empty() ? nullptr : &run->_prefix_ids;
    if (o.beam_size == 1 && o.prefix_bias_beta == 0) {     // make_search_strategy's greedy search
      run->_greedy = std::make_unique<GreedySearch>(o.length_penalty, o.coverage_penalty, o.callback, o.group_size,
                                                    o.sampling_seeds, o.sampling_temperatures);
      run->_greedy_run = run->_greedy->start(decoder, state, *run->_sampler, run->_start_ids, run->_end_ids,
                                             o.start_step, o.max_length, o.min_length, o.return_scores,
                                             o.return_attention, o.return_logits_vocab, o.return_prefix,
                                             o.num_hypotheses, o.include_eos_in_hypotheses, run->_processors,
                                             prefix_ids);
      return run;
    }
    run->_strategy = std::make_unique<BeamSearch>(o.beam_size, o.length_penalty, o.coverage_penalty,
                                                  o.prefix_bias_beta, o.patience, o.group_size);
    run->_run = run->_strategy->start(decoder,
                                      state,
                                      *run->_sampler,
                                      run->_start_ids,
                                      run->_end_ids,
                                      o.start_step,
                                      o.max_length,
                                      o.min_length,
                                      o.return_scores,
                                      o.return_attention,
                                      o.return_logits_vocab,
                                      o.return_prefix,
                                      o.num_hypotheses,
                                      o.include_eos_in_hypotheses,
                                      run->_processors,
                                      prefix_ids);
    return run;
  }

  std::vector<DecodingResult> DecodeRun::finish() {
    std::vector<DecodingResult> results = _run ? _run->finish() : _greedy_run->finish();
    restore_word_ids(*_decoder, results);
    return results;
  }

}
