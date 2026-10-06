#pragma once

#include <functional>
#include <optional>

#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  struct DecodingResult {
    std::vector<std::vector<size_t>> hypotheses;
    std::vector<float> scores;
    std::vector<std::vector<std::vector<float>>> attention;
    std::vector<std::vector<StorageView>> logits_vocab;
  };

  struct DecodingStepResult {
    size_t step;
    size_t batch_id;
    size_t token_id;
    size_t hypothesis_id;
    std::optional<float> score;
    std::optional<StorageView> logits;
    bool is_last = false;
  };


  class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;
    virtual std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_ids,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const = 0;
  };

  class BeamSearchRun;

  class BeamSearch : public SearchStrategy {
  public:
    BeamSearch(const dim_t beam_size,
               const float length_penalty = 0,
               const float coverage_penalty = 0,
               const float prefix_bias_beta = 0,
               const float patience = 1,
               const dim_t group_size = 0);

    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_ids,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

    // The same search a step at a time, the decoder step being the caller's (search() runs it to the end), so that
    // several searches can share one decoder call (WhisperReplica::decode_stream). The run keeps references to
    // its arguments, but for end_ids and logits_processors, which it copies.
    std::unique_ptr<BeamSearchRun>
    start(layers::Decoder& decoder,
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
          const std::vector<std::vector<size_t>>* prefix_ids) const;

  private:
    const dim_t _beam_size;
    const float _length_penalty;
    const float _coverage_penalty;
    const float _prefix_bias_beta;
    const size_t _max_candidates;
    const dim_t _group_size;
  };

  // A beam search between steps (BeamSearch::start). Each step: next_ids() gives the ids to decode (false: the
  // search is over), the caller runs the decoder on them (decoder_step(), attention_output()) into logits(), then
  // advance() takes the step's logits (false: the search is over). finish() returns the results.
  class BeamSearchRun {
  public:
    ~BeamSearchRun();
    bool next_ids(StorageView& step_ids);
    dim_t step() const;
    dim_t decoder_step() const;
    bool with_attention() const;
    StorageView* attention_output();
    StorageView& logits();
    // The run's own clip groups and memory slots for a decoder call it makes alone (search()).
    std::shared_ptr<void> own_scopes();
    // The inputs still decoding (original indices, as in the decoder state's entries kept one per input).
    const std::vector<dim_t>& alive_inputs() const;
    bool keeps_memory_in_place() const;
    bool advance();
    // advance() in three phases with the device's results read on the host only after the caller has synchronized
    // the device stream between them, so that several searches sharing a decoder call wait for the device once a
    // phase instead of twice each (WhisperReplica::decode_stream); each search's device work is the same.
    // queue_processors() applies the logits processors up to their host reads (true: one is pending, synchronize
    // before queue_candidates), queue_candidates() completes them and queues the step's top candidates and their
    // copy to the host (synchronize before take_candidates), take_candidates() takes them as advance() does (false:
    // the search is over).
    bool queue_processors();
    void queue_candidates();
    bool take_candidates();
    // queue_candidates() in two halves, so that searches whose logits are consecutive rows of one tensor (the
    // stream's) take theirs together: prepare_candidates() completes the processors and disables the tokens (true:
    // the rest may be joint), then either own_candidates() (the rest, alone) or, for all of them at once,
    // joint_candidates(): one LogSoftMax, one add of the beams' scores and one TopK over all their rows, copied to the
    // host once (a row's arithmetic is its own in each: ops/softmax_kernels.cuh's choice and ops/topk_gpu.cu's blocks
    // depend on a row's length only).
    bool prepare_candidates();
    void own_candidates();
    static void joint_candidates(const std::vector<BeamSearchRun*>& runs, StorageView& logits);
    std::vector<DecodingResult> finish();

    struct Impl;                                       // the loop state (decoding.cc)

  private:
    friend class BeamSearch;
    std::unique_ptr<Impl> _impl;
  };

  class BiasedDecoder {
  public:
    BiasedDecoder(const float prefix_bias_beta,
                  const std::vector<std::vector<size_t>>& prefix_ids);

    void
    decode(const dim_t cur_batch_size,
           const size_t step,
           const std::vector<dim_t>& batch_offset,
           const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
           const StorageView& logits,
           StorageView& log_probs);
  private:
    StorageView _spare_beam;
    const float _prefix_bias_beta;
    std::vector<std::vector<size_t>> _prefix_ids;
  };


  class GreedySearch : public SearchStrategy {
  public:
    // Penalties are only applied to return scores consistent with the beam search.
    // seeds: on CUDA, each input's sampling seed (DecodingOptions::sampling_seeds), empty for the shared states.
    // temperatures: DecodingOptions::sampling_temperatures (the sampler then a RandomSampler at temperature 1).
    GreedySearch(const float length_penalty = 0,
                 const float coverage_penalty = 0,
                 std::function<bool(DecodingStepResult)> callback = nullptr,
                 const dim_t group_size = 0,
                 std::vector<uint64_t> seeds = {},
                 std::vector<float> temperatures = {});

    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

  private:
    const float _length_penalty;
    const float _coverage_penalty;
    const std::function<bool(DecodingStepResult)> _callback;
    const dim_t _group_size;
    const std::vector<uint64_t> _seeds;
    const std::vector<float> _temperatures;
  };


  struct DecodingOptions {
    size_t beam_size = 1;
    float patience = 1;
    float length_penalty = 0;
    float coverage_penalty = 0;
    float repetition_penalty = 1;
    size_t no_repeat_ngram_size = 0;
    float prefix_bias_beta = 0;
    dim_t start_step = 0;
    size_t max_length = 256;
    size_t min_length = 0;
    size_t sampling_topk = 1;
    float sampling_topp = 1;
    float sampling_temperature = 1;
    size_t num_hypotheses = 1;
    bool include_eos_in_hypotheses = true;
    bool return_scores = false;
    bool return_attention = false;
    bool return_logits_vocab = false;
    bool return_alternatives = false;
    bool return_prefix = true;
    float min_alternative_expansion_prob = 0;
    // On CUDA: the batch is consecutive groups of this many inputs, each decoded with the arithmetic of a batch of
    // its own (cuda/clip_groups.h), in beam search and in sampling; 0 = one batch.
    dim_t group_size = 0;
    // Random sampling on CUDA: a seed per input, from which each of its hypotheses draws a stream of its own
    // (cuda/row_random.h), so an input draws the same in any batch and every run; empty: the thread's shared states.
    std::vector<uint64_t> sampling_seeds;
    // Random sampling (beam_size 1) of every input at each of these temperatures in one search: the variants share
    // the input's decoder steps and its memory keys and values, and each variant samples exactly what a search of
    // its input alone at its temperature samples (on CUDA its hypotheses decode as a group of their own,
    // cuda/clip_groups.h). Results input-major: input i's variant v is result i * variants + v; sampling_seeds then
    // holds a seed per input and variant in that order. Empty: every input at sampling_temperature.
    std::vector<float> sampling_temperatures;
    std::vector<size_t> disable_ids;
    std::vector<size_t> disable_ids_begin;
    std::vector<std::vector<size_t>> disable_sequences;
    std::vector<std::shared_ptr<LogitsProcessor>> logits_processors;
    std::function<bool(DecodingStepResult)> callback = nullptr;
  };

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         std::vector<size_t> end_ids,
         DecodingOptions options = DecodingOptions());

  // decode()'s beam search a step at a time (start_decode): the BeamSearchRun with what decode() builds for it
  // (the strategy, sampler and logits processors), so that several can share decoder steps.
  class DecodeRun {
  public:
    BeamSearchRun& search() {
      return *_run;
    }
    // The results as decode() returns them.
    std::vector<DecodingResult> finish();

  private:
    friend std::unique_ptr<DecodeRun> start_decode(layers::Decoder&, layers::DecoderState&,
                                                   std::vector<std::vector<size_t>>, std::vector<size_t>,
                                                   DecodingOptions);
    DecodeRun() = default;
    layers::Decoder* _decoder = nullptr;
    DecodingOptions _options;
    std::vector<size_t> _end_ids;
    std::vector<size_t> _start_ids;
    std::vector<std::vector<size_t>> _prefix_ids;
    std::unique_ptr<const BeamSearch> _strategy;
    std::unique_ptr<const Sampler> _sampler;
    std::vector<std::shared_ptr<LogitsProcessor>> _processors;
    std::unique_ptr<BeamSearchRun> _run;
  };

  // decode() for a beam search (beam_size > 1, no alternatives) up to its first step.
  std::unique_ptr<DecodeRun> start_decode(layers::Decoder& decoder,
                                          layers::DecoderState& state,
                                          std::vector<std::vector<size_t>> start_tokens,
                                          std::vector<size_t> end_ids,
                                          DecodingOptions options);

}
