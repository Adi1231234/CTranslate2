#include "module.h"

#include <ctranslate2/models/whisper.h>
#include <ctranslate2/models/whisper_stream.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    static models::WhisperOptions whisper_options(size_t beam_size,
                                                  float patience,
                                                  size_t num_hypotheses,
                                                  float length_penalty,
                                                  float repetition_penalty,
                                                  size_t no_repeat_ngram_size,
                                                  size_t max_length,
                                                  bool return_scores,
                                                  bool return_logits_vocab,
                                                  bool return_no_speech_prob,
                                                  size_t max_initial_timestamp_index,
                                                  bool suppress_blank,
                                                  const std::optional<std::vector<int>>& suppress_tokens,
                                                  size_t sampling_topk,
                                                  float sampling_temperature,
                                                  size_t group_size,
                                                  std::vector<uint64_t> sampling_seeds = {},
                                                  std::vector<float> sampling_temperatures = {}) {
      models::WhisperOptions options;
      options.beam_size = beam_size;
      options.patience = patience;
      options.length_penalty = length_penalty;
      options.repetition_penalty = repetition_penalty;
      options.no_repeat_ngram_size = no_repeat_ngram_size;
      options.sampling_topk = sampling_topk;
      options.sampling_temperature = sampling_temperature;
      options.max_length = max_length;
      options.num_hypotheses = num_hypotheses;
      options.return_scores = return_scores;
      options.return_logits_vocab = return_logits_vocab;
      options.return_no_speech_prob = return_no_speech_prob;
      options.max_initial_timestamp_index = max_initial_timestamp_index;
      options.suppress_blank = suppress_blank;
      options.group_size = group_size;
      options.sampling_seeds = std::move(sampling_seeds);
      options.sampling_temperatures = std::move(sampling_temperatures);

      if (suppress_tokens)
        options.suppress_tokens = suppress_tokens.value();
      else
        options.suppress_tokens.clear();
      return options;
    }

    // models::WhisperStream for Python: batches in, finished batches out; closed when dropped.
    class WhisperStreamWrapper {
    public:
      explicit WhisperStreamWrapper(std::shared_ptr<models::WhisperStream> stream)
        : _stream(std::move(stream)) {
      }
      WhisperStreamWrapper(WhisperStreamWrapper&&) = default;
      WhisperStreamWrapper(const WhisperStreamWrapper&) = delete;
      ~WhisperStreamWrapper() {
        if (_stream)
          _stream->close();
      }

      void submit(uint64_t tag, const StorageView& encoder_output, BatchIds prompts) {
        _stream->submit(tag, encoder_output.sync_copy(), std::move(prompts));
      }

      void submit_sampled(uint64_t tag,
                          const StorageView& encoder_output,
                          BatchIds prompts,
                          size_t beam_size,
                          float patience,
                          size_t num_hypotheses,
                          float length_penalty,
                          float repetition_penalty,
                          size_t no_repeat_ngram_size,
                          size_t max_length,
                          bool return_scores,
                          bool return_no_speech_prob,
                          size_t max_initial_timestamp_index,
                          bool suppress_blank,
                          const std::optional<std::vector<int>>& suppress_tokens,
                          size_t sampling_topk,
                          float sampling_temperature,
                          size_t group_size,
                          std::vector<uint64_t> sampling_seeds,
                          std::vector<float> sampling_temperatures) {
        _stream->submit_sampled(tag, encoder_output.sync_copy(), std::move(prompts), whisper_options(
          beam_size, patience, num_hypotheses, length_penalty, repetition_penalty, no_repeat_ngram_size, max_length,
          return_scores, /*return_logits_vocab=*/false, return_no_speech_prob, max_initial_timestamp_index,
          suppress_blank, suppress_tokens, sampling_topk, sampling_temperature, group_size, std::move(sampling_seeds),
          std::move(sampling_temperatures)));
      }

      std::optional<std::pair<uint64_t, std::vector<models::WhisperGenerationResult>>> next() {
        uint64_t tag = 0;
        std::vector<models::WhisperGenerationResult> results;
        if (!_stream->next(tag, results))
          return std::nullopt;
        return std::make_pair(tag, std::move(results));
      }

      void close() {
        _stream->close();
      }

    private:
      std::shared_ptr<models::WhisperStream> _stream;
    };

    class WhisperWrapper : public ReplicaPoolHelper<models::Whisper> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      bool is_multilingual() const {
        return _pool->is_multilingual();
      }

      size_t n_mels() const {
        return _pool->n_mels();
      }

      size_t num_languages() const {
        return _pool->num_languages();
      }

      StorageView encode(const StorageView& features, const bool to_cpu, const size_t group_size) {
        return _pool->encode(features, to_cpu, group_size).get();
      }

      std::variant<std::vector<models::WhisperGenerationResult>,
                   std::vector<AsyncResult<models::WhisperGenerationResult>>>
      generate(const StorageView& features,
               std::variant<BatchTokens, BatchIds> prompts,
               bool asynchronous,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               bool return_logits_vocab,
               bool return_no_speech_prob,
               size_t max_initial_timestamp_index,
               bool suppress_blank,
               const std::optional<std::vector<int>>& suppress_tokens,
               size_t sampling_topk,
               float sampling_temperature,
               size_t group_size,
               std::vector<uint64_t> sampling_seeds,
               std::vector<float> sampling_temperatures) {
        std::vector<std::future<models::WhisperGenerationResult>> futures;

        const models::WhisperOptions options = whisper_options(
          beam_size, patience, num_hypotheses, length_penalty, repetition_penalty, no_repeat_ngram_size,
          max_length, return_scores, return_logits_vocab, return_no_speech_prob, max_initial_timestamp_index,
          suppress_blank, suppress_tokens, sampling_topk, sampling_temperature, group_size,
          std::move(sampling_seeds), std::move(sampling_temperatures));
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (prompts.index() == 0)
          futures = _pool->generate(features, std::get<BatchTokens>(prompts), options);
        else
          futures = _pool->generate(features, std::get<BatchIds>(prompts), options);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      WhisperStreamWrapper open_stream(size_t max_batches,
                                       size_t max_rows,
                                       size_t max_pending,
                                       size_t beam_size,
                                       float patience,
                                       size_t num_hypotheses,
                                       float length_penalty,
                                       float repetition_penalty,
                                       size_t no_repeat_ngram_size,
                                       size_t max_length,
                                       bool return_scores,
                                       bool return_no_speech_prob,
                                       size_t max_initial_timestamp_index,
                                       bool suppress_blank,
                                       const std::optional<std::vector<int>>& suppress_tokens) {
        models::WhisperOptions options = whisper_options(
          beam_size, patience, num_hypotheses, length_penalty, repetition_penalty, no_repeat_ngram_size,
          max_length, return_scores, /*return_logits_vocab=*/false, return_no_speech_prob,
          max_initial_timestamp_index, suppress_blank, suppress_tokens, /*sampling_topk=*/1,
          /*sampling_temperature=*/1, /*group_size=*/0);
        const models::WhisperStreamLimits limits{max_batches, max_rows, max_pending};
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return WhisperStreamWrapper(_pool->open_stream(std::move(options), limits));
      }

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(const StorageView& features) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        auto futures = _pool->detect_language(features);
        return wait_on_futures(std::move(futures));
      }

      std::vector<models::WhisperAlignmentResult>
      align(const StorageView& features,
            Ids start_sequence,
            BatchIds text_tokens,
            const std::variant<size_t, std::vector<size_t>>& num_frames,
            size_t median_filter_width) {
        const size_t batch_size = text_tokens.size();

        std::vector<size_t> batch_num_frames;
        if (num_frames.index() == 0)
          batch_num_frames.resize(batch_size, std::get<size_t>(num_frames));
        else
          batch_num_frames = std::get<std::vector<size_t>>(num_frames);
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        auto futures = _pool->align(features,
                                    std::move(start_sequence),
                                    std::move(text_tokens),
                                    std::move(batch_num_frames),
                                    median_filter_width);
        return wait_on_futures(std::move(futures));
      }
    };


    void register_whisper(py::module& m) {
      py::class_<models::WhisperGenerationResult>(m, "WhisperGenerationResult",
                                                  "A generation result from the Whisper model.")

        .def_readonly("sequences", &models::WhisperGenerationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &models::WhisperGenerationResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &models::WhisperGenerationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")
        .def_readonly("logits", &models::WhisperGenerationResult::logits,
                      "logits in each sequence (empty if :obj:`return_logits_vocab` was disabled).")
        .def_readonly("no_speech_prob", &models::WhisperGenerationResult::no_speech_prob,
                      "Probability of the no speech token (0 if :obj:`return_no_speech_prob` was disabled).")

        .def("__repr__", [](const models::WhisperGenerationResult& result) {
          return "WhisperGenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", logits=" + std::string(py::repr(py::cast(result.logits)))
            + ", no_speech_prob=" + std::string(py::repr(py::cast(result.no_speech_prob)))
            + ")";
        })
        ;

      declare_async_wrapper<models::WhisperGenerationResult>(m, "WhisperGenerationResultAsync");

      py::class_<models::WhisperAlignmentResult>(m, "WhisperAlignmentResult",
                                                 "An alignment result from the Whisper model.")

        .def_readonly("alignments", &models::WhisperAlignmentResult::alignments,
                      "List of aligned text and time indices.")
        .def_readonly("text_token_probs", &models::WhisperAlignmentResult::text_token_probs,
                      "Probabilities of text tokens.")

        .def("__repr__", [](const models::WhisperAlignmentResult& result) {
          return "WhisperAlignmentResult(alignments=" + std::string(py::repr(py::cast(result.alignments)))
            + ", text_token_probs=" + std::string(py::repr(py::cast(result.text_token_probs)))
            + ")";
        })
        ;

      py::class_<WhisperStreamWrapper>(
        m, "WhisperStream",
        R"pbdoc(
            Batches decoded together by one worker of a :class:`Whisper` model, each exactly as
            :meth:`Whisper.generate` decodes it alone, a batch joining as soon as there is room
            (:meth:`Whisper.open_stream`).
        )pbdoc")
        .def("submit", &WhisperStreamWrapper::submit,
             py::arg("tag"),
             py::arg("encoder_output"),
             py::arg("prompts"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Queues a batch; blocks while ``max_pending`` batches wait.

                 Arguments:
                   tag: The batch's number, returned with its results.
                   encoder_output: Its encoder output (:meth:`Whisper.encode`, on the model's device).
                   prompts: Its prompts, as token IDs.
             )pbdoc")
        .def("submit_sampled", &WhisperStreamWrapper::submit_sampled,
             py::arg("tag"),
             py::arg("encoder_output"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("beam_size")=1,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("max_initial_timestamp_index")=50,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::arg("group_size")=0,
             py::arg("sampling_seeds")=std::vector<uint64_t>(),
             py::arg("sampling_temperatures")=std::vector<float>(),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Queues a batch of random sampling (``beam_size`` 1) with its own options, which
                 :meth:`Whisper.generate` takes alike: decoded with the stream's other batches exactly as
                 :meth:`Whisper.generate` alone decodes it (shared memory rows and capacity caches on CUDA:
                 ``CT2_CAPACITY_CACHES=1``). Its results come from :meth:`next` under its tag, as
                 :meth:`Whisper.generate` returns them.
             )pbdoc")
        .def("next", &WhisperStreamWrapper::next,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Waits for a finished batch: ``(tag, results)`` as :meth:`Whisper.generate` returns them,
                 or ``None`` once the stream is closed and every batch returned.
             )pbdoc")
        .def("close", &WhisperStreamWrapper::close,
             py::call_guard<py::gil_scoped_release>(),
             "No more batches.")
        ;

      py::class_<WhisperWrapper>(
        m, "Whisper",
        R"pbdoc(
            Implements the Whisper speech recognition model published by OpenAI.

            See Also:
               https://github.com/openai/whisper
        )pbdoc")

        .def_property_readonly("is_multilingual", &WhisperWrapper::is_multilingual,
                               "Returns ``True`` if this model is multilingual.")

        .def_property_readonly("n_mels", &WhisperWrapper::n_mels,
                               "Returns dimension of mel input features.")

        .def_property_readonly("num_languages", &WhisperWrapper::num_languages,
                               "Returns the number of languages supported.")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Whisper model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &WhisperWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &WhisperWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &WhisperWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &WhisperWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &WhisperWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &WhisperWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &WhisperWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &WhisperWrapper::encode,
             py::arg("features"),
             py::arg("to_cpu")=false,
             py::arg("group_size")=0,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``.
                   to_cpu: Copy the encoder output to the CPU before returning the value.
                   group_size: Encode consecutive groups of this many inputs as batches of their own
                     (0: one batch), for :meth:`generate` with the same ``group_size``.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("generate", &WhisperWrapper::generate,
             py::arg("features"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_logits_vocab")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("max_initial_timestamp_index")=50,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::arg("group_size")=0,
             py::arg("sampling_seeds")=std::vector<uint64_t>(),
             py::arg("sampling_temperatures")=std::vector<float>(),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features and generates from the given prompt.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.
                   prompts: Batch of initial string tokens or token IDs.
                   asynchronous: Run the model asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
                   return_logits_vocab: Include the log probs in the output
                   return_no_speech_prob: Include the probability of the no speech token in the
                     result.
                   max_initial_timestamp_index: Maximum index of the first predicted timestamp.
                   suppress_blank: Suppress blank outputs at the beginning of the sampling.
                   suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                     of symbols as defined in the model ``config.json`` file.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.
                   group_size: Beam search on CUDA: decode consecutive groups of this many inputs, each
                     exactly as a batch of its own would be (0: one batch); the groups' products run back to
                     back, so the decoder weights are read once for all of them.
                   sampling_seeds: Random sampling on CUDA: a seed per input; each of its hypotheses draws a
                     random stream of its own, so an input samples the same in any batch and on every run.
                   sampling_temperatures: Random sampling of every input at each of these temperatures in one
                     search (sharing its decoder steps and memory): a result per input and temperature,
                     input-major, each what this method returns for that input alone at that temperature;
                     sampling_seeds then holds a seed per input and temperature, in that order.

                 Returns:
                   A list of generation results.
             )pbdoc")

        .def("open_stream", &WhisperWrapper::open_stream,
             py::kw_only(),
             py::arg("max_batches")=8,
             py::arg("max_rows")=320,
             py::arg("max_pending")=2,
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("max_initial_timestamp_index")=50,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Opens a stream of batches decoded together with a beam search on one worker (it keeps
                 that worker until closed): every batch gets exactly what :meth:`generate` with these
                 options returns for it alone, while each decoding step runs once for every batch in flight.

                 Arguments:
                   max_batches: Batches decoding at once.
                   max_rows: Rows (inputs x beams) decoding when a batch may join.
                   max_pending: Batches submitted and not yet decoding.
                   The other arguments: as in :meth:`generate`.

                 Returns:
                   A :class:`WhisperStream`.
             )pbdoc")

        .def("detect_language", &WhisperWrapper::detect_language,
             py::arg("features"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Returns the probability of each language.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.

                 Returns:
                   For each batch, a list of pairs (language, probability) ordered from
                   best to worst probability.

                 Raises:
                   RuntimeError: if the model is not multilingual.
             )pbdoc")

        .def("align", &WhisperWrapper::align,
             py::arg("features"),
             py::arg("start_sequence"),
             py::arg("text_tokens"),
             py::arg("num_frames"),
             py::kw_only(),
             py::arg("median_filter_width")=7,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Computes the alignments between the text tokens and the audio.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.
                   start_sequence: The start sequence tokens.
                   text_tokens: Batch of text tokens to align.
                   num_frames: Number of non padding frames in the features.
                   median_filter_width: Width of the median filter kernel.

                 Returns:
                   A list of alignment results.
             )pbdoc")

        .def("unload_model", &WhisperWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this whisper but keep enough runtime context
                 to quickly resume whisper on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &WhisperWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &WhisperWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
