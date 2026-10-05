#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>

#include "ctranslate2/models/whisper.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperStreamLimits {
      size_t max_batches = 8;      // batches decoding at once
      size_t max_rows = 320;       // their rows (inputs x beams) a batch may join with
      size_t max_pending = 2;      // batches submitted and not yet decoding
    };

    // Batches decoded together, each exactly as WhisperReplica::generate decodes it alone (the same results, bit
    // for bit), a batch joining as soon as there is room: every step is one decoder call for all the batches
    // decoding (TransformerDecoder::decode_joint), so the decoder weights are read once for them all and a batch's
    // last long hypotheses never decode alone. Beam search on CUDA; one worker of the pool runs it
    // (Whisper::open_stream, WhisperReplica::decode_stream).
    class WhisperStream {
    public:
      WhisperStream(WhisperOptions options, WhisperStreamLimits limits);

      // A batch as generate takes it: its encoder output (on the model's device) and prompts. Blocks while
      // max_pending batches wait.
      void submit(uint64_t tag, StorageView encoder_output, std::vector<std::vector<size_t>> prompts);
      // No more batches.
      void close();
      // Blocks until a batch is finished; false once the stream is closed and every batch returned.
      bool next(uint64_t& tag, std::vector<WhisperGenerationResult>& results);

      // The decoding side (WhisperReplica::decode_stream).
      struct Batch {
        uint64_t tag = 0;
        StorageView encoder_output;
        std::vector<std::vector<size_t>> prompts;
      };
      // The next batch: false when none is waiting (with wait, only once the stream is closed and empty).
      bool take(Batch& batch, bool wait);
      void finished(uint64_t tag, std::vector<WhisperGenerationResult> results);
      void failed(std::exception_ptr error);
      const WhisperOptions& options() const {
        return _options;
      }
      const WhisperStreamLimits& limits() const {
        return _limits;
      }

    private:
      const WhisperOptions _options;
      const WhisperStreamLimits _limits;
      std::mutex _mutex;
      std::condition_variable _changed;
      std::deque<Batch> _pending;
      std::deque<std::pair<uint64_t, std::vector<WhisperGenerationResult>>> _done;
      size_t _submitted = 0;
      size_t _returned = 0;
      bool _closed = false;
      std::exception_ptr _error;
    };

  }
}
