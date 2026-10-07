#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  // The logits processors' device work of several searches whose logits are rows of one tensor (a Whisper stream's
  // joint step, models/whisper_stream.cc), in one launch each instead of several a search. While a JointLogits
  // lives on the thread (current()), Whisper's timestamp rules hand it a search's disabled tokens and the rows whose
  // timestamp mass they need (add); flush() then queues every search's disabled tokens in one launch, one
  // log-softmax over their rows and one timestamp-mass reduction for all of them, and each search's answers are read
  // once the stream has synchronized. Every row's values are its search's own: the same disabled tokens, and a
  // row's log-softmax and reductions depend on that row only (prof3: ~150 launches a step after the last layer,
  // 3.1 ms, for ~35 searches).
  class JointLogits {
  public:
    explicit JointLogits(StorageView& logits);
    ~JointLogits();
    JointLogits(const JointLogits&) = delete;
    JointLogits& operator=(const JointLogits&) = delete;

    // The thread's, or null.
    static JointLogits* current();

    // Whether `part` (a search's logits) is rows of this tensor and the joint work applies to it.
    bool accepts(const StorageView& part) const;

    // A search's logits, its disabled tokens (listed, applied by flush, then forgotten) and its rows to check
    // (indices into `part`); returns what reads their answers once flushed and synchronized.
    std::function<std::vector<bool>()> add(const StorageView& part, DisableTokens& disable_tokens,
                                          const std::vector<dim_t>& check_rows, dim_t timestamp_begin,
                                          dim_t timestamp_end);

    // Queues the device work of every search added.
    void flush();

  private:
    struct Part {
      dim_t row;                                             // its first row in the tensor
      dim_t rows;
      DisableTokens* disable_tokens;
    };
    struct Answers;

    StorageView& _logits;
    JointLogits* _previous;
    std::vector<Part> _parts;
    std::vector<dim_t> _check_rows;                          // rows of the tensor, in the order added
    dim_t _timestamp_begin = -1;
    dim_t _timestamp_end = -1;
    std::shared_ptr<Answers> _answers;
  };

}
