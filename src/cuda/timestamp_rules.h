#pragma once

#include <vector>

#include "ctranslate2/types.h"
#include "cuda/pinned_buffer.h"

namespace ctranslate2 {
  namespace cuda {

    // For each row in `rows` of the [*, vocabulary] log-probs on the GPU: whether the log of the
    // summed probabilities of the timestamp tokens [begin, end] exceeds the best text token [0, begin),
    // i.e. should_sample_timestamp of models/whisper.cc for every row with one host synchronization
    // instead of three per row. Same values, bit for bit (see timestamp_rules.cuh).
    template <typename T>
    std::vector<bool> sample_timestamps(const T* log_probs,
                                        dim_t vocabulary_size,
                                        const std::vector<dim_t>& rows,
                                        dim_t timestamp_begin,
                                        dim_t timestamp_end);

    // sample_timestamps in two halves, so that several searches wait for the device once: queue_ queues the rows'
    // maxima and sums and their copy to the host into `host`, read_ gives the answers once the stream has
    // synchronized.
    template <typename T>
    void queue_sample_timestamps(const T* log_probs,
                                 dim_t vocabulary_size,
                                 const std::vector<dim_t>& rows,
                                 dim_t timestamp_begin,
                                 dim_t timestamp_end,
                                 PinnedBuffer& host);
    template <typename T>
    std::vector<bool> read_sample_timestamps(const PinnedBuffer& host, size_t rows);

  }
}
