#pragma once

#include <vector>

#include "ctranslate2/types.h"

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

  }
}
