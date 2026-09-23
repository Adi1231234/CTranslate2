#pragma once

#include <cstdint>
#include <vector>

namespace ctranslate2 {
  namespace cuda {

    // Sets logits[i] = value for every flat index i in the [begin, end) `ranges` pairs, in `singles`,
    // and at the token ids `columns` of every one of `rows` rows. The lists travel as kernel
    // arguments (split over several launches when long), so no host-to-device copy and no stream
    // synchronization is involved. T is float, float16_t or bfloat16_t.
    template <typename T>
    void disable_tokens(T* logits,
                        T value,
                        const std::vector<int32_t>& ranges,
                        const std::vector<int32_t>& singles,
                        const std::vector<int32_t>& columns,
                        int32_t rows,
                        int32_t vocabulary_size);

  }
}
