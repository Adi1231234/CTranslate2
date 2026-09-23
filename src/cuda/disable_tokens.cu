#ifndef CT2_USE_HIP

#include "cuda/disable_tokens.h"

#include <algorithm>

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // One launch's share of the lists, passed by value: 4 KB of kernel arguments at most.
    constexpr int32_t disable_chunk_capacity = 1000;

    struct DisableChunk {
      int32_t num_ranges;    // [begin, end) pairs at the start of data
      int32_t num_singles;   // flat indices after the ranges
      int32_t num_columns;   // token ids to disable in every row, after the singles
      int32_t rows;
      int32_t vocabulary_size;
      int32_t data[disable_chunk_capacity];
    };

    // Blocks [0, num_ranges): one range each; block num_ranges: the singles; the next `rows` blocks
    // (when there are columns): the columns of one row each.
    template <typename T>
    __global__ void disable_tokens_kernel(T* logits, const T value, const DisableChunk chunk) {
      const int32_t b = blockIdx.x;
      if (b < chunk.num_ranges) {
        const int32_t end = chunk.data[2 * b + 1];
        for (int32_t i = chunk.data[2 * b] + threadIdx.x; i < end; i += blockDim.x)
          logits[i] = value;
      } else if (b == chunk.num_ranges) {
        const int32_t* singles = chunk.data + 2 * chunk.num_ranges;
        for (int32_t i = threadIdx.x; i < chunk.num_singles; i += blockDim.x)
          logits[singles[i]] = value;
      } else {
        const int64_t row = b - chunk.num_ranges - 1;
        const int32_t* columns = chunk.data + 2 * chunk.num_ranges + chunk.num_singles;
        for (int32_t i = threadIdx.x; i < chunk.num_columns; i += blockDim.x)
          logits[row * chunk.vocabulary_size + columns[i]] = value;
      }
    }

    template <typename T>
    void disable_tokens(T* logits,
                        T value,
                        const std::vector<int32_t>& ranges,
                        const std::vector<int32_t>& singles,
                        const std::vector<int32_t>& columns,
                        int32_t rows,
                        int32_t vocabulary_size) {
      auto* x = device_cast(logits);
      const auto v = device_type<T>(value);
      size_t r = 0, s = 0, c = 0;                     // consumed range ints, singles, columns
      while (r < ranges.size() || s < singles.size() || c < columns.size()) {
        DisableChunk chunk;
        int32_t used = 0;
        chunk.num_ranges = chunk.num_singles = chunk.num_columns = 0;
        while (r + 1 < ranges.size() && used + 2 <= disable_chunk_capacity) {
          chunk.data[used++] = ranges[r++];
          chunk.data[used++] = ranges[r++];
          ++chunk.num_ranges;
        }
        const int32_t n_singles = std::min<int32_t>(singles.size() - s, disable_chunk_capacity - used);
        std::copy(singles.begin() + s, singles.begin() + s + n_singles, chunk.data + used);
        used += n_singles; s += n_singles; chunk.num_singles = n_singles;
        const int32_t n_columns = std::min<int32_t>(columns.size() - c, disable_chunk_capacity - used);
        std::copy(columns.begin() + c, columns.begin() + c + n_columns, chunk.data + used);
        used += n_columns; c += n_columns; chunk.num_columns = n_columns;
        chunk.rows = rows;
        chunk.vocabulary_size = vocabulary_size;
        const int32_t blocks = chunk.num_ranges + 1 + (chunk.num_columns > 0 ? rows : 0);
        disable_tokens_kernel<<<blocks, 256, 0, get_cuda_stream()>>>(x, v, chunk);
      }
    }

#define DECLARE_IMPL(T)                                                  \
    template void disable_tokens(T*, T, const std::vector<int32_t>&,    \
                                 const std::vector<int32_t>&,            \
                                 const std::vector<int32_t>&, int32_t, int32_t);

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
