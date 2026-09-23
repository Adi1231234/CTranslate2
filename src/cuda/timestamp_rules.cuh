#pragma once

// Whisper's timestamp rule, batched over rows. For each row of the [*, vocabulary] log-probs: the max
// over the text tokens [0, begin) and, for the logsumexp over the timestamp tokens [begin, end], their
// max and the sum of exp(x - max). Every reduction is the same cub::DeviceReduce::Reduce call that
// primitives<CUDA>::max and ::logsumexp make through thrust::reduce (same input type, operator, init
// and 32-bit item count, so the same kernels and summation order). What is gone is the per-row
// temporary allocation, stream synchronization and copy of each scalar to the host: all rows are
// enqueued, then read back once. Checked bit for bit by tools/turing/kernels/ts_check.cu.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace cuda {

    // exp(x - max): the expression of exp_minus_max_func in primitives.cu, with the max read from
    // device memory instead of being passed by value from the host.
    template <typename DT>
    struct exp_minus_device_max {
      const DT* max_value;
      __device__ float operator()(DT x) const {
        return expf(float(x) - float(*max_value));
      }
    };

    template <typename DT>
    using exp_minus_max_iterator = thrust::transform_iterator<exp_minus_device_max<DT>,
                                                              thrust::device_ptr<const DT>>;

    // cub temporary storage for any of the three reductions of one row.
    template <typename DT>
    size_t timestamp_mass_temp_bytes(int begin, int end, DT lowest, cudaStream_t stream) {
      const int num_timestamps = end - begin + 1;
      size_t text = 0, timestamps = 0, sum = 0;
      cub::DeviceReduce::Reduce(nullptr, text, static_cast<const DT*>(nullptr), static_cast<DT*>(nullptr),
                                begin, maximum<DT>(), lowest, stream);
      cub::DeviceReduce::Reduce(nullptr, timestamps, static_cast<const DT*>(nullptr),
                                static_cast<DT*>(nullptr), num_timestamps, maximum<DT>(), lowest, stream);
      const exp_minus_max_iterator<DT> it(thrust::device_pointer_cast(static_cast<const DT*>(nullptr)),
                                          exp_minus_device_max<DT>{nullptr});
      cub::DeviceReduce::Reduce(nullptr, sum, it, static_cast<float*>(nullptr), num_timestamps,
                                thrust::plus<float>(), 0.f, stream);
      return std::max(text, std::max(timestamps, sum));
    }

    // Enqueues the reductions of every row; no host synchronization. Results: text_max[r],
    // timestamp_max[r] and timestamp_sum[r] (the exp sum) for rows[r].
    template <typename DT>
    void timestamp_mass_enqueue(const DT* log_probs, int64_t vocabulary_size,
                                const std::vector<int64_t>& rows, int begin, int end, DT lowest,
                                DT* text_max, DT* timestamp_max, float* timestamp_sum,
                                void* temp, size_t temp_bytes, cudaStream_t stream) {
      const int num_timestamps = end - begin + 1;
      for (size_t r = 0; r < rows.size(); ++r) {
        const DT* row = log_probs + rows[r] * vocabulary_size;
        size_t bytes = temp_bytes;
        cub::DeviceReduce::Reduce(temp, bytes, row, text_max + r, begin, maximum<DT>(), lowest, stream);
        bytes = temp_bytes;
        cub::DeviceReduce::Reduce(temp, bytes, row + begin, timestamp_max + r, num_timestamps,
                                  maximum<DT>(), lowest, stream);
        bytes = temp_bytes;
        const exp_minus_max_iterator<DT> it(thrust::device_pointer_cast(row + begin),
                                            exp_minus_device_max<DT>{timestamp_max + r});
        cub::DeviceReduce::Reduce(temp, bytes, it, timestamp_sum + r, num_timestamps,
                                  thrust::plus<float>(), 0.f, stream);
      }
    }

  }
}
