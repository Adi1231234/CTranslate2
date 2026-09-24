#pragma once

// Whisper's timestamp rule, batched over rows. For each selected row of the [*, vocabulary]
// log-probs: the max over the text tokens [0, begin) and, for the logsumexp over the timestamp
// tokens [begin, end], their max and the sum of exp(x - max). primitives<CUDA>::max and ::logsumexp
// make one cub::DeviceReduce::Reduce per value per row (through thrust::reduce); here all rows go
// through two cub::DeviceSegmentedReduce::Reduce launches. The sums keep their exact order: a
// segment runs the same AgentReduce, with the same policy (SegmentedReducePolicy is ReducePolicy is
// SingleTilePolicy), on the same input iterator values and the same init as the single-tile kernel
// of a 1501-item DeviceReduce. The maxima are exact in any order. No per-row launch, allocation or
// synchronization. Checked bit for bit by tools/turing/kernels/ts_check.cu.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace cuda {

    // Maxima segment s: the text tokens of selected row s (s < n), else the timestamps of row s - n.
    template <bool IsEnd>
    struct timestamp_max_bound {
      const int32_t* rows;
      int vocabulary_size, n, begin, end;
      __device__ int operator()(int s) const {
        const bool text = s < n;
        const int row_start = rows[text ? s : s - n] * vocabulary_size;
        return row_start + (text ? (IsEnd ? begin : 0) : (IsEnd ? end + 1 : begin));
      }
    };

    // exp(x - max) for timestamp j = e % count of selected row e / count: the expression of
    // exp_minus_max_func in primitives.cu, with the row's max read from device memory.
    template <typename DT>
    struct timestamp_exp {
      const DT* log_probs;
      const DT* timestamp_max;
      const int32_t* rows;
      int vocabulary_size, begin, count;
      __device__ float operator()(int e) const {
        const int r = e / count;
        const DT x = log_probs[rows[r] * vocabulary_size + begin + (e - r * count)];
        return expf(float(x) - float(timestamp_max[r]));
      }
    };

    struct times {
      int k;
      __device__ int operator()(int i) const { return i * k; }
    };

    using counting = thrust::counting_iterator<int>;

    // Temporary storage of timestamp_mass_enqueue for n rows: the row list, then cub's storage.
    template <typename DT>
    size_t timestamp_mass_temp_bytes(int n, int begin, int end, DT lowest, cudaStream_t stream) {
      const int count = end - begin + 1;
      const auto lo = thrust::make_transform_iterator(counting(0), timestamp_max_bound<false>{});
      const auto hi = thrust::make_transform_iterator(counting(0), timestamp_max_bound<true>{});
      const auto exps = thrust::make_transform_iterator(counting(0), timestamp_exp<DT>{});
      const auto starts = thrust::make_transform_iterator(counting(0), times{count});
      size_t max_bytes = 0, sum_bytes = 0;
      cub::DeviceSegmentedReduce::Reduce(nullptr, max_bytes, static_cast<const DT*>(nullptr),
                                         static_cast<DT*>(nullptr), 2 * n, lo, hi, maximum<DT>(),
                                         lowest, stream);
      cub::DeviceSegmentedReduce::Reduce(nullptr, sum_bytes, exps, static_cast<float*>(nullptr), n,
                                         starts, starts + 1, thrust::plus<float>(), 0.f, stream);
      return (n * sizeof (int32_t) + 255) / 256 * 256 + std::max(max_bytes, sum_bytes);
    }

    // Enqueues the reductions of the selected rows; no host synchronization. maxima[r] is the text
    // max of rows[r], maxima[n + r] its timestamp max, sums[r] its exp sum.
    template <typename DT>
    void timestamp_mass_enqueue(const DT* log_probs, int vocabulary_size,
                                const std::vector<int32_t>& rows, int begin, int end, DT lowest,
                                DT* maxima, float* sums, void* temp, size_t temp_bytes,
                                cudaStream_t stream) {
      const int n = static_cast<int>(rows.size()), count = end - begin + 1;
      int32_t* d_rows = static_cast<int32_t*>(temp);
      const size_t rows_bytes = (n * sizeof (int32_t) + 255) / 256 * 256;
      void* cub_temp = static_cast<char*>(temp) + rows_bytes;
      CUDA_CHECK(cudaMemcpyAsync(d_rows, rows.data(), n * sizeof (int32_t), cudaMemcpyHostToDevice, stream));
      const auto lo = thrust::make_transform_iterator(
        counting(0), timestamp_max_bound<false>{d_rows, vocabulary_size, n, begin, end});
      const auto hi = thrust::make_transform_iterator(
        counting(0), timestamp_max_bound<true>{d_rows, vocabulary_size, n, begin, end});
      size_t bytes = temp_bytes - rows_bytes;
      CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(cub_temp, bytes, log_probs, maxima, 2 * n, lo, hi,
                                                    maximum<DT>(), lowest, stream));
      const auto exps = thrust::make_transform_iterator(
        counting(0), timestamp_exp<DT>{log_probs, maxima + n, d_rows, vocabulary_size, begin, count});
      const auto starts = thrust::make_transform_iterator(counting(0), times{count});
      bytes = temp_bytes - rows_bytes;
      CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(cub_temp, bytes, exps, sums, n, starts, starts + 1,
                                                    thrust::plus<float>(), 0.f, stream));
    }

  }
}
