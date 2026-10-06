#ifndef CT2_USE_HIP

#include "cuda/timestamp_rules.h"

#include <cmath>
#include <limits>

#include "ctranslate2/allocator.h"
#include "cuda/timestamp_rules.cuh"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    template <typename T>
    void queue_sample_timestamps(const T* log_probs,
                                 dim_t vocabulary_size,
                                 const std::vector<dim_t>& rows,
                                 dim_t timestamp_begin,
                                 dim_t timestamp_end,
                                 PinnedBuffer& host) {
      using DT = device_type<T>;
      const size_t n = rows.size();
      if (n == 0)
        return;
      cudaStream_t stream = get_cuda_stream();
      const DT lowest = DT(std::numeric_limits<T>::lowest());   // the init of primitives::max
      const int begin = static_cast<int>(timestamp_begin);
      const int end = static_cast<int>(timestamp_end);
      const size_t temp_bytes = timestamp_mass_temp_bytes<DT>(static_cast<int>(n), begin, end, lowest, stream);

      // One allocation: [n] text max, [n] timestamp max, [n] exp sums, then the temporary storage.
      const size_t results_bytes = n * (2 * sizeof (DT) + sizeof (float));
      const size_t temp_offset = (results_bytes + 255) / 256 * 256;
      Allocator& allocator = get_allocator<Device::CUDA>();
      char* buffer = static_cast<char*>(allocator.allocate(temp_offset + temp_bytes));
      DT* device_maxima = reinterpret_cast<DT*>(buffer);
      float* device_sums = reinterpret_cast<float*>(device_maxima + 2 * n);   // 2n * sizeof(DT): 4-aligned

      const std::vector<int32_t> rows32(rows.begin(), rows.end());
      timestamp_mass_enqueue<DT>(device_cast(log_probs), static_cast<int>(vocabulary_size), rows32,
                                 begin, end, lowest, device_maxima, device_sums, buffer + temp_offset,
                                 temp_bytes, stream);
      host.copy_from_device(buffer, results_bytes);
      allocator.free(buffer);                                // in stream order: reused after the copy
    }

    template <typename T>
    std::vector<bool> read_sample_timestamps(const PinnedBuffer& host, size_t n) {
      const T* maxima = static_cast<const T*>(host.data());
      const float* sums = reinterpret_cast<const float*>(maxima + 2 * n);
      std::vector<bool> sample(n);
      for (size_t r = 0; r < n; ++r) {
        // The host arithmetic of should_sample_timestamp and primitives<CUDA>::logsumexp.
        const float max_text_token_log_prob = maxima[r];
        const float max_value = maxima[n + r];
        const float timestamp_log_prob = std::log(sums[r]) + max_value;
        sample[r] = timestamp_log_prob > max_text_token_log_prob;
      }
      return sample;
    }

    template <typename T>
    std::vector<bool> sample_timestamps(const T* log_probs,
                                        dim_t vocabulary_size,
                                        const std::vector<dim_t>& rows,
                                        dim_t timestamp_begin,
                                        dim_t timestamp_end) {
      if (rows.empty())
        return {};
      PinnedBuffer host;
      queue_sample_timestamps(log_probs, vocabulary_size, rows, timestamp_begin, timestamp_end, host);
      synchronize_stream();
      return read_sample_timestamps<T>(host, rows.size());
    }

#define DECLARE_IMPL(T)                                                  \
    template std::vector<bool> sample_timestamps(const T*, dim_t,        \
                                                 const std::vector<dim_t>&, \
                                                 dim_t, dim_t);          \
    template void queue_sample_timestamps(const T*, dim_t,               \
                                          const std::vector<dim_t>&,     \
                                          dim_t, dim_t, PinnedBuffer&);  \
    template std::vector<bool> read_sample_timestamps<T>(const PinnedBuffer&, size_t);

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
