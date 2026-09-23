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
    std::vector<bool> sample_timestamps(const T* log_probs,
                                        dim_t vocabulary_size,
                                        const std::vector<dim_t>& rows,
                                        dim_t timestamp_begin,
                                        dim_t timestamp_end) {
      using DT = device_type<T>;
      const size_t n = rows.size();
      if (n == 0)
        return {};
      cudaStream_t stream = get_cuda_stream();
      const DT lowest = DT(std::numeric_limits<T>::lowest());   // the init of primitives::max
      const int begin = static_cast<int>(timestamp_begin);
      const int end = static_cast<int>(timestamp_end);
      const size_t temp_bytes = timestamp_mass_temp_bytes<DT>(begin, end, lowest, stream);

      // One allocation: [n] text max, [n] timestamp max, [n] exp sums, then cub's temporary storage.
      const size_t results_bytes = n * (2 * sizeof (DT) + sizeof (float));
      const size_t temp_offset = (results_bytes + 255) / 256 * 256;
      Allocator& allocator = get_allocator<Device::CUDA>();
      char* buffer = static_cast<char*>(allocator.allocate(temp_offset + temp_bytes));
      DT* text_max = reinterpret_cast<DT*>(buffer);
      DT* timestamp_max = text_max + n;
      float* timestamp_sum = reinterpret_cast<float*>(timestamp_max + n);   // 2n * sizeof(DT): 4-aligned

      const std::vector<int64_t> rows64(rows.begin(), rows.end());
      timestamp_mass_enqueue<DT>(device_cast(log_probs), vocabulary_size, rows64, begin, end, lowest,
                                 text_max, timestamp_max, timestamp_sum, buffer + temp_offset,
                                 temp_bytes, stream);

      std::vector<T> maxima(2 * n);
      std::vector<float> sums(n);
      CUDA_CHECK(cudaMemcpyAsync(maxima.data(), text_max, 2 * n * sizeof (DT), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(sums.data(), timestamp_sum, n * sizeof (float), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      allocator.free(buffer);

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

#define DECLARE_IMPL(T)                                                  \
    template std::vector<bool> sample_timestamps(const T*, dim_t,        \
                                                 const std::vector<dim_t>&, \
                                                 dim_t, dim_t);

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
