#include "cuda/row_random.h"

#include <algorithm>

#include "ctranslate2/allocator.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local const RowRandom* active = nullptr;

    const RowRandom* row_random() {
      return active;
    }

    RowRandomScope::RowRandomScope(const RowRandom& random)
      : _previous(active) {
      active = &random;
    }

    RowRandomScope::~RowRandomScope() {
      active = _previous;
    }

    __global__ void init_row_states_kernel(curandStatePhilox4_32_10_t* states, const unsigned long long* seeds,
                                           int count) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < count)
        curand_init(seeds[2 * i], seeds[2 * i + 1], 0, states + i);
    }

    RowStates::RowStates(const std::vector<std::pair<uint64_t, uint64_t>>& seeds) {
      Allocator& allocator = get_allocator<Device::CUDA>();
      const int count = static_cast<int>(seeds.size());
      _states = static_cast<curandStatePhilox4_32_10_t*>(
        allocator.allocate(std::max(count, 1) * sizeof (curandStatePhilox4_32_10_t)));
      std::vector<unsigned long long> flat;
      flat.reserve(2 * seeds.size());
      for (const auto& [seed, subsequence] : seeds) {
        flat.push_back(seed);
        flat.push_back(subsequence);
      }
      auto* device_seeds = static_cast<unsigned long long*>(
        allocator.allocate(std::max<size_t>(flat.size(), 1) * sizeof (unsigned long long)));
      CUDA_CHECK(cudaMemcpyAsync(device_seeds, flat.data(), flat.size() * sizeof (unsigned long long),
                                 cudaMemcpyHostToDevice, get_cuda_stream()));
      if (count > 0)
        init_row_states_kernel<<<(count + 63) / 64, 64, 0, get_cuda_stream()>>>(_states, device_seeds, count);
      CUDA_CHECK(cudaStreamSynchronize(get_cuda_stream()));  // flat (host) must outlive the copy
      allocator.free(device_seeds);
    }

    RowStates::~RowStates() {
      get_allocator<Device::CUDA>().free(_states);
    }

  }
}
