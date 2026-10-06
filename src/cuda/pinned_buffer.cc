#ifndef CT2_USE_HIP

#include "cuda/pinned_buffer.h"

#include <map>
#include <mutex>
#include <utility>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Free blocks by size (powers of two from 256 bytes). Never released: the process holds a few kilobytes, and a
    // pool torn down at exit would call the runtime while it unloads.
    static std::mutex pool_mutex;
    static std::multimap<size_t, void*>& free_blocks() {
      static auto* blocks = new std::multimap<size_t, void*>();
      return *blocks;
    }

    static void give_back(void* data, size_t size) {
      if (!data)
        return;
      const std::lock_guard<std::mutex> lock(pool_mutex);
      free_blocks().emplace(size, data);
    }

    PinnedBuffer::~PinnedBuffer() {
      give_back(_data, _size);
    }

    PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
      : _data(std::exchange(other._data, nullptr))
      , _size(std::exchange(other._size, 0)) {
    }

    PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
      if (this != &other) {
        give_back(_data, _size);
        _data = std::exchange(other._data, nullptr);
        _size = std::exchange(other._size, 0);
      }
      return *this;
    }

    void PinnedBuffer::copy_from_device(const void* device, size_t bytes) {
      if (bytes > _size) {
        give_back(std::exchange(_data, nullptr), std::exchange(_size, 0));
        size_t size = 256;
        while (size < bytes)
          size *= 2;
        {
          const std::lock_guard<std::mutex> lock(pool_mutex);
          auto& blocks = free_blocks();
          if (auto it = blocks.find(size); it != blocks.end()) {
            _data = it->second;
            blocks.erase(it);
          }
        }
        if (!_data)
          CUDA_CHECK(cudaMallocHost(&_data, size));
        _size = size;
      }
      if (bytes > 0)
        CUDA_CHECK(cudaMemcpyAsync(_data, device, bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
    }

    void synchronize_stream() {
      CUDA_CHECK(cudaStreamSynchronize(get_cuda_stream()));
    }

  }
}

#endif
