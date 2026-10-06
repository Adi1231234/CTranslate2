#pragma once

#include <cstddef>

namespace ctranslate2 {
  namespace cuda {

    // Page-locked host memory for a device-to-host copy that does not wait for the device: a copy into pageable
    // memory returns only when the device has done it, so each such read on the host waits for all the work queued
    // before it. Blocks come from a pool shared by the process and go back to it with their PinnedBuffer (pinning
    // memory costs far more than a copy). The holder reads the bytes once the stream that copies them has
    // synchronized (synchronize_stream).
    class PinnedBuffer {
    public:
      PinnedBuffer() = default;
      ~PinnedBuffer();
      PinnedBuffer(const PinnedBuffer&) = delete;
      PinnedBuffer& operator=(const PinnedBuffer&) = delete;
      PinnedBuffer(PinnedBuffer&& other) noexcept;
      PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

      // Queues the copy of `bytes` from device memory on the calling thread's stream (the buffer grows to hold them).
      void copy_from_device(const void* device, size_t bytes);
      const void* data() const {
        return _data;
      }

    private:
      void* _data = nullptr;
      size_t _size = 0;
    };

    // Waits for the calling thread's stream.
    void synchronize_stream();

  }
}
