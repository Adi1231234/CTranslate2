#ifndef CT2_USE_HIP

#include "cuda/pool_report.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static double gib(uint64_t bytes) {
      return double(bytes) / double(1ull << 30);
    }

    void start_pool_report(int device, cudaMemPool_t pool) {
      const int every = read_int_from_env("CT2_CUDA_POOL_REPORT_S", 0);
      if (every <= 0)
        return;
      std::thread([device, pool, every] {
        if (cudaSetDevice(device) != cudaSuccess)
          return;
        while (true) {
          std::this_thread::sleep_for(std::chrono::seconds(every));
          uint64_t used = 0, used_high = 0, reserved = 0, reserved_high = 0, zero = 0;
          size_t free_bytes = 0, total_bytes = 0;
          if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess
              || cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high) != cudaSuccess
              || cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) != cudaSuccess
              || cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_high) != cudaSuccess
              || cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
            return;
          // The highs are since the last report.
          cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero);
          cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero);
          std::fprintf(stderr, "POOL device %d: in use %.2f GiB (high %.2f), held %.2f GiB (high %.2f); "
                       "device free %.2f of %.2f GiB\n", device, gib(used), gib(used_high), gib(reserved),
                       gib(reserved_high), gib(free_bytes), gib(total_bytes));
          std::fflush(stderr);
        }
      }).detach();
    }

  }
}

#endif
