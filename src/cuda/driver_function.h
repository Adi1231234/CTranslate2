#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace cuda {

    // A driver API function as of the runtime this library is built with, from the driver in use; null when it
    // has none. The library links the runtime only.
    template <typename F>
    F driver_function(const char* name) {
      void* fn = nullptr;
      cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSymbolNotFound;
      const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &fn, CUDART_VERSION, cudaEnableDefault, &found);
      if (e != cudaSuccess || found != cudaDriverEntryPointSuccess) {
        spdlog::warn("No driver entry point {} (error {}, status {})", name, int(e), int(found));
        return nullptr;
      }
      return reinterpret_cast<F>(fn);
    }

  }
}
