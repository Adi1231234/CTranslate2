#pragma once

// ladder_cross.cuh's shapes, the rows' list and device helpers.

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/partial_sums.cuh"

namespace ctranslate2 {
  namespace cuda {

    constexpr int lc_keys = 1500, lc_depth = 64, lc_max_rows = 32;

    struct LadderRows {
      int count;                             // rows listed
      int8_t row[lc_max_rows];               // the rows (indices into the queries and outputs)
      int8_t group[lc_max_rows];             // each one's group's rows: its cuBLAS call has group x heads entries
    };

    static __device__ __forceinline__ float hf(__half x) {
      return __half2float(x);
    }

    static __device__ __forceinline__ unsigned pair(const __half* p) {   // p[0], p[1] (4-byte aligned)
      return *reinterpret_cast<const unsigned*>(p);
    }

  }
}
