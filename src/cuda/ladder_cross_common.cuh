#pragma once

// ladder_cross.cuh's shapes, the rows' lists and device helpers.

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/partial_sums.cuh"

namespace ctranslate2 {
  namespace cuda {

    constexpr int lc_keys = 1500, lc_depth = 64;
    constexpr int lc_part_rows = 32;                         // a ladder's rows at most
    constexpr int lc_max_parts = 8, lc_max_rows = 128;      // a launch's ladders and rows at most

    // The ladders of one launch: part p's rows are the queries', scores' and outputs' rows from row_begin[p], against
    // its clip's memory keys k[p] and values v[p] ([heads][1500][64] each).
    struct LcParts {
      int count;
      const __half* k[lc_max_parts];
      const __half* v[lc_max_parts];
      int row_begin[lc_max_parts];
    };

    struct LadderRows {
      int count;                             // rows listed
      int8_t row[lc_max_rows];               // each one's row in its part
      int8_t group[lc_max_rows];             // its group's rows: its cuBLAS call has group x heads entries
      int8_t part[lc_max_rows];              // its part
      int16_t part_begin[lc_max_parts + 1];  // part p's rows in the list: [part_begin[p], part_begin[p + 1])
    };

    // Part p's entry of a launch's per-part array, picked over constant indices (a parameter array indexed by a
    // computed value is copied to local memory by every thread).
    template <typename T>
    static __device__ __forceinline__ T lc_pick(const T (&a)[lc_max_parts], int p) {
      T x = a[0];
      #pragma unroll
      for (int q = 1; q < lc_max_parts; ++q)
        if (q == p)
          x = a[q];
      return x;
    }

    static __device__ __forceinline__ float hf(__half x) {
      return __half2float(x);
    }

    static __device__ __forceinline__ unsigned pair(const __half* p) {   // p[0], p[1] (4-byte aligned)
      return *reinterpret_cast<const unsigned*>(p);
    }

  }
}
