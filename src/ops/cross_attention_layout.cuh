#pragma once

// Operand layouts of cross_attention.cuh: the Whisper decoder's cross-attention keys and values (1500 encoder
// positions, 64 dims per head), rearranged once per batch (data movement only) so that every mma.sync m16n8k16
// A fragment is one coalesced 16-byte load per lane. Lane L = 4g + t holds registers a0..a3 of the standard
// A layout (rows g and g + 8, k pairs 2t and 2t + 8):
//   kf[e][T][c][L] = (K[16T + g][16c + 2t, +1], K[16T + g + 8][16c + 2t, +1],
//                     K[16T + g][16c + 8 + 2t, +1], K[16T + g + 8][16c + 8 + 2t, +1]),  keys >= 1500 as zeros;
//   vf[e][D][G][L] = the same with rows = dims 16D + g, 16D + g + 8 and k = the keys of group G:
//                    (V[s + 2t][16D + g], V[s + 2t + 1][16D + g]), ... with s = ca_group_start(G), keys past
//                    the group's end as zeros (cuBLAS's key order, see cross_attention.cuh).
// Both are [entries][1504][64] halves, the size of a 1504-key cache.

#include <cstdint>
#include <cuda_fp16.h>

namespace at {
  namespace native {

    constexpr int ca_keys = 1500, ca_tiles = 94, ca_groups = 94, ca_depth = 64, ca_residue = ca_keys % 32;

    // The keys of the output product's 16-key group G in cuBLAS's order: the k tile's residue first ([0, 16),
    // then [16, 28) and zeros), then 16 at a time from 28.
    __host__ __device__ __forceinline__ int ca_group_start(int G) {
      return G < 2 ? 16 * G : ca_residue + 16 * (G - 2);
    }
    __host__ __device__ __forceinline__ int ca_group_end(int G) {
      return G < 2 ? ca_residue : ca_keys;
    }

    __device__ __forceinline__ unsigned ca_pair(const __half* row, int d, bool valid) {
      return valid ? *reinterpret_cast<const unsigned*>(row + d) : 0u;
    }

    __device__ __forceinline__ unsigned ca_pack(__half lo, __half hi) {
      return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
    }

    // One thread per 16-byte fragment word of kf, then of vf. k, v: [entries][1500][64].
    static __global__ void cross_attention_layout(const __half* k, const __half* v, uint4* kf, uint4* vf,
                                                  size_t entries) {
      const size_t per = (size_t)ca_tiles * 4 * 32, count = entries * per;
      const __half zero = __float2half(0.f);
      for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < 2 * count;
           o += (size_t)gridDim.x * blockDim.x) {
        const size_t p = o < count ? o : o - count, e = p / per;
        const int lane = p % 32, g = lane / 4, t = lane % 4;
        if (o < count) {
          const int c = (p / 32) % 4, T = (p / 128) % ca_tiles, r0 = 16 * T + g, r1 = r0 + 8, d = 16 * c + 2 * t;
          const __half* ke = k + e * ca_keys * ca_depth;
          kf[p] = make_uint4(ca_pair(ke + (size_t)r0 * ca_depth, d, r0 < ca_keys),
                             ca_pair(ke + (size_t)r1 * ca_depth, d, r1 < ca_keys),
                             ca_pair(ke + (size_t)r0 * ca_depth, d + 8, r0 < ca_keys),
                             ca_pair(ke + (size_t)r1 * ca_depth, d + 8, r1 < ca_keys));
        } else {
          const int G = (p / 32) % ca_groups, D = (p / (32 * ca_groups)) % 4;
          const int s = ca_group_start(G) + 2 * t, end = ca_group_end(G), d0 = 16 * D + g, d1 = d0 + 8;
          const __half* ve = v + e * ca_keys * ca_depth;
          auto at = [&](int key, int dim) { return key < end ? ve[(size_t)key * ca_depth + dim] : zero; };
          vf[p] = make_uint4(ca_pack(at(s, d0), at(s + 1, d0)), ca_pack(at(s, d1), at(s + 1, d1)),
                             ca_pack(at(s + 8, d0), at(s + 9, d0)), ca_pack(at(s + 8, d1), at(s + 9, d1)));
        }
      }
    }

  }
}
