#pragma once

// Operand layouts of exact_attention.cuh: keys and values rearranged (data movement only) so that each warp's
// mma.sync m16n8k16 B fragments are one coalesced 8-byte load per lane (a plain [key][dim] layout spreads a
// fragment over 8 rows, i.e. 8 cache lines per load instruction). Word w of lane L = 4g + t is a pair of halves:
//   kf[b][tile][c][L] = (k[b][8 tile + g][16 c + 2t .. + 1], k[b][8 tile + g][16 c + 8 + 2t .. + 1]),
//                       c = 0..3 the 16-dim groups of the scores, keys >= n as zeros;
//   vf[b][dt][G][L]   = (v[b][s + 2t .. + 1][8 dt + g], v[b][s + 8 + 2t .. + 1][8 dt + g]), dt = 0..7 the output's
//                       8-dim tiles and G the output product's 16-key groups in cuBLAS's order: [0, 16), [16, r)
//                       with zeros from r = n % 64 (16 < r < 32), then s = r + 16 (G - 2); keys >= end as zeros.

#include <cstdint>
#include <cuda_fp16.h>

namespace at {
  namespace native {

    constexpr int eal_lanes = 32;

    __host__ __device__ inline int eal_key_tiles(int n) { return (n + 7) / 8; }
    __host__ __device__ inline int eal_groups(int n) { return 2 + (n - n % 64) / 16; }

    __device__ __forceinline__ unsigned eal_pack(__half lo, __half hi) {
      return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
    }

    // One thread per (batch entry, fragment, lane) of kf, then of vf.
    static __global__ void exact_attention_layout(const __half* k, const __half* v, uint2* kf, uint2* vf,
                                                  int batch, int n, int depth) {
      const int tiles = eal_key_tiles(n), groups = eal_groups(n), residue = n % 64;
      const size_t kf_count = (size_t)batch * tiles * 4 * eal_lanes;
      const size_t vf_count = (size_t)batch * (depth / 8) * groups * eal_lanes;
      const __half zero = __float2half(0.f);
      for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < kf_count + vf_count;
           o += (size_t)gridDim.x * blockDim.x) {
        if (o < kf_count) {
          const int lane = o % eal_lanes, c = (o / eal_lanes) % 4, tile = (o / (4 * eal_lanes)) % tiles;
          const size_t b = o / ((size_t)4 * eal_lanes * tiles);
          const int key = 8 * tile + lane / 4, d = 16 * c + 2 * (lane % 4);
          const __half* row = k + (b * n + key) * depth;
          kf[o] = key < n ? make_uint2(*reinterpret_cast<const unsigned*>(row + d),
                                       *reinterpret_cast<const unsigned*>(row + d + 8))
                          : make_uint2(0u, 0u);
        } else {
          const size_t p = o - kf_count;
          const int lane = p % eal_lanes, G = (p / eal_lanes) % groups, dt = (p / ((size_t)eal_lanes * groups)) % (depth / 8);
          const size_t b = p / ((size_t)eal_lanes * groups * (depth / 8));
          const int s = G < 2 ? 16 * G : residue + 16 * (G - 2), end = G < 2 ? residue : n;
          const int d = 8 * dt + lane / 4;
          auto at = [&](int key) { return key < end ? v[(b * n + key) * depth + d] : zero; };
          const int i = s + 2 * (lane % 4);
          vf[p] = make_uint2(eal_pack(at(i), at(i + 1)), eal_pack(at(i + 8), at(i + 9)));
        }
      }
    }

  }
}
