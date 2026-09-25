#pragma once

// Operand layouts of exact_attention.cuh: keys and values rearranged (data movement, and where the operands come
// from the fused projection, its bias add) so that each warp's
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

    // Where a q, k or v operand is read: row j (a query or key) of batch entry b = clip * heads + head, dim d at
    // p[clip * clip_stride + head * head_stride + j * row_stride + d], plus bias[head * depth + d] when there is a
    // bias, with Dense's fp16 bias add (__hadd, which is cuda::plus<__half>). A head-split tensor [batch, n, depth]
    // has no bias; the fused projection [clips, n, 3 * heads * depth] without its bias gives each part with its
    // part of the bias, so the values equal Dense + split_heads + Split (split_heads.cu) bit for bit.
    struct EalSource {
      const __half* p;
      const __half* bias;
      long long clip_stride, head_stride, row_stride;
    };

    __host__ __device__ inline EalSource eal_split(const __half* x, int heads, int n, int depth) {  // [batch, n, depth]
      return EalSource{x, nullptr, (long long)heads * n * depth, (long long)n * depth, depth};
    }

    __device__ __forceinline__ const __half* eal_row(const EalSource& s, int b, int heads, int j) {
      const int clip = b / heads;
      return s.p + clip * s.clip_stride + (b - clip * heads) * s.head_stride + (long long)j * s.row_stride;
    }

    // Dims d, d + 1 of a row as one word, with their bias added.
    __device__ __forceinline__ unsigned eal_pair(const EalSource& s, const __half* row, int b, int heads,
                                                 int d, int depth) {
      __half2 x = *reinterpret_cast<const __half2*>(row + d);
      if (s.bias)
        x = __hadd2(*reinterpret_cast<const __half2*>(s.bias + (b % heads) * depth + d), x);
      return *reinterpret_cast<const unsigned*>(&x);
    }

    // One thread per (batch entry, fragment, lane) of kf, then of vf.
    static __global__ void exact_attention_layout(EalSource k, EalSource v, int heads, uint2* kf, uint2* vf,
                                                  int batch, int n, int depth) {
      const int tiles = eal_key_tiles(n), groups = eal_groups(n), residue = n % 64;
      const size_t kf_count = (size_t)batch * tiles * 4 * eal_lanes;
      const size_t vf_count = (size_t)batch * (depth / 8) * groups * eal_lanes;
      const __half zero = __float2half(0.f);
      for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < kf_count + vf_count;
           o += (size_t)gridDim.x * blockDim.x) {
        if (o < kf_count) {
          const int lane = o % eal_lanes, c = (o / eal_lanes) % 4, tile = (o / (4 * eal_lanes)) % tiles;
          const int b = int(o / ((size_t)4 * eal_lanes * tiles));
          const int key = 8 * tile + lane / 4, d = 16 * c + 2 * (lane % 4);
          const __half* row = eal_row(k, b, heads, key);
          kf[o] = key < n ? make_uint2(eal_pair(k, row, b, heads, d, depth), eal_pair(k, row, b, heads, d + 8, depth))
                          : make_uint2(0u, 0u);
        } else {
          const size_t p = o - kf_count;
          const int lane = p % eal_lanes, G = (p / eal_lanes) % groups, dt = (p / ((size_t)eal_lanes * groups)) % (depth / 8);
          const int b = int(p / ((size_t)eal_lanes * groups * (depth / 8)));
          const int s = G < 2 ? 16 * G : residue + 16 * (G - 2), end = G < 2 ? residue : n;
          const int d = 8 * dt + lane / 4;
          const __half vb = v.bias ? v.bias[(b % heads) * depth + d] : zero;
          auto at = [&](int key) {
            if (key >= end)
              return zero;
            const __half x = eal_row(v, b, heads, key)[d];
            return v.bias ? __hadd(vb, x) : x;
          };
          const int i = s + 2 * (lane % 4);
          vf[p] = make_uint2(eal_pack(at(i), at(i + 1)), eal_pack(at(i + 8), at(i + 9)));
        }
      }
    }

  }
}
