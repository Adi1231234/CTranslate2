#pragma once

// Whisper encoder self-attention in one pass, o = SoftMax(MatMul(q, k^T, alpha)) v for each batch entry
// (clip x head), 1024 < n <= 2048 keys, 64 dims, with the arithmetic of the three ops it replaces on sm_120
// with cuBLAS 12.9.2, so o is bit for bit theirs (exact_attention_check.cu):
//   scores:  one mma.sync m16n8k16 chain over the 64 dims from zero, s = half(alpha * acc) (qk_hmma_probe.cu)
//   softmax: rows1024_row, the fork's replay of the legacy kernel's order
//   output:  one m16n8k16 chain per output over the keys, 16 at a time with the residue of the 64-key tiles
//            first ([0, 16), [16, r) + zeros, then from r = n % 64 on), o = half(acc) (av_hmma_probe.cu)
// A work item is 16 query rows of one batch entry; neither the scores nor the probabilities leave the block's
// shared memory. Its 8 warps compute the scores of 8-key tiles w, w + 8, ..., run the softmax in place on 2 rows
// each, then warp w chains the output's dims 8w..8w + 7. Keys and values come in exact_attention_layout.cuh's
// fragment order (one coalesced load per fragment); the queries are read where the EalSource says (the head-split
// tensor, or the fused projection plus its bias). A row is stored as rows1024_row reads it: lane L's 4-value
// slot s of the first 1024 values at (s * 33 + L) * 4, of the rest at ea_part2 + (s * tail_lanes + L) * 4, so
// softmax loads hit consecutive slots and, with a row pitch of 4 mod 64 halves, the output product's
// fragment loads (8 rows, 2 slots) hit distinct banks. A block computes one item, or (persistent,
// CT2_EA_BLOCKS=<blocks per SM>, cuda/persistent.h) the items it takes from the work counter.

#include "exact_attention_parts.cuh"
#include "cuda/persistent.cuh"

namespace at {
  namespace native {

    template <int N>
    __device__ __forceinline__ void ea_item(const EalSource& q, const uint2* kf, const uint2* vf, __half* o,
                                            int heads, float alpha, int row_tile, int entry, __half* s) {
      using S = ea_shape<N>;
      constexpr int m = N, n = N, tiles = S::tiles, groups = S::groups, tail_lanes = S::tail_lanes, pitch = S::pitch;
      const int warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE, g = lane / 4, t = lane % 4;
      const int j0 = row_tile * ea_rows;
      const uint2* kb = kf + (size_t)entry * tiles * 4 * C10_WARP_SIZE + lane;
      const uint2* vb = vf + ((size_t)entry * (ea_depth / 8) + warp) * groups * C10_WARP_SIZE + lane;
      unsigned a[4][4];                                          // the item's 16 queries, 4 groups of 16 dims
      #pragma unroll
      for (int c = 0; c < 4; ++c)
        for (int e = 0; e < 4; ++e) {
          const int j = j0 + g + 8 * (e % 2);
          a[c][e] = j < m ? eal_pair(q, eal_row(q, entry, heads, j), entry, heads, 16 * c + 8 * (e / 2) + 2 * t,
                                     ea_depth) : 0u;
        }
      // The warp's tiles are 8 apart, so its keys i are 64 apart: 8 halves further within a part of the row
      // (ea_slot), and at the one step where i enters the second part (kc) the place is that part's (no branch).
      const int i0 = 8 * warp + 2 * t, kc = (1024 - i0 + 63) / 64, jump = ea_slot(i0 + 64 * kc, tail_lanes);
      int place = ea_slot(i0, tail_lanes);
      #pragma unroll 4
      for (int k = 0, tile = warp; tile < tiles; ++k, tile += ea_warps) {  // scores of keys 8 tile .. 8 tile + 7
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < 4; ++c)                             // 16-dim groups in increasing order
          ea_mma(d, a[c][0], a[c][1], a[c][2], a[c][3], kb[(tile * 4 + c) * C10_WARP_SIZE]);
        if (i0 + 64 * k < n) {                                  // n % 4 == 0: keys i and i + 1 share a slot
          *reinterpret_cast<__half2*>(s + g * pitch + place) = __floats2half2_rn(alpha * d[0], alpha * d[1]);
          *reinterpret_cast<__half2*>(s + (g + 8) * pitch + place) = __floats2half2_rn(alpha * d[2], alpha * d[3]);
        }
        place = k + 1 == kc ? jump : place + 8;
      }
      __syncthreads();
      for (int r = warp; r < ea_rows; r += ea_warps) {         // softmax in place (rows past m are unused)
        __half* row = s + r * pitch;
        rows1024_row(row + 4 * lane, row + ea_part2 + 4 * lane, row + 4 * lane, row + ea_part2 + 4 * lane,
                     ea_lanes0, tail_lanes, n, lane);
      }
      __syncthreads();
      constexpr int residue = S::residue;
      const __half* s0 = s + g * pitch;                         // the fragment's rows g and g + 8
      const __half* s8 = s + (g + 8) * pitch;
      auto at = [](const __half* row, int off) { return *reinterpret_cast<const unsigned*>(row + off); };
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int G = 0; G < 2; ++G) {                            // vf holds zeros past a residue group's end
        const int lo = ea_slot(16 * G + 2 * t, tail_lanes), hi = ea_slot(16 * G + 8 + 2 * t, tail_lanes);
        ea_mma(acc, at(s0, lo), at(s8, lo), at(s0, hi), at(s8, hi), vb[G * C10_WARP_SIZE]);
      }
      // From group 2 on, groups G and G + 2 read keys 32 apart: within a part of the row (ea_slot) that is
      // 4 halves further, and at the one step where a key enters the second part (kc) its place is that part's,
      // selected without a branch (same loads, fewer instructions).
      int off[2][2], jump_to[2][2], k_in[2][2];                 // [G parity][keys i, i + 8]
      #pragma unroll
      for (int p = 0; p < 2; ++p)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int key0 = residue + 16 * p + 8 * h + 2 * t;
          off[p][h] = ea_slot(key0, tail_lanes);
          k_in[p][h] = (1024 - key0 + 31) / 32;
          jump_to[p][h] = ea_slot(key0 + 32 * k_in[p][h], tail_lanes);
        }
      #pragma unroll 2
      for (int k = 0; 2 + 2 * k < groups; ++k) {
        #pragma unroll
        for (int p = 0; p < 2; ++p)
          if (2 + 2 * k + p < groups)
            ea_mma(acc, at(s0, off[p][0]), at(s8, off[p][0]), at(s0, off[p][1]), at(s8, off[p][1]),
                   vb[(2 + 2 * k + p) * C10_WARP_SIZE]);
        #pragma unroll
        for (int p = 0; p < 2; ++p)
          #pragma unroll
          for (int h = 0; h < 2; ++h)
            off[p][h] = k + 1 == k_in[p][h] ? jump_to[p][h] : off[p][h] + 4;
      }
      const int clip = entry / heads, head = entry % heads;     // o is [clip, query, head, dim]
      for (int h = 0; h < 2; ++h)
        if (j0 + g + 8 * h < m)
          *reinterpret_cast<__half2*>(o + (((size_t)clip * m + j0 + g + 8 * h) * heads + head) * ea_depth
                                      + 8 * warp + 2 * t) = __floats2half2_rn(acc[2 * h], acc[2 * h + 1]);
    }

    // Without a counter, block (x, y) computes row tile x of entry y; with one, items row_tile + row_tiles * entry.
    template <int N>
    __global__ void __launch_bounds__(ea_warps * C10_WARP_SIZE)
    exact_attention_kernel(const EalSource q, const uint2* kf, const uint2* vf, __half* o, int heads, float alpha,
                           unsigned* counter, int batch) {
      extern __shared__ __align__(16) unsigned char ea_smem[];
      __half* s = reinterpret_cast<__half*>(ea_smem);           // [ea_rows][pitch] scores, then probabilities
      constexpr int row_tiles = ea_shape<N>::row_tiles;
      if (!counter) {
        ea_item<N>(q, kf, vf, o, heads, alpha, blockIdx.x, blockIdx.y, s);
        return;
      }
      __shared__ int slot;
      const int items = row_tiles * batch;
      for (int item = ctranslate2::cuda::next_work_item(counter, items, slot); item < items;
           item = ctranslate2::cuda::next_work_item(counter, items, slot))
        ea_item<N>(q, kf, vf, o, heads, alpha, item % row_tiles, item / row_tiles, s);
    }

  }
}
