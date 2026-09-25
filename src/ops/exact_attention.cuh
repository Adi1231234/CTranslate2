#pragma once

// Whisper encoder self-attention in one pass, o = SoftMax(MatMul(q, k^T, alpha)) v for each batch entry
// (clip x head), 1024 < n <= 2048 keys, 64 dims, with the arithmetic of the three ops it replaces on sm_120
// with cuBLAS 12.9.2, so o is bit for bit theirs (exact_attention_check.cu):
//   scores:  one mma.sync m16n8k16 chain over the 64 dims from zero, s = half(alpha * acc) (qk_hmma_probe.cu)
//   softmax: rows1024_row, the fork's replay of the legacy kernel's order
//   output:  one m16n8k16 chain per output over the keys, 16 at a time with the residue of the 64-key tiles
//            first ([0, 16), [16, r) + zeros, then from r = n % 64 on), o = half(acc) (av_hmma_probe.cu)
// A block owns 16 query rows of one batch entry; neither the scores nor the probabilities leave its shared
// memory. Its 8 warps compute the scores of 8-key tiles w, w + 8, ..., run the softmax in place on 2 rows
// each, then warp w chains the output's dims 8w..8w + 7. Keys and values come in exact_attention_layout.cuh's
// fragment order (one coalesced load per fragment). A row is stored as rows1024_row reads it: lane L's 4-value
// slot s of the first 1024 values at (s * 33 + L) * 4, of the rest at ea_part2 + (s * tail_lanes + L) * 4, so
// softmax loads hit consecutive slots and, with a row pitch of 4 mod 64 halves, the output product's
// fragment loads (8 rows, 2 slots) hit distinct banks.

#include "softmax_kernels.cuh"
#include "exact_attention_layout.cuh"

namespace at {
  namespace native {

    constexpr int ea_warps = 8, ea_rows = 16, ea_depth = 64, ea_max_cols = 2048;
    constexpr int ea_lanes0 = 33, ea_part2 = 8 * ea_lanes0 * 4;   // slot stride and size of the first 1024 values

    __device__ __forceinline__ int ea_slot(int i, int tail_lanes) {   // key i -> its place in a stored row
      const int part = i >= 1024, j = i - 1024 * part, lanes = part ? tail_lanes : ea_lanes0;
      return ea_part2 * part + ((j % 32) / 4 * lanes + j / 32) * 4 + j % 4;
    }

    __device__ __forceinline__ void ea_mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint2 b) {
#if __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
#endif
    }

    static __global__ void __launch_bounds__(ea_warps * C10_WARP_SIZE)
    exact_attention_kernel(const __half* q, const uint2* kf, const uint2* vf, __half* o, int m, int n, int heads,
                           int pitch, int tail_lanes, float alpha) {
      extern __shared__ __align__(16) unsigned char ea_smem[];
      __half* s = reinterpret_cast<__half*>(ea_smem);           // [ea_rows][pitch] scores, then probabilities
      const int warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE, g = lane / 4, t = lane % 4;
      const int j0 = blockIdx.x * ea_rows, tiles = eal_key_tiles(n), groups = eal_groups(n);
      const __half* qb = q + (size_t)blockIdx.y * m * ea_depth;
      const uint2* kb = kf + (size_t)blockIdx.y * tiles * 4 * C10_WARP_SIZE + lane;
      const uint2* vb = vf + ((size_t)blockIdx.y * (ea_depth / 8) + warp) * groups * C10_WARP_SIZE + lane;
      unsigned a[4][4];                                          // the block's 16 queries, 4 groups of 16 dims
      #pragma unroll
      for (int c = 0; c < 4; ++c)
        for (int e = 0; e < 4; ++e) {
          const int j = j0 + g + 8 * (e % 2);
          a[c][e] = j < m ? *reinterpret_cast<const unsigned*>(qb + (size_t)j * ea_depth + 16 * c + 8 * (e / 2) + 2 * t) : 0u;
        }
      #pragma unroll 4
      for (int tile = warp; tile < tiles; tile += ea_warps) {  // scores of keys 8 tile .. 8 tile + 7
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < 4; ++c)                             // 16-dim groups in increasing order
          ea_mma(d, a[c][0], a[c][1], a[c][2], a[c][3], kb[(tile * 4 + c) * C10_WARP_SIZE]);
        const int i = 8 * tile + 2 * t;
        if (i < n) {                                            // n % 4 == 0: keys i and i + 1 share a slot
          const int at = ea_slot(i, tail_lanes);
          *reinterpret_cast<__half2*>(s + g * pitch + at) = __floats2half2_rn(alpha * d[0], alpha * d[1]);
          *reinterpret_cast<__half2*>(s + (g + 8) * pitch + at) = __floats2half2_rn(alpha * d[2], alpha * d[3]);
        }
      }
      __syncthreads();
      for (int r = warp; r < ea_rows; r += ea_warps) {         // softmax in place (rows past m are unused)
        __half* row = s + r * pitch;
        rows1024_row(row + 4 * lane, row + ea_part2 + 4 * lane, row + 4 * lane, row + ea_part2 + 4 * lane,
                     ea_lanes0, tail_lanes, n, lane);
      }
      __syncthreads();
      const int residue = n % 64;
      const __half* s0 = s + g * pitch;                         // the fragment's rows g and g + 8
      const __half* s8 = s + (g + 8) * pitch;
      auto at = [](const __half* row, int off) { return *reinterpret_cast<const unsigned*>(row + off); };
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int G = 0; G < 2; ++G) {                            // vf holds zeros past a residue group's end
        const int lo = ea_slot(16 * G + 2 * t, tail_lanes), hi = ea_slot(16 * G + 8 + 2 * t, tail_lanes);
        ea_mma(acc, at(s0, lo), at(s8, lo), at(s0, hi), at(s8, hi), vb[G * C10_WARP_SIZE]);
      }
      // From group 2 on, groups G and G + 2 read keys 32 apart: within a part of the row (ea_slot) that is
      // 4 halves further, so the places are stepped instead of recomputed (same loads, fewer instructions).
      int key[2][2], off[2][2];                                 // [G parity][keys i, i + 8]
      #pragma unroll
      for (int p = 0; p < 2; ++p)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          key[p][h] = residue + 16 * p + 8 * h + 2 * t;
          off[p][h] = ea_slot(key[p][h], tail_lanes);
        }
      #pragma unroll 2
      for (int G = 2; G < groups; G += 2) {
        #pragma unroll
        for (int p = 0; p < 2; ++p)
          if (G + p < groups)
            ea_mma(acc, at(s0, off[p][0]), at(s8, off[p][0]), at(s0, off[p][1]), at(s8, off[p][1]),
                   vb[(G + p) * C10_WARP_SIZE]);
        #pragma unroll
        for (int p = 0; p < 2; ++p)
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            key[p][h] += 32;
            off[p][h] = key[p][h] >= 1024 && key[p][h] < 1056 ? ea_slot(key[p][h], tail_lanes) : off[p][h] + 4;
          }
      }
      const int clip = blockIdx.y / heads, head = blockIdx.y % heads;  // o is [clip, query, head, dim]
      for (int h = 0; h < 2; ++h)
        if (j0 + g + 8 * h < m)
          *reinterpret_cast<__half2*>(o + (((size_t)clip * m + j0 + g + 8 * h) * heads + head) * ea_depth
                                      + 8 * warp + 2 * t) = __floats2half2_rn(acc[2 * h], acc[2 * h + 1]);
    }

    // Bytes of the kf and vf workspace for batch entries of n keys.
    inline size_t exact_attention_workspace(int batch, int n) {
      return sizeof (uint2) * batch * (eal_key_tiles(n) * 4 + (ea_depth / 8) * eal_groups(n)) * eal_lanes;
    }

    // o = SoftMax(q k^T * alpha) v for q [batch, m, 64], k and v [batch, n, 64] with batch = clips x heads, o
    // [clips, m, heads, 64] (the heads combined, as MultiHeadAttention::combine_heads would lay them out; heads 1
    // gives [batch, m, 64]); workspace: exact_attention_workspace(batch, n) bytes. 1024 < n <= 2048,
    // 16 < n % 64 < 32, all 4-byte aligned.
    inline void exact_attention(const __half* q, const __half* k, const __half* v, void* workspace, __half* o,
                                int batch, int heads, int m, int n, float alpha, cudaStream_t stream) {
      static const bool configured = cudaFuncSetAttribute(exact_attention_kernel,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          int(ea_rows * (ea_max_cols + 128) * sizeof (__half))) == cudaSuccess;
      (void)configured;
      uint2* kf = static_cast<uint2*>(workspace);
      uint2* vf = kf + (size_t)batch * eal_key_tiles(n) * 4 * eal_lanes;
      exact_attention_layout<<<1024, 256, 0, stream>>>(k, v, kf, vf, batch, n, ea_depth);
      const int tail_lanes = (n - 1024 + C10_WARP_SIZE - 1) / C10_WARP_SIZE;
      const int pitch = (ea_part2 + tail_lanes * C10_WARP_SIZE + 59) / 64 * 64 + 4;   // halves per row, 4 mod 64
      exact_attention_kernel<<<dim3((m + ea_rows - 1) / ea_rows, batch), ea_warps * C10_WARP_SIZE,
                               ea_rows * pitch * sizeof (__half), stream>>>(q, kf, vf, o, m, n, heads, pitch,
                                                                            tail_lanes, alpha);
    }

  }
}
