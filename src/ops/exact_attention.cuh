#pragma once

// Whisper encoder self-attention in one pass, o = SoftMax(MatMul(q, k^T, alpha)) v for each batch entry
// (clip x head), 1024 < n <= 2048 keys, 64 dims, with the arithmetic of the three ops it replaces on sm_120
// with cuBLAS 12.9.2, so o is bit for bit theirs (exact_attention_check.cu):
//   scores:  one mma.sync m16n8k16 chain over the 64 dims from zero, s = half(alpha * acc) (qk_hmma_probe.cu)
//   softmax: rows1024_row, the fork's replay of the legacy kernel's order
//   output:  one m16n8k16 chain per output over the keys, 16 at a time with the residue of the 64-key tiles
//            first ([0, 16), [16, r) + zeros, then from r = n % 64 on), o = half(acc) (av_hmma_probe.cu)
// A block owns 16 query rows of one batch entry; neither the scores nor the probabilities leave its shared
// memory. Its 8 warps compute the scores of 8-key tiles w, w + 8, ... (keys from k, queries from q), run the
// softmax in place on 2 rows each, then warp w chains the output's dims 8w..8w + 7 (values from vt, the
// transposed v). A row is stored as rows1024_row reads it: lane L's 4-value slot s of the first 1024 values
// at (s * 33 + L) * 4, of the rest at ea_part2 + (s * tail_lanes + L) * 4, so the 32 lanes of a softmax load
// hit consecutive slots; the stride 33 and a row pitch of 4 mod 64 halves also keep the output product's
// fragment loads (8 rows, 2 slots) on distinct banks.

#include "softmax_kernels.cuh"

namespace at {
  namespace native {

    constexpr int ea_warps = 8, ea_rows = 16, ea_depth = 64, ea_max_cols = 2048;
    constexpr int ea_lanes0 = 33, ea_part2 = 8 * ea_lanes0 * 4;   // slot stride and size of the first 1024 values

    __device__ __forceinline__ unsigned ea_pair(const __half* p, size_t stride, int rows, int r, int c) {
      return r < rows ? *reinterpret_cast<const unsigned*>(p + r * stride + c) : 0u;
    }

    __device__ __forceinline__ int ea_slot(int i, int tail_lanes) {   // key i -> its place in a stored row
      const int part = i >= 1024, j = i - 1024 * part, lanes = part ? tail_lanes : ea_lanes0;
      return ea_part2 * part + ((j % 32) / 4 * lanes + j / 32) * 4 + j % 4;
    }

    __device__ __forceinline__ void ea_mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                           unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
    }

    static __global__ void __launch_bounds__(ea_warps * C10_WARP_SIZE)
    exact_attention_kernel(const __half* q, const __half* k, const __half* vt, __half* o, int m, int n,
                           int pitch, int tail_lanes, float alpha) {
      extern __shared__ __align__(16) unsigned char ea_smem[];
      __half* s = reinterpret_cast<__half*>(ea_smem);           // [ea_rows][pitch] scores, then probabilities
      const int warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE, g = lane / 4, t = lane % 4;
      const int j0 = blockIdx.x * ea_rows;
      const __half* qb = q + (size_t)blockIdx.y * m * ea_depth;
      const __half* kb = k + (size_t)blockIdx.y * n * ea_depth;
      const __half* vb = vt + (size_t)blockIdx.y * ea_depth * n;
      unsigned a[4][4];                                          // the block's 16 queries, 4 groups of 16 dims
      #pragma unroll
      for (int c = 0; c < 4; ++c)
        for (int e = 0; e < 4; ++e)
          a[c][e] = ea_pair(qb, ea_depth, m, j0 + g + 8 * (e % 2), 16 * c + 8 * (e / 2) + 2 * t);
      #pragma unroll 4
      for (int i0 = warp * 8; i0 < n; i0 += ea_warps * 8) {     // scores of keys i0..i0 + 7
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < 4; ++c)                             // 16-dim groups in increasing order
          ea_mma(d, a[c][0], a[c][1], a[c][2], a[c][3], ea_pair(kb, ea_depth, n, i0 + g, 16 * c + 2 * t),
                 ea_pair(kb, ea_depth, n, i0 + g, 16 * c + 8 + 2 * t));
        const int i = i0 + 2 * t;
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
      const int d0 = warp * 8, residue = n % 64;
      auto p_at = [&](int r, int i, int end) {                   // probabilities of keys i, i + 1 of row r
        return i < end ? *reinterpret_cast<const unsigned*>(s + r * pitch + ea_slot(i, tail_lanes)) : 0u;
      };
      auto group = [&](float* acc, int i0, int end) {           // keys i0..i0 + 15, zero at or past end
        ea_mma(acc, p_at(g, i0 + 2 * t, end), p_at(g + 8, i0 + 2 * t, end), p_at(g, i0 + 8 + 2 * t, end),
               p_at(g + 8, i0 + 8 + 2 * t, end), i0 + 2 * t < end ? ea_pair(vb, n, ea_depth, d0 + g, i0 + 2 * t) : 0u,
               i0 + 8 + 2 * t < end ? ea_pair(vb, n, ea_depth, d0 + g, i0 + 8 + 2 * t) : 0u);
      };
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (residue) {
        group(acc, 0, residue);
        if (residue > 16)
          group(acc, 16, residue);
      }
      #pragma unroll 4
      for (int i0 = residue; i0 < n; i0 += 16) {               // every key present: plain loads, prefetchable
        ea_mma(acc, *reinterpret_cast<const unsigned*>(s + g * pitch + ea_slot(i0 + 2 * t, tail_lanes)),
               *reinterpret_cast<const unsigned*>(s + (g + 8) * pitch + ea_slot(i0 + 2 * t, tail_lanes)),
               *reinterpret_cast<const unsigned*>(s + g * pitch + ea_slot(i0 + 8 + 2 * t, tail_lanes)),
               *reinterpret_cast<const unsigned*>(s + (g + 8) * pitch + ea_slot(i0 + 8 + 2 * t, tail_lanes)),
               *reinterpret_cast<const unsigned*>(vb + (size_t)(d0 + g) * n + i0 + 2 * t),
               *reinterpret_cast<const unsigned*>(vb + (size_t)(d0 + g) * n + i0 + 8 + 2 * t));
      }
      for (int h = 0; h < 2; ++h)
        if (j0 + g + 8 * h < m)
          *reinterpret_cast<__half2*>(o + ((size_t)blockIdx.y * m + j0 + g + 8 * h) * ea_depth + d0 + 2 * t) =
            __floats2half2_rn(acc[2 * h], acc[2 * h + 1]);
    }

    // vt[b][d][i] = v[b][i][d] (64 dims), through 32 x 32 tiles of shared memory: data movement only.
    static __global__ void ea_transpose_values(const __half* v, __half* vt, int n) {
      __shared__ __half tile[32][33];
      const __half* vb = v + (size_t)blockIdx.z * n * ea_depth;
      __half* tb = vt + (size_t)blockIdx.z * ea_depth * n;
      const int i0 = blockIdx.x * 32, d0 = blockIdx.y * 32;
      for (int r = threadIdx.y; r < 32; r += blockDim.y)
        if (i0 + r < n)
          tile[r][threadIdx.x] = vb[(size_t)(i0 + r) * ea_depth + d0 + threadIdx.x];
      __syncthreads();
      for (int r = threadIdx.y; r < 32; r += blockDim.y)
        if (i0 + threadIdx.x < n)
          tb[(size_t)(d0 + r) * n + i0 + threadIdx.x] = tile[threadIdx.x][r];
    }

    // o = SoftMax(q k^T * alpha) v for q [batch, m, 64], k and v [batch, n, 64], o [batch, m, 64]; vt is a
    // [batch, 64, n] workspace. 1024 < n <= 2048, n % 4 == 0, q, k, v and o 4-byte aligned.
    inline void exact_attention(const __half* q, const __half* k, const __half* v, __half* vt, __half* o,
                                int batch, int m, int n, float alpha, cudaStream_t stream) {
      static const bool configured = cudaFuncSetAttribute(exact_attention_kernel,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          int(ea_rows * (ea_max_cols + 128) * sizeof (__half))) == cudaSuccess;
      (void)configured;
      ea_transpose_values<<<dim3((n + 31) / 32, ea_depth / 32, batch), dim3(32, 8), 0, stream>>>(v, vt, n);
      const int tail_lanes = (n - 1024 + C10_WARP_SIZE - 1) / C10_WARP_SIZE;
      const int pitch = (ea_part2 + tail_lanes * C10_WARP_SIZE + 59) / 64 * 64 + 4;   // halves per row, 4 mod 64
      exact_attention_kernel<<<dim3((m + ea_rows - 1) / ea_rows, batch), ea_warps * C10_WARP_SIZE,
                               ea_rows * pitch * sizeof (__half), stream>>>(q, k, vt, o, m, n, pitch, tail_lanes, alpha);
    }

  }
}
