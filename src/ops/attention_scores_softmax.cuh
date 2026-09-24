#pragma once

// Whisper encoder self-attention scores and their softmax in one pass: for each batch entry b (clip x head)
// and query row j, p[b][j][i] = softmax_i(half(alpha * dot(q[b][j][0:64], k[b][i][0:64]))), 1024 < n <= 2048.
// The scores take cuBLAS's arithmetic for this MatMul on sm_120 (one mma.sync m16n8k16 chain over the 64
// dims from zero, then half(alpha * acc): tools/turing/kernels/qk_hmma_probe.cu) and the softmax is
// rows1024_row's, so p is MatMul(trans_b, alpha) + SoftMax bit for bit (scores_softmax_check.cu), while the
// scores themselves never leave shared memory (a 1500 x 1500 head writes and reads back 4.5 MB less).
// A block owns 16 query rows of one batch entry: its 8 warps compute the scores of 8-key tiles w, w + 8, ...
// into shared memory, then run the softmax on 2 rows each. A row is stored as rows1024_row reads it: lane
// L's 4-value slot q of the first 1024 values at (q * 32 + L) * 4, of the rest at 1024 + (q * tail_lanes +
// L) * 4, so the 32 lanes of a softmax load hit 32 consecutive 8-byte slots.

#include "softmax_kernels.cuh"

namespace at {
  namespace native {

    constexpr int ass_warps = 8, ass_rows = 16, ass_depth = 64, ass_max_cols = 2048;

    __device__ __forceinline__ unsigned ass_pair(const __half* p, int rows, int r, int c) {
      return r < rows ? *reinterpret_cast<const unsigned*>(p + (size_t)r * ass_depth + c) : 0u;
    }

    __device__ __forceinline__ int ass_slot(int i, int tail_lanes) {  // key i -> its place in a stored row
      const int part = i >= 1024, j = i - 1024 * part, lanes = part ? tail_lanes : C10_WARP_SIZE;
      return 1024 * part + ((j % 32) / 4 * lanes + j / 32) * 4 + j % 4;
    }

    static __global__ void __launch_bounds__(ass_warps * C10_WARP_SIZE)
    attention_scores_softmax_kernel(const __half* q, const __half* k, __half* p, int m, int n, int pitch,
                                    int tail_lanes, float alpha) {
#if __CUDA_ARCH__ >= 800                                     // mma.sync m16n8k16: sm_80 and newer
      extern __shared__ __align__(16) unsigned char ass_smem[];
      __half* s = reinterpret_cast<__half*>(ass_smem);          // [ass_rows][pitch] scores
      const int warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE, g = lane / 4, t = lane % 4;
      const int j0 = blockIdx.x * ass_rows;
      const __half* qb = q + (size_t)blockIdx.y * m * ass_depth;
      const __half* kb = k + (size_t)blockIdx.y * n * ass_depth;
      unsigned a[4][4];                                          // the block's 16 query rows, 4 groups of 16 dims
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        a[kk][0] = ass_pair(qb, m, j0 + g, 16 * kk + 2 * t);
        a[kk][1] = ass_pair(qb, m, j0 + g + 8, 16 * kk + 2 * t);
        a[kk][2] = ass_pair(qb, m, j0 + g, 16 * kk + 8 + 2 * t);
        a[kk][3] = ass_pair(qb, m, j0 + g + 8, 16 * kk + 8 + 2 * t);
      }
      #pragma unroll 4
      for (int i0 = warp * 8; i0 < n; i0 += ass_warps * 8) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {                        // 16-dim groups in increasing order
          const unsigned b0 = ass_pair(kb, n, i0 + g, 16 * kk + 2 * t);
          const unsigned b1 = ass_pair(kb, n, i0 + g, 16 * kk + 8 + 2 * t);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                       : "r"(a[kk][0]), "r"(a[kk][1]), "r"(a[kk][2]), "r"(a[kk][3]), "r"(b0), "r"(b1));
        }
        const int i = i0 + 2 * t;
        if (i < n) {                                            // n % 4 == 0: keys i and i + 1 share a slot
          const int at = ass_slot(i, tail_lanes);
          *reinterpret_cast<__half2*>(s + g * pitch + at) = __floats2half2_rn(alpha * d[0], alpha * d[1]);
          *reinterpret_cast<__half2*>(s + (g + 8) * pitch + at) = __floats2half2_rn(alpha * d[2], alpha * d[3]);
        }
      }
      __syncthreads();
      for (int r = warp; r < ass_rows && j0 + r < m; r += ass_warps) {
        const __half* row = s + r * pitch;
        rows1024_row(p + ((size_t)blockIdx.y * m + j0 + r) * n, row + 4 * lane, C10_WARP_SIZE,
                     row + 1024 + 4 * lane, tail_lanes, n, lane);
      }
#endif
    }

    // p = SoftMax(MatMul(q, k^T, alpha)) for q [batch, m, 64], k [batch, n, 64], p [batch, m, n]:
    // 1024 < n <= 2048, n % 4 == 0, q and k 4-byte aligned, p 8-byte aligned.
    inline void attention_scores_softmax(const __half* q, const __half* k, __half* p, int batch, int m, int n,
                                         float alpha, cudaStream_t stream) {
      static const bool configured = cudaFuncSetAttribute(attention_scores_softmax_kernel,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          int(ass_rows * ass_max_cols * sizeof (__half))) == cudaSuccess;
      (void)configured;
      const int tail_lanes = (n - 1024 + C10_WARP_SIZE - 1) / C10_WARP_SIZE;
      const int pitch = 1024 + tail_lanes * C10_WARP_SIZE;       // halves per stored row
      attention_scores_softmax_kernel<<<dim3((m + ass_rows - 1) / ass_rows, batch), ass_warps * C10_WARP_SIZE,
                                        ass_rows * pitch * sizeof (__half), stream>>>(q, k, p, m, n, pitch,
                                                                                       tail_lanes, alpha);
    }

  }
}
