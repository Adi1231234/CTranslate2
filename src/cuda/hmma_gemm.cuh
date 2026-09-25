#pragma once

// Launch of hmma_gemm_kernel.cuh: block width by the number of columns (or a given number of warps, for
// tools/turing/kernels/hmma_check.cu), kernel by rows of A, and recipe 3's slice combination.

#include "hmma_gemm_kernel.cuh"

namespace ctranslate2 {
  namespace cuda {

    template <int MT, int Warps>
    void hg_launch(const __half* A, const __half* W, __half* C, float* P, int M, int N, int K,
                   int slices, int slice_k, cudaStream_t stream) {
      constexpr int smem = hg_stages * (MT * 16 + Warps * 16) * hg_pitch * sizeof (__half);
      static const bool configured = cudaFuncSetAttribute(hmma_gemm_kernel<MT, Warps>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      const dim3 grid((N + Warps * 16 - 1) / (Warps * 16), slices);
      hmma_gemm_kernel<MT, Warps><<<grid, Warps * 32, smem, stream>>>(A, W, C, P, M, N, K, slice_k);
    }

    // warps: 16-column warps per block (1, 2, 4 or 8); 0 picks by the number of columns.
    template <int MT>
    void hg_columns(const __half* A, const __half* W, __half* C, float* P, int M, int N, int K,
                    int slices, int slice_k, cudaStream_t stream, int warps) {
      if (warps == 0)
        warps = N <= 1280 ? 2 : 4;
      switch (warps) {
      case 1: hg_launch<MT, 1>(A, W, C, P, M, N, K, slices, slice_k, stream); break;
      case 2: hg_launch<MT, 2>(A, W, C, P, M, N, K, slices, slice_k, stream); break;
      case 4: hg_launch<MT, 4>(A, W, C, P, M, N, K, slices, slice_k, stream); break;
      default: hg_launch<MT, 8>(A, W, C, P, M, N, K, slices, slice_k, stream); break;
      }
    }

    // C = A W^T for a routed shape (hmma_gemm_recipe(M, N, K) != 0); A, W 16-byte aligned rows. workspace:
    // hmma_gemm_workspace_bytes(M, N, K) bytes on the device (unused by recipe 1).
    inline void hmma_gemm(const __half* A, const __half* W, __half* C, int M, int N, int K,
                          void* workspace, cudaStream_t stream, int warps = 0) {
      const int slices = hmma_gemm_slices(hmma_gemm_recipe(M, N, K));
      const int slice_k = static_cast<int>(hmma_gemm_slice_k(K, slices));
      float* P = slices > 1 ? static_cast<float*>(workspace) : nullptr;
      if (M <= 16)
        hg_columns<1>(A, W, C, P, M, N, K, slices, slice_k, stream, warps);
      else if (M <= 32)
        hg_columns<2>(A, W, C, P, M, N, K, slices, slice_k, stream, warps);
      else
        hg_columns<3>(A, W, C, P, M, N, K, slices, slice_k, stream, warps);
      if (P)
        hmma_gemm_combine<<<(M * N + 255) / 256, 256, 0, stream>>>(P, C, (size_t)M * N, slices);
    }

  }
}
