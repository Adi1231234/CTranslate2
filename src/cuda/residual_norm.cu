#ifndef CT2_USE_HIP

#include "cuda/residual_norm.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int residual_norm_threads = 512;                 // ops::LayerNorm's CUDA_NUM_THREADS

    // LayerNormForwardCUDAKernel (src/ops/layer_norm_gpu.cu) reading X = (bias + x) + residual, which it also
    // stores to S: the loops, the reduction and every expression are that kernel's.
    __global__ void residual_norm_kernel(unsigned N, float eps, const __half* X, const __half* bias,
                                         const __half* R, const __half* gamma, const __half* beta,
                                         __half* S, __half* Y) {
      typedef cub::BlockReduce<float, residual_norm_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage m_temp_storage;
      __shared__ typename BlockReduce::TempStorage v_temp_storage;
      __shared__ float s_mean;
      __shared__ float s_variance;

      const unsigned i = blockIdx.x;

      float sum1 = 0;
      float sum2 = 0;
      for (unsigned j = threadIdx.x; j < N; j += blockDim.x) {
        const unsigned index = i * N + j;
        const __half s = __hadd(__hadd(bias[j], X[index]), R[index]);   // bias_add_vec_apply<residual>
        S[index] = s;
        sum1 += float(s);
        sum2 += float(s) * float(s);
      }
      sum1 = BlockReduce(m_temp_storage).Sum(sum1);
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      if (threadIdx.x == 0) {
        const float scale = float(1) / float(N);
        sum1 *= scale;
        sum2 = fmaxf(sum2 * scale - sum1 * sum1, float(0));
        s_mean = sum1;
        s_variance = rsqrtf(sum2 + eps);
      }

      __syncthreads();

      for (unsigned j = threadIdx.x; j < N; j += blockDim.x) {
        const unsigned index = i * N + j;
        const float gamma_v = gamma == nullptr ? float(1) : float(gamma[j]);
        const float beta_v = beta == nullptr ? float(0) : float(beta[j]);
        Y[index] = __half((float(S[index]) - s_mean) * s_variance * gamma_v + beta_v);
      }
    }

    void residual_norm(const float16_t* x, const float16_t* bias, const float16_t* residual,
                       const float16_t* gamma, const float16_t* beta, float epsilon,
                       float16_t* sum, float16_t* normed, dim_t rows, dim_t depth) {
      if (rows == 0)
        return;
      auto h = [](const float16_t* p) { return reinterpret_cast<const __half*>(p); };
      residual_norm_kernel<<<rows, residual_norm_threads, 0, get_cuda_stream()>>>(
        static_cast<unsigned>(depth), epsilon, h(x), h(bias), h(residual), h(gamma), h(beta),
        reinterpret_cast<__half*>(sum), reinterpret_cast<__half*>(normed));
    }

  }
}

#endif
