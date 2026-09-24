#pragma once

// Row softmax kernels: the upstream cunn_SoftMaxForward (adapted from PyTorch, unchanged), this
// fork's bit-exact warp_softmax_forward, and softmax_rows, which picks one. Shared by softmax_gpu.cu
// and tools/turing/kernels/softmax_check.cu, which compares the two bit for bit on every row length.

#include <limits>
#include <type_traits>

#include "cuda/helpers.h"

// The following CUDA kernels are adapted from:
// https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
// which has the following license notice:

/*
  From PyTorch:

  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

  From Caffe2:

  Copyright (c) 2016-present, Facebook Inc. All rights reserved.

  All contributions by Facebook:
  Copyright (c) 2016 Facebook Inc.

  All contributions by Google:
  Copyright (c) 2015 Google Inc.
  All rights reserved.

  All contributions by Yangqing Jia:
  Copyright (c) 2015 Yangqing Jia
  All rights reserved.

  All contributions from Caffe:
  Copyright(c) 2013, 2014, 2015, the respective contributors
  All rights reserved.

  All other contributions:
  Copyright(c) 2015, 2016 the respective contributors
  All rights reserved.

  Caffe2 uses a copyright model similar to Caffe: each contributor holds
  copyright over their contributions to Caffe2. The project versioning records
  all such contribution and copyright details. If a contributor wants to further
  mark their specific copyright on a particular contribution, they should
  indicate their copyright solely in the commit message of the change when it is
  committed.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
     and IDIAP Research Institute nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

namespace at {
  namespace native {

    constexpr float max_float = std::numeric_limits<float>::max();

    template<typename T, typename AccumT, typename OutT>
    struct LogSoftMaxForwardEpilogue {
      __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input),  logsum(std::log(sum)) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(static_cast<AccumT>(input) - max_input - logsum);
      }

      const AccumT max_input;
      const AccumT logsum;
    };

    template<typename T, typename AccumT, typename OutT>
    struct SoftMaxForwardEpilogue {
      __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input)
        , sum(sum) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(std::exp(static_cast<AccumT>(input) - max_input) / sum);
      }

      const AccumT max_input;
      const AccumT sum;
    };

    template<typename T, typename AccumT>
    struct SumExpFloat
    {
      __device__ __forceinline__ SumExpFloat(AccumT v)
        : max_k(v) {}

      __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
        return sum + std::exp(static_cast<AccumT>(v) - max_k);
      }

      const AccumT max_k;
    };

    template<typename T>
    struct Max {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? b : a;
      }
    };

    template <typename T, typename AccumT>
    struct MaxFloat
    {
      __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
        return ::max(max, (AccumT)v);
      }
    };

    template<typename T>
    struct Add {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
      }
    };

    template <typename scalar_t,
              typename accscalar_t,
              typename outscalar_t,
              typename index_t,
              typename length_t,
              template <typename, typename, typename> class Epilogue>
    __global__ void
    cunn_SoftMaxForward(outscalar_t *output,
                        const scalar_t *input,
                        const index_t classes,
                        const length_t *lengths)
    {
      extern __shared__ unsigned char smem[];
      auto sdata = reinterpret_cast<accscalar_t*>(smem);
      // forward pointers to batch[blockIdx.x]
      // each block handles a sample in the mini-batch
      const index_t row = blockIdx.x;
      input += row * classes;
      output += row * classes;

      index_t size = classes;
      if (lengths)
      {
        // Directly set 0 in output for out of range positions.
        size = lengths[row];
        for (index_t i = size + threadIdx.x; i < classes; i += blockDim.x)
          output[i] = 0.f;
      }

      // find the max
      accscalar_t threadMax = ctranslate2::cuda::ilp_reduce(
        input, size, MaxFloat<scalar_t, accscalar_t>(), -max_float);
      accscalar_t max_k = ctranslate2::cuda::block_reduce(
        sdata, threadMax, Max<accscalar_t>(), -max_float);

      // reduce all values
      accscalar_t threadExp = ctranslate2::cuda::ilp_reduce(
        input, size, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
      accscalar_t sumAll = ctranslate2::cuda::block_reduce(
        sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

      // apply epilogue
      ctranslate2::cuda::apply_epilogue(
        input, size, Epilogue<scalar_t, accscalar_t, outscalar_t>(max_k, sumAll), output);
    }

    // Bit-exact replacement of cunn_SoftMaxForward for rows of at most 2048: one warp per row.
    //
    // cunn_SoftMaxForward runs B = get_block_size(cols) threads per row (1024 for the 1500-wide
    // Whisper attention rows, so only one row fits on an SM at a time). Its sum is order-dependent
    // and fixed: thread v adds exp(x[v] - max), exp(x[v+B] - max), exp(x[v+2B] - max), ... in that
    // order (ilp_reduce; B is the power of two >= cols/2 rounded down, so a thread has up to three
    // terms, three only when cols = 2B + 1: 65, 129, 257, ...), block_reduce adds threads
    // 32L..32L+31 in order for each L < B/32, then those B/32 partial sums in order. Here the row is
    // loaded once, coalesced, into shared memory; lane L adds its 32 virtual threads in exactly that
    // order and the B/32 lane sums are added in order through shuffles. The max is exact whatever the
    // order (no rounding), so it is reduced straight from the coalesced load. exp(x - max) is
    // computed once, kept in shared memory and reused by the epilogue: the same expression on the
    // same operands as the legacy epilogue's. Same float operations on the same values: identical
    // output, checked on every row length by tools/turing/kernels/softmax_check.cu.
    constexpr unsigned warp_softmax_rows_per_block = 4;
    constexpr unsigned warp_softmax_max_cols = 2048;

    inline __host__ __device__ __forceinline__ unsigned warp_softmax_slot(unsigned j) {
      return j + j / 32;  // one pad float per 32: lane L reading 32L + i hits bank (L + i) % 32
    }

    template <typename scalar_t, bool LogSoftmax>
    __global__ void __launch_bounds__(warp_softmax_rows_per_block * C10_WARP_SIZE)
    warp_softmax_forward(scalar_t* output,
                         const scalar_t* input,
                         const unsigned rows,
                         const unsigned classes,
                         const unsigned legacy_block,
                         const int32_t* lengths)
    {
      extern __shared__ float row_smem[];
      const unsigned warp = threadIdx.x / C10_WARP_SIZE;
      const unsigned lane = threadIdx.x % C10_WARP_SIZE;
      const unsigned row = blockIdx.x * warp_softmax_rows_per_block + warp;
      if (row >= rows)
        return;
      float* buf = row_smem + warp * (warp_softmax_slot(classes) + 1);
      input += size_t(row) * classes;
      output += size_t(row) * classes;

      unsigned size = classes;
      if (lengths) {
        size = lengths[row];
        for (unsigned i = size + lane; i < classes; i += C10_WARP_SIZE)
          output[i] = 0.f;
      }

      float lane_max = -max_float;
      for (unsigned j = lane; j < size; j += C10_WARP_SIZE) {
        const float v = static_cast<float>(input[j]);
        buf[warp_softmax_slot(j)] = v;
        lane_max = MaxFloat<float, float>()(lane_max, v);
      }
      #pragma unroll
      for (unsigned offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2)
        lane_max = Max<float>()(lane_max, __shfl_xor_sync(0xffffffff, lane_max, offset));
      const float max_k = lane_max;
      __syncwarp();

      const unsigned groups = legacy_block / C10_WARP_SIZE;
      float warp_sum = 0.f;
      if (lane < groups) {
        const unsigned base = lane * C10_WARP_SIZE;
        #pragma unroll
        for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
          float thread_sum = 0.f;
          for (unsigned j = base + i; j < size; j += legacy_block) {   // virtual thread base + i
            const float e = std::exp(buf[warp_softmax_slot(j)] - max_k);
            if (!LogSoftmax)
              buf[warp_softmax_slot(j)] = e;  // element j belongs to this lane only
            thread_sum = thread_sum + e;
          }
          warp_sum = Add<float>()(warp_sum, thread_sum);
        }
      }
      float sum = 0.f;
      for (unsigned g = 0; g < groups; ++g)
        sum = Add<float>()(sum, __shfl_sync(0xffffffff, warp_sum, g));
      __syncwarp();

      const float logsum = LogSoftmax ? std::log(sum) : 0.f;
      for (unsigned j = lane; j < size; j += C10_WARP_SIZE) {
        const float v = buf[warp_softmax_slot(j)];
        output[j] = static_cast<scalar_t>(LogSoftmax ? v - max_k - logsum : v / sum);
      }
    }

  }
}

#include "softmax_rows1024.cuh"

namespace at {
  namespace native {

    // Softmax (Epilogue = SoftMaxForwardEpilogue) or log-softmax of `rows` rows of `cols` values,
    // masked to lengths[row] when given: when `warp` is set, softmax_rows1024 where it applies,
    // else the warp kernel for rows up to warp_softmax_max_cols; the legacy kernel otherwise.
    template <typename T, template <typename, typename, typename> class Epilogue>
    void softmax_rows(cudaStream_t stream, const T* x, T* y, unsigned rows, unsigned cols,
                      const int32_t* lengths, bool warp) {
      const dim3 block(ctranslate2::cuda::get_block_size(cols));
      constexpr bool is_log = std::is_same<Epilogue<T, float, T>,
                                           LogSoftMaxForwardEpilogue<T, float, T>>::value;
      constexpr bool is_half = std::is_same<T, __half>::value;
      if (warp && softmax_rows1024_applies(x, y, cols, block.x, is_half, is_log, lengths != nullptr)) {
        const unsigned grid = (rows + rows1024_per_block - 1) / rows1024_per_block;
        softmax_rows1024<<<grid, rows1024_per_block * C10_WARP_SIZE, 0, stream>>>(
          reinterpret_cast<__half*>(y), reinterpret_cast<const __half*>(x), rows, cols);
        return;
      }
      if (warp && cols <= warp_softmax_max_cols) {
        const dim3 grid((rows + warp_softmax_rows_per_block - 1) / warp_softmax_rows_per_block);
        const size_t smem = warp_softmax_rows_per_block * (warp_softmax_slot(cols) + 1) * sizeof (float);
        warp_softmax_forward<T, is_log>
          <<<grid, warp_softmax_rows_per_block * C10_WARP_SIZE, smem, stream>>>(y, x, rows, cols,
                                                                                block.x, lengths);
        return;
      }
      cunn_SoftMaxForward<T, float, T, ctranslate2::cuda::index_t, int32_t, Epilogue>
        <<<dim3(rows), block, block.x * sizeof (float), stream>>>(y, x, cols, lengths);
    }

  }
}
