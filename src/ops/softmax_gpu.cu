#include "ctranslate2/ops/softmax.h"

#include <type_traits>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               T* y);

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      softmax_kernel(cuda::get_cuda_stream(),
                     _log,
                     input.data<T>(),
                     batch_size,
                     depth,
                     lengths ? lengths->data<int32_t>() : nullptr,
                     output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

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

    // Bit-exact replacement of cunn_SoftMaxForward for short rows: one warp per row.
    //
    // cunn_SoftMaxForward runs B = get_block_size(cols) threads per row (1024 for the 1500-wide
    // Whisper attention rows, so only one row fits on an SM at a time). Its arithmetic is fixed:
    // thread v folds elements v, v+B, v+2B.. in increasing order (ilp_reduce), block_reduce then
    // folds threads 32L..32L+31 in order for each L < B/32, and finally folds those B/32 values
    // in order. For cols <= 2048, B >= cols / 2, so a thread holds at most two elements, v and
    // v+B. Here lane L keeps the elements of its 32 virtual threads (32L+i and 32L+i+B) in
    // registers and computes exactly those folds in the same order; the B/32 lane results are
    // folded in order through shuffles. Same float operations in the same order on the same
    // values, so the output is identical. exp(x - max) is computed once and reused by the
    // epilogue: it is the same expression on the same operands as the legacy epilogue's.
    constexpr unsigned warp_softmax_rows_per_block = 8;
    constexpr unsigned warp_softmax_max_cols = 2048;

    template <typename scalar_t, bool LogSoftmax>
    __global__ void __launch_bounds__(warp_softmax_rows_per_block * C10_WARP_SIZE)
    warp_softmax_forward(scalar_t* output,
                         const scalar_t* input,
                         const unsigned rows,
                         const unsigned classes,
                         const unsigned legacy_block,
                         const int32_t* lengths)
    {
      const unsigned warp = threadIdx.x / C10_WARP_SIZE;
      const unsigned lane = threadIdx.x % C10_WARP_SIZE;
      const unsigned row = blockIdx.x * warp_softmax_rows_per_block + warp;
      if (row >= rows)
        return;
      input += size_t(row) * classes;
      output += size_t(row) * classes;

      unsigned size = classes;
      if (lengths) {
        size = lengths[row];
        for (unsigned i = size + lane; i < classes; i += C10_WARP_SIZE)
          output[i] = 0.f;
      }

      const unsigned groups = legacy_block / C10_WARP_SIZE;
      const unsigned base = lane * C10_WARP_SIZE;
      const bool owner = lane < groups;
      float lo[C10_WARP_SIZE], hi[C10_WARP_SIZE];      // elements v = base + i and v + B
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
        const unsigned j = base + i;
        lo[i] = owner && j < size ? static_cast<float>(input[j]) : 0.f;
        hi[i] = owner && j + legacy_block < size ? static_cast<float>(input[j + legacy_block]) : 0.f;
      }

      float warp_max = -max_float;
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
        const unsigned j = base + i;
        float thread_max = -max_float;
        if (j < size)
          thread_max = MaxFloat<float, float>()(thread_max, lo[i]);
        if (j + legacy_block < size)
          thread_max = MaxFloat<float, float>()(thread_max, hi[i]);
        warp_max = Max<float>()(warp_max, thread_max);
      }
      float max_k = -max_float;
      for (unsigned g = 0; g < groups; ++g)
        max_k = Max<float>()(max_k, __shfl_sync(0xffffffff, warp_max, g));

      float warp_sum = 0.f;
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
        const unsigned j = base + i;
        float thread_sum = 0.f;
        if (j < size) {
          const float e = std::exp(lo[i] - max_k);
          if (!LogSoftmax)
            lo[i] = e;
          thread_sum = thread_sum + e;
        }
        if (j + legacy_block < size) {
          const float e = std::exp(hi[i] - max_k);
          if (!LogSoftmax)
            hi[i] = e;
          thread_sum = thread_sum + e;
        }
        warp_sum = Add<float>()(warp_sum, thread_sum);
      }
      float sum = 0.f;
      for (unsigned g = 0; g < groups; ++g)
        sum = Add<float>()(sum, __shfl_sync(0xffffffff, warp_sum, g));

      if (!owner)
        return;
      const float logsum = LogSoftmax ? std::log(sum) : 0.f;
      #pragma unroll
      for (unsigned i = 0; i < C10_WARP_SIZE; ++i) {
        const unsigned j = base + i;
        if (j < size)
          output[j] = static_cast<scalar_t>(LogSoftmax ? lo[i] - max_k - logsum : lo[i] / sum);
        if (j + legacy_block < size)
          output[j + legacy_block] = static_cast<scalar_t>(LogSoftmax ? hi[i] - max_k - logsum
                                                                      : hi[i] / sum);
      }
    }

  }
}

namespace ctranslate2 {
  namespace ops {

    // CT2_CUDA_LEGACY_SOFTMAX=1 forces cunn_SoftMaxForward everywhere (A/B checks of the
    // bit-exact warp kernel, which must produce identical outputs).
    static bool use_legacy_softmax() {
      static const bool legacy = read_bool_from_env("CT2_CUDA_LEGACY_SOFTMAX");
      return legacy;
    }

    template <typename T, template <typename, typename, typename> class Epilogue>
    static void softmax_kernel_impl(cudaStream_t stream,
                                    const T* x,
                                    const dim_t rows,
                                    const dim_t cols,
                                    const int32_t* lengths,
                                    T* y) {
      const dim3 block(cuda::get_block_size(cols));
      if (cols <= at::native::warp_softmax_max_cols && !use_legacy_softmax()) {
        constexpr bool is_log = std::is_same<Epilogue<T, float, T>,
                                          at::native::LogSoftMaxForwardEpilogue<T, float, T>>::value;
        const unsigned per_block = at::native::warp_softmax_rows_per_block;
        const dim3 grid((rows + per_block - 1) / per_block);
        at::native::warp_softmax_forward<T, is_log>
          <<<grid, per_block * C10_WARP_SIZE, 0, stream>>>(y, x, rows, cols, block.x, lengths);
        return;
      }
      const dim3 grid(rows);
      at::native::cunn_SoftMaxForward<T, float, T, cuda::index_t, int32_t, Epilogue>
        <<<grid, block, block.x * sizeof (float), stream>>>(y,
                                                            x,
                                                            cols,
                                                            lengths);
    }

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               T* y) {
      if (log_softmax)
        softmax_kernel_impl<cuda::device_type<T>, at::native::LogSoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, cuda::device_cast(y));
      else
        softmax_kernel_impl<cuda::device_type<T>, at::native::SoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, cuda::device_cast(y));
    }

  }
}
