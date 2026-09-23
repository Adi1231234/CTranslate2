#include "ctranslate2/ops/softmax.h"

#include <type_traits>

#include "cuda/utils.h"
#include "softmax_kernels.cuh"

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

namespace ctranslate2 {
  namespace ops {

    template <typename T, template <typename, typename, typename> class Epilogue>
    static void softmax_kernel_impl(cudaStream_t stream,
                                    const T* x,
                                    const dim_t rows,
                                    const dim_t cols,
                                    const int32_t* lengths,
                                    T* y) {
      at::native::softmax_rows<T, Epilogue>(stream, x, y, rows, cols, lengths,
                                            /*warp=*/!cuda::use_stock_kernels());
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
