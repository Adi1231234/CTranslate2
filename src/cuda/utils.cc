#include "./utils.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"
#include "cuda/graph.h"

#include "env.h"

#ifdef CT2_USE_HIP
#define CUBLAS_STATUS_SUCCESS          HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED  HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED     HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE    HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH    HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR    HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR   HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED    HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_LICENSE_ERROR    HIPBLAS_STATUS_UNKNOWN
#define cudaStreamDefault hipStreamDefault
#define cudaGetDevice hipGetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSuccess hipSuccess
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif

namespace ctranslate2 {
  namespace cuda {

    const char* cublasGetStatusName(cublasStatus_t status)
    {
      switch (status)
      {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "UNKNOWN";
      }
    }

    // We assign the default CUDA stream to the main thread since it can interact with
    // multiple devices (e.g. load replicas on each GPU). The main thread is created
    // before the others, so it will be the first to see the flag below set to true.
    static std::atomic<bool> is_main_thread(true);

    // Worker streams get the highest priority, a thread's low-priority stream the lowest (the
    // default of a plain stream); CT2_CUDA_STOCK_KERNELS=1 keeps plain streams. CT2_CUDA_STREAM_PRIORITIES:
    // "equal" gives every stream the default priority, "encoder_high" swaps the two (for scheduling A/B
    // runs; only the order in which the GPU takes up kernels changes, never a result).
    static int stream_priority(bool low) {
      int least = 0, greatest = 0;
      if (use_stock_kernels() || cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess)
        return 0;
      const std::string mode = read_string_from_env("CT2_CUDA_STREAM_PRIORITIES", "decoder_high");
      if (mode == "equal")
        return 0;
      return (low != (mode == "encoder_high")) ? least : greatest;
    }

    class CudaStream {
    public:
      CudaStream(bool low = false) {
        if (is_main_thread && !low && !graphs_enabled()) {   // graphs capture created streams only (graph.h)
          is_main_thread = false;
          _stream = cudaStreamDefault;
        } else {
          CUDA_CHECK(cudaGetDevice(&_device));
          CUDA_CHECK(cudaStreamCreateWithPriority(&_stream, cudaStreamDefault, stream_priority(low)));
        }
      }
      ~CudaStream() {
        if (_stream != cudaStreamDefault) {
          ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
          cudaStreamDestroy(_stream);
        }
      }
      cudaStream_t get() const {
        return _stream;
      }
    private:
      int _device;
      cudaStream_t _stream;
    };

    class CublasHandle {
    public:
      CublasHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, get_cuda_stream()));
      }
      ~CublasHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cublasDestroy(_handle);
      }
      cublasHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cublasHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    static thread_local bool low_priority_stream = false;

    cudaStream_t get_cuda_stream() {
      static thread_local CudaStream cuda_stream;
      if (low_priority_stream) {
        static thread_local CudaStream low_stream(/*low=*/true);
        return low_stream.get();
      }
      return cuda_stream.get();
    }

    UseLowPriorityStreamInScope::UseLowPriorityStreamInScope()
      : _previous_value(low_priority_stream) {
      low_priority_stream = true;
    }

    UseLowPriorityStreamInScope::~UseLowPriorityStreamInScope() {
      low_priority_stream = _previous_value;
    }

    cublasHandle_t get_cublas_handle() {
      static thread_local CublasHandle cublas_handle;
      static thread_local cudaStream_t bound = get_cuda_stream();   // the handle's stream at creation
      const cudaStream_t stream = get_cuda_stream();
      if (stream != bound) {                                        // follow the thread's active stream
        CUBLAS_CHECK(cublasSetStream(cublas_handle.get(), stream));
        bound = stream;
      }
      return cublas_handle.get();
    }

#ifdef CT2_WITH_CUDNN
    class CudnnHandle {
    public:
      CudnnHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, get_cuda_stream()));
      }
      ~CudnnHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cudnnDestroy(_handle);
      }
      cudnnHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cudnnHandle_t _handle;
    };

    cudnnHandle_t get_cudnn_handle() {
      static thread_local CudnnHandle cudnn_handle;
      return cudnn_handle.get();
    }

    cudnnDataType_t get_cudnn_data_type(DataType dtype) {
      switch (dtype) {
      case DataType::FLOAT32:
        return CUDNN_DATA_FLOAT;
      case DataType::FLOAT16:
        return CUDNN_DATA_HALF;
      case DataType::BFLOAT16:
        return CUDNN_DATA_BFLOAT16;
      case DataType::INT32:
        return CUDNN_DATA_INT32;
      case DataType::INT8:
        return CUDNN_DATA_INT8;
      default:
        throw std::invalid_argument("No cuDNN data type for type " + dtype_name(dtype));
      }
    }
#endif

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    const cudaDeviceProp& get_device_properties(int device) {
      static thread_local std::vector<std::unique_ptr<cudaDeviceProp>> cache;

      if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
      }
      if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
      }

      auto& device_prop = cache[device];
      if (!device_prop) {
        device_prop = std::make_unique<cudaDeviceProp>();
        CUDA_CHECK(cudaGetDeviceProperties(device_prop.get(), device));
      }
      return *device_prop;
    }

#ifdef CT2_USE_HIP
    // https://rocm.docs.amd.com/en/latest/reference/precision-support.html
    // All archs supported by ROCm 7 support the following precisions

    bool gpu_supports_int8(int device) {
      return true;
    }

    bool gpu_has_int8_tensor_cores(int device) {
      return true;
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      return true;
    }
#else

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool gpu_supports_int8(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool gpu_has_int8_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 7 || (device_prop.major == 7 && device_prop.minor >= 2);
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }
#endif

    bool have_same_compute_capability(const std::vector<int>& devices) {
      if (devices.size() > 1) {
        int ref_major = -1;
        int ref_minor = -1;
        for (const int device : devices) {
          const cudaDeviceProp& device_prop = get_device_properties(device);
          const int major = device_prop.major;
          const int minor = device_prop.minor;
          if (ref_major < 0) {
            ref_major = major;
            ref_minor = minor;
          } else if (major != ref_major || minor != ref_minor)
            return false;
        }
      }

      return true;
    }

    static thread_local bool true_fp16_gemm = read_bool_from_env("CT2_CUDA_TRUE_FP16_GEMM", true);

    bool use_true_fp16_gemm() {
      return true_fp16_gemm;
    }

    void use_true_fp16_gemm(bool use) {
      true_fp16_gemm = use;
    }

    bool use_stock_kernels() {
      static const bool stock = read_bool_from_env("CT2_CUDA_STOCK_KERNELS");
      return stock;
    }

#ifndef CT2_USE_HIP
    // True where the fork's replicas of this cuBLAS build's kernels were verified on the device's arch.
    static bool replicas_verified_on(int major, int minor) {
      constexpr int verified_cublas_version = 120902;
      static const int cublas_version = [] {
        int version = 0;
        CUBLAS_CHECK(cublasGetVersion(get_cublas_handle(), &version));
        return version;
      }();
      const cudaDeviceProp& device_prop = get_device_properties();
      return !use_stock_kernels()
        && device_prop.major == major && device_prop.minor == minor
        && cublas_version == verified_cublas_version;
    }
#endif

    bool cublas_replicas_verified() {
#ifdef CT2_USE_HIP
      return false;
#else
      return replicas_verified_on(7, 5);
#endif
    }

    bool hmma_replicas_verified() {
#ifdef CT2_USE_HIP
      return false;
#else
      return replicas_verified_on(12, 0);
#endif
    }

  }
}
