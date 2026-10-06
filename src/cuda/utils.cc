#include "./utils.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"
#include "cuda/graph.h"
#include "cuda/green_stream.h"

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
    // A thread's streams: its own, a low-priority one (UseLowPriorityStreamInScope: Whisper's encoder) and a
    // high-priority one (UseHighPriorityStreamInScope: a stream of sampled ladders that recording threads wait for).
    enum class StreamTier { normal, low, high };

    static int stream_priority(StreamTier tier) {
      int least = 0, greatest = 0;
      if (use_stock_kernels() || cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess)
        return 0;
      const std::string mode = read_string_from_env("CT2_CUDA_STREAM_PRIORITIES", "decoder_high");
      if (mode == "equal")
        return 0;
      if (tier == StreamTier::high)
        return greatest;
      const bool low = tier == StreamTier::low;
      // decoder_high: a thread's own stream one level under the high one where the range has room (CUDA: lower is
      // more urgent), so that the high one comes first and the low one still last.
      const int normal = greatest < least - 1 ? greatest + 1 : greatest;
      return (low != (mode == "encoder_high")) ? least : normal;
    }

    // CT2_CUDA_SCHEDULE=spin|yield|blocking: how a host thread waits for the GPU (cudaSetDeviceFlags), default the
    // runtime's heuristic, which spins whenever the process has no more contexts than the machine has cores. Several
    // processes sharing a GPU (MPS) on a few cores then spin away the cores their launching threads need: on a
    // 4-core L40S box with 4 processes the threads spent 2.2 cores in waits (cudaMemcpyAsync to the host and
    // cudaStreamSynchronize) and every launch took 13 us. Only how the host waits changes, never a result.
    static void apply_schedule_flags() {
#ifndef CT2_USE_HIP
      static const std::string mode = read_string_from_env("CT2_CUDA_SCHEDULE", "auto");
      if (mode == "auto")
        return;
      const unsigned flags = (mode == "spin" ? cudaDeviceScheduleSpin
                              : mode == "yield" ? cudaDeviceScheduleYield
                              : mode == "blocking" ? cudaDeviceScheduleBlockingSync
                              : throw std::invalid_argument("CT2_CUDA_SCHEDULE: spin, yield, blocking or auto"));
      static std::mutex mutex;
      static std::vector<int> applied;
      int device = 0;
      CUDA_CHECK(cudaGetDevice(&device));
      const std::lock_guard<std::mutex> lock(mutex);
      if (std::find(applied.begin(), applied.end(), device) != applied.end())
        return;
      CUDA_CHECK(cudaSetDeviceFlags(flags));   // overwrites the flags of an initialized device (CUDA >= 11)
      applied.push_back(device);
#endif
    }

    class CudaStream {
    public:
      CudaStream(StreamTier tier = StreamTier::normal) {
        apply_schedule_flags();
        if (is_main_thread && tier == StreamTier::normal && !graphs_enabled()) {   // graphs capture created streams
          is_main_thread = false;                                                    // only (graph.h)
          _stream = cudaStreamDefault;
        } else {
          CUDA_CHECK(cudaGetDevice(&_device));
          const bool low = tier == StreamTier::low;
          _stream = create_partition_stream(low, stream_priority(tier));   // on part of the GPU (green_stream.h)
          if (!_stream)
            CUDA_CHECK(cudaStreamCreateWithPriority(&_stream, cudaStreamDefault, stream_priority(tier)));
        }
      }
      ~CudaStream() {
        if (_stream == cudaStreamDefault)
          return;
        // The main thread's stream (a created one when graphs are on) goes at process exit, when the runtime may
        // be unloading already: then there is nothing to release, and nothing may throw here.
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess)
          return;
        if (current != _device)
          cudaSetDevice(_device);
        cudaStreamDestroy(_stream);
        if (current != _device)
          cudaSetDevice(current);
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
        if (graphs_enabled()) {
          // Inside a captured step cuBLAS would allocate its workspace with stream-ordered allocations, which
          // become graph memory nodes (no in-place update): a workspace of its own, the size of cuBLAS's default
          // pool on this GPU, so the same routines are chosen (cuBLAS 12.9 cublasSetWorkspace: 32 MiB on sm_90
          // and sm_10x, 12 MiB on sm_12x, 4 MiB otherwise). Set after cublasSetStream, which resets it.
          cudaDeviceProp prop;
          CUDA_CHECK(cudaGetDeviceProperties(&prop, _device));
          _workspace_size = size_t(prop.major == 12 ? 12 : prop.major >= 9 ? 32 : 4) << 20;
          CUDA_CHECK(cudaMalloc(&_workspace, _workspace_size));
          CUBLAS_CHECK(cublasSetWorkspace(_handle, _workspace, _workspace_size));
        }
      }
      ~CublasHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cublasDestroy(_handle);
        if (_workspace)
          cudaFree(_workspace);
      }
      cublasHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cublasHandle_t _handle;
      void* _workspace = nullptr;
      size_t _workspace_size = 0;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    static thread_local StreamTier stream_tier = StreamTier::normal;

    cudaStream_t get_cuda_stream() {
      static thread_local CudaStream cuda_stream;
      if (stream_tier == StreamTier::low) {
        static thread_local CudaStream low_stream(StreamTier::low);
        return low_stream.get();
      }
      if (stream_tier == StreamTier::high) {
        static thread_local CudaStream high_stream(StreamTier::high);
        return high_stream.get();
      }
      return cuda_stream.get();
    }

    UseLowPriorityStreamInScope::UseLowPriorityStreamInScope()
      : _previous_value(stream_tier == StreamTier::low) {
      stream_tier = StreamTier::low;
    }

    UseLowPriorityStreamInScope::~UseLowPriorityStreamInScope() {
      stream_tier = _previous_value ? StreamTier::low : StreamTier::normal;
    }

    UseHighPriorityStreamInScope::UseHighPriorityStreamInScope()
      : _previous(static_cast<int>(stream_tier)) {
      stream_tier = StreamTier::high;
    }

    UseHighPriorityStreamInScope::~UseHighPriorityStreamInScope() {
      stream_tier = static_cast<StreamTier>(_previous);
    }

    // One handle per stream of the thread, each bound to its stream once: cublasSetStream resets the handle's
    // workspace, and a worker that alternated between its streams (Whisper's encoder on the low-priority one)
    // then waited for the device, i.e. for the other threads' queued kernels, before its next job could start.
    cublasHandle_t get_cublas_handle() {
      if (stream_tier == StreamTier::low) {
        static thread_local CublasHandle low_handle;                // made while the low stream is active
        return low_handle.get();
      }
      if (stream_tier == StreamTier::high) {
        static thread_local CublasHandle high_handle;               // made while the high stream is active
        return high_handle.get();
      }
      static thread_local CublasHandle cublas_handle;
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

    bool cublas_verified_on(int major, int minor) {
#ifdef CT2_USE_HIP
      (void)major; (void)minor;
      return false;
#else
      return replicas_verified_on(major, minor);
#endif
    }

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
      // sm_89 (AWS L40S, the 8.6+PTX build's sm_86 code): exact_attention_check 0 mismatches, qk_hmma_probe and
      // av_hmma_probe the same orders as sm_120, hmma_probe chain 16 for 1280 x 1280 at 2..48 rows, and the
      // cross-attention residues of cross_sweep (cross_attention_gpu.cu).
      return replicas_verified_on(12, 0) || replicas_verified_on(8, 9);
#endif
    }

  }
}
