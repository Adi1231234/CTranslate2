#include "cuda/green_stream.h"

#include <cuda.h>
#include <spdlog/spdlog.h>

#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    int encoder_sm_count() {
      static const int sms = use_stock_kernels() ? 0 : read_int_from_env("CT2_ENCODER_SMS", 0);
      return sms;
    }

    // A driver API function of CUDA 12.4 (green contexts), from the driver in use; null when it has none.
    template <typename F>
    static F driver_function(const char* name) {
      void* fn = nullptr;
      cudaDriverEntryPointQueryResult found;
      if (cudaGetDriverEntryPointByVersion(name, &fn, 12040, cudaEnableDefault, &found) != cudaSuccess
          || found != cudaDriverEntryPointSuccess)
        return nullptr;
      return reinterpret_cast<F>(fn);
    }

    // The encoder keeps the whole GPU when a requested partition cannot be made: says which step failed.
    static cudaStream_t failed(const char* step, int code) {
      spdlog::warn("CT2_ENCODER_SMS: {} failed ({}), the encoder uses the whole GPU", step, code);
      return nullptr;
    }

    cudaStream_t create_green_stream(int sms, int priority) {
      const auto device_get = driver_function<CUresult (*)(CUdevice*, int)>("cuDeviceGet");
      const auto get_resource = driver_function<CUresult (*)(CUdevice, CUdevResource*, CUdevResourceType)>(
        "cuDeviceGetDevResource");
      const auto split = driver_function<CUresult (*)(CUdevResource*, unsigned*, const CUdevResource*,
                                                      CUdevResource*, unsigned, unsigned)>(
        "cuDevSmResourceSplitByCount");
      const auto describe = driver_function<CUresult (*)(CUdevResourceDesc*, CUdevResource*, unsigned)>(
        "cuDevResourceGenerateDesc");
      const auto create = driver_function<CUresult (*)(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned)>(
        "cuGreenCtxCreate");
      const auto stream_create = driver_function<CUresult (*)(CUstream*, CUgreenCtx, unsigned, int)>(
        "cuGreenCtxStreamCreate");
      if (!device_get || !get_resource || !split || !describe || !create || !stream_create)
        return failed("finding the driver's green context functions", 0);
      int ordinal = 0;
      CUdevice device;
      CUdevResource all, rest, groups[8];
      CUresult r = device_get(&device, cudaGetDevice(&ordinal) == cudaSuccess ? ordinal : 0);
      if (r != CUDA_SUCCESS)
        return failed("cuDeviceGet", r);
      if ((r = get_resource(device, &all, CU_DEV_RESOURCE_TYPE_SM)) != CUDA_SUCCESS)
        return failed("cuDeviceGetDevResource", r);
      if (sms <= 0 || unsigned(sms) >= all.sm.smCount)
        return failed("checking the SM count", int(all.sm.smCount));
      constexpr unsigned group = 8;                                // this GPU's SM group size
      const bool whole_groups = sms % group == 0;
      const unsigned wanted = whole_groups ? unsigned(sms) / group : 1;
      unsigned count = wanted;
      if (wanted > 8)
        return failed("checking the SM count", sms);
      if ((r = split(groups, &count, &all, &rest, 0, whole_groups ? group : all.sm.smCount - sms)) != CUDA_SUCCESS)
        return failed("cuDevSmResourceSplitByCount", r);
      if (count != wanted)
        return failed("splitting into SM groups", int(count));
      CUdevResourceDesc desc;
      CUgreenCtx context;                                          // lives as long as the process
      CUstream stream = nullptr;
      if ((r = describe(&desc, whole_groups ? groups : &rest, whole_groups ? count : 1)) != CUDA_SUCCESS)
        return failed("cuDevResourceGenerateDesc", r);
      if ((r = create(&context, desc, device, CU_GREEN_CTX_DEFAULT_STREAM)) != CUDA_SUCCESS)
        return failed("cuGreenCtxCreate", r);
      // Green context streams must be non-blocking.
      if ((r = stream_create(&stream, context, CU_STREAM_NON_BLOCKING, priority)) != CUDA_SUCCESS)
        return failed("cuGreenCtxStreamCreate", r);
      spdlog::info("CT2_ENCODER_SMS: the encoder runs on {} of {} SMs", sms, all.sm.smCount);
      return reinterpret_cast<cudaStream_t>(stream);
    }

  }
}
