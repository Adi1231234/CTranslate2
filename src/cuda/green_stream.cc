#include "cuda/green_stream.h"

#include <cstdio>
#include <cuda.h>
#include <spdlog/spdlog.h>

#include "cuda/driver_function.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    // Green contexts: CUDA 12.4, their streams 12.5.
    struct Driver {
      CUresult (*device_get)(CUdevice*, int) = driver_function<decltype(device_get)>("cuDeviceGet");
      CUresult (*get_resource)(CUdevice, CUdevResource*, CUdevResourceType)
        = driver_function<decltype(get_resource)>("cuDeviceGetDevResource");
      CUresult (*split)(CUdevResource*, unsigned*, const CUdevResource*, CUdevResource*, unsigned, unsigned)
        = driver_function<decltype(split)>("cuDevSmResourceSplitByCount");
      CUresult (*describe)(CUdevResourceDesc*, CUdevResource*, unsigned)
        = driver_function<decltype(describe)>("cuDevResourceGenerateDesc");
      CUresult (*create)(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned)
        = driver_function<decltype(create)>("cuGreenCtxCreate");
      CUresult (*stream_create)(CUstream*, CUgreenCtx, unsigned, int)
        = driver_function<decltype(stream_create)>("cuGreenCtxStreamCreate");
    };

    static const Driver& driver() {
      static const Driver d;
      return d;
    }

    static std::nullptr_t failed(const char* step, int code) {
      spdlog::warn("SM partition: {} failed ({}), the streams use the whole GPU", step, code);
      return nullptr;
    }

    // A green context of `count` SM resources of `device`.
    static CUgreenCtx green_context(CUdevice device, CUdevResource* sms, unsigned count) {
      CUdevResourceDesc desc;
      CUgreenCtx context = nullptr;                                // lives as long as the process
      CUresult r = driver().describe(&desc, sms, count);
      if (r == CUDA_SUCCESS)
        r = driver().create(&context, desc, device, CU_GREEN_CTX_DEFAULT_STREAM);
      return r == CUDA_SUCCESS ? context : failed("making a green context", r);
    }

    struct Partition {
      CUgreenCtx decoder = nullptr, encoder = nullptr;
    };

    // CT2_SM_PARTITION or CT2_ENCODER_SMS (green_stream.h) on the current device, from one split of its SMs:
    // `groups` sets of `size` SMs and the rest.
    static Partition make_partition() {
      Partition p;
      const std::string spec = use_stock_kernels() ? "" : read_string_from_env("CT2_SM_PARTITION", "");
      const unsigned encoder_only = use_stock_kernels() ? 0 : read_int_from_env("CT2_ENCODER_SMS", 0);
      unsigned dec = 0, enc = 0;
      if (spec.empty() && encoder_only == 0)
        return p;
      if (!spec.empty() && (std::sscanf(spec.c_str(), "%u:%u", &dec, &enc) != 2 || (dec % 8 && enc % 8))) {
        failed("reading CT2_SM_PARTITION (decoder:encoder, one of them a multiple of 8)", 0);
        return p;
      }
      const Driver& d = driver();
      if (!d.device_get || !d.get_resource || !d.split || !d.describe || !d.create || !d.stream_create) {
        failed("finding the driver's green context functions", 0);
        return p;
      }
      int ordinal = 0;
      CUdevice device;
      CUdevResource all, sets[8], rest;
      CUresult r = d.device_get(&device, cudaGetDevice(&ordinal) == cudaSuccess ? ordinal : 0);
      if (r == CUDA_SUCCESS)
        r = d.get_resource(device, &all, CU_DEV_RESOURCE_TYPE_SM);
      if (r != CUDA_SUCCESS) {
        failed("reading the device's SMs", r);
        return p;
      }
      // The side given as whole groups of 8 is split off; the other side is the rest.
      const bool decoder_split = !spec.empty() && dec % 8 == 0;
      const unsigned split_sms = !spec.empty() ? (decoder_split ? dec : enc) : encoder_only;
      const bool whole = split_sms % 8 == 0;
      const unsigned groups = whole ? split_sms / 8 : 1, size = whole ? 8 : all.sm.smCount - split_sms;
      unsigned made = groups;
      if (groups > 8 || (r = d.split(sets, &made, &all, &rest, 0, size)) != CUDA_SUCCESS || made != groups) {
        failed("cuDevSmResourceSplitByCount", r != CUDA_SUCCESS ? int(r) : int(made));
        return p;
      }
      CUgreenCtx part = whole ? green_context(device, sets, groups) : green_context(device, &rest, 1);
      if (spec.empty()) {
        p.encoder = part;                                          // the decoder keeps the whole GPU
        return p;
      }
      CUgreenCtx other = green_context(device, &rest, 1);
      p.decoder = decoder_split ? part : other;
      p.encoder = decoder_split ? other : part;
      spdlog::info("SM partition: decoder {} and encoder {} of {} SMs", decoder_split ? split_sms : rest.sm.smCount,
                   decoder_split ? rest.sm.smCount : split_sms, all.sm.smCount);
      return p;
    }

    cudaStream_t create_partition_stream(bool low, int priority) {
      static const Partition partition = make_partition();         // one GPU: made on the first worker's device
      const CUgreenCtx context = low ? partition.encoder : partition.decoder;
      if (!context)
        return nullptr;
      CUstream stream = nullptr;                                   // green context streams must be non-blocking
      const CUresult r = driver().stream_create(&stream, context, CU_STREAM_NON_BLOCKING, priority);
      return r == CUDA_SUCCESS ? reinterpret_cast<cudaStream_t>(stream) : failed("cuGreenCtxStreamCreate", r);
    }

  }
}
