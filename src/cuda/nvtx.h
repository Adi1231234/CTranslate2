#pragma once

// A named range for profilers (NVTX v3, header-only: nvtxRangePushA does nothing unless a tool such as Nsight
// Systems is attached), so host phases such as a generate() call's prompt and decoding line up with their GPU
// work in a profile (nsys --trace=cuda,nvtx).

#include <nvtx3/nvToolsExt.h>

namespace ctranslate2 {
  namespace cuda {

    class NvtxRange {
    public:
      explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
      }
      ~NvtxRange() {
        nvtxRangePop();
      }
      NvtxRange(const NvtxRange&) = delete;
      NvtxRange& operator=(const NvtxRange&) = delete;
    };

  }
}
