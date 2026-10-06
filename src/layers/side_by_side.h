#pragma once

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#include <algorithm>
#include <vector>

#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace layers {

    // CT2_JOINT_STREAMS=<n> (default 0: off): a joint step's parts' self-attention products on n side streams of the
    // thread (cuda::SideStreamScope, a cuBLAS handle each), the parts dealt to them in turn. Each part runs the very
    // calls, kernels and data it runs on the thread's own stream; only side by side: with a part a window (long
    // recordings), a step's ~30 parts' calls of 100 entries each fill a small part of the GPU one after the other
    // (long5's profile: ~2/3 of the GPU's time in the per-part self-attention).
    inline int joint_streams() {
      static const int streams = read_int_from_env("CT2_JOINT_STREAMS", 0);
      return streams;
    }

    // run(p) for p in [0, count) on the side streams, after the thread's stream's work so far and before its work
    // from now on (events). run allocates nothing (the allocator is the thread's stream's).
    template <typename Run>
    void side_by_side(size_t count, Run&& run) {
      const int streams = static_cast<int>(std::min<size_t>(joint_streams(), count));
      static thread_local std::vector<cudaEvent_t> events;   // [0]: the fork; [s]: side stream s done
      while (events.size() <= static_cast<size_t>(streams)) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events.push_back(event);
      }
      const cudaStream_t own = cuda::get_cuda_stream();
      CUDA_CHECK(cudaEventRecord(events[0], own));
      for (int s = 1; s <= streams; ++s) {
        const cuda::SideStreamScope side(s);
        CUDA_CHECK(cudaStreamWaitEvent(cuda::get_cuda_stream(), events[0], 0));
        for (size_t p = s - 1; p < count; p += streams)
          run(p);
        CUDA_CHECK(cudaEventRecord(events[s], cuda::get_cuda_stream()));
      }
      for (int s = 1; s <= streams; ++s)
        CUDA_CHECK(cudaStreamWaitEvent(own, events[s], 0));
    }

  }
}
#endif
