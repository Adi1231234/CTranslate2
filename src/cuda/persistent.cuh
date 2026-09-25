#pragma once

// Device side of persistent.h: every thread of the block gets the block's next work item, or an index >= total
// when there is none left. The indices are handed out in increasing order and each block takes exactly one
// index >= total (then leaves), so the block given total + gridDim.x - 1 is the counter's last user in this
// kernel and resets it for the next kernel on the stream. `slot` is a shared int of the block.

namespace ctranslate2 {
  namespace cuda {

    __device__ __forceinline__ int next_work_item(unsigned* counter, int total, int& slot) {
      __syncthreads();                                  // the previous item is done with shared memory and slot
      if (threadIdx.x == 0) {
        const unsigned i = atomicAdd(counter, 1u);
        if (i == unsigned(total) + gridDim.x - 1)
          *counter = 0u;
        slot = int(i);
      }
      __syncthreads();
      return slot;
    }

  }
}
