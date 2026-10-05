#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Beam search keeps the decoder's memory keys and values in place when inputs finish (Decoder::update_state
    // otherwise compacts them, moving the 3.84 MB per input, layer and keys or values of every input after a
    // finished one: ~5% of the L40S's DRAM traffic in grouped decoding): while a MemorySlotsScope is active on a
    // thread, the fused cross-attention reads input i's keys and values at slot[i] of the uncompacted cache. Only
    // where the read of each input is unchanged: the kernel's arithmetic does not depend on where the values are.
    struct MemorySlots {
      const int32_t* slot = nullptr;   // on the device: the cache entry of each input still decoding
      dim_t inputs = 0;
    };

    const MemorySlots* memory_slots();

    // Where the cross-attention kernel can read through slots (hmma replicas) and not disabled
    // (CT2_MEMORY_SLOTS=0).
    bool memory_slots_enabled();

    class MemorySlotsScope {
    public:
      explicit MemorySlotsScope(const MemorySlots& slots);
      ~MemorySlotsScope();
      MemorySlotsScope(const MemorySlotsScope&) = delete;
      MemorySlotsScope& operator=(const MemorySlotsScope&) = delete;
    private:
      const MemorySlots* _previous;
    };

  }
}
