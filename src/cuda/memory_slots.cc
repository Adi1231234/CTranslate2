#include "cuda/memory_slots.h"

#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local const MemorySlots* active = nullptr;

    const MemorySlots* memory_slots() {
      return active;
    }

    bool memory_slots_enabled() {
      static const bool enabled = read_bool_from_env("CT2_MEMORY_SLOTS", true) && hmma_replicas_verified();
      return enabled;
    }

    MemorySlotsScope::MemorySlotsScope(const MemorySlots& slots)
      : _previous(active) {
      active = &slots;
    }

    MemorySlotsScope::~MemorySlotsScope() {
      active = _previous;
    }

  }
}
