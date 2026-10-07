#include "cuda/cache_pool.h"

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local bool active = false;

    bool cache_pool_enabled() {
      static const bool on = read_bool_from_env("CT2_CACHE_POOL");
      return on;
    }

    bool cache_pool_active() {
      return active;
    }

    CachePoolScope::CachePoolScope()
      : _previous(active) {
      active = cache_pool_enabled();
    }

    CachePoolScope::~CachePoolScope() {
      active = _previous;
    }

  }
}
