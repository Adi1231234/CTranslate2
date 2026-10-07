#pragma once

namespace ctranslate2 {
  namespace cuda {

    // CT2_CACHE_POOL=1: the decoder's long-lived buffers (a window's memory keys and values, its slots, a ladder's
    // capacity caches; they live for a window's or a ladder's whole decoding) from a memory pool of their own, apart
    // from a step's short-lived ones. In one pool they sat between the short ones, and the pool's reserve grew to the
    // whole device with ~30 of its 44 GiB in use (joint5/joint6: out of memory with the reserve full and the in-use
    // flat): a block with one long-lived buffer in it can be neither reused nor trimmed. Only where memory comes from
    // changes, never a value (its alignment is the pool's, 256 bytes at least, as the default's).
    bool cache_pool_enabled();
    bool cache_pool_active();                 // the calling thread's allocations go to the cache pool now

    // While an instance lives, the calling thread's CUDA allocations come from the cache pool (where enabled).
    class CachePoolScope {
    public:
      CachePoolScope();
      ~CachePoolScope();
      CachePoolScope(const CachePoolScope&) = delete;
      CachePoolScope& operator=(const CachePoolScope&) = delete;
    private:
      const bool _previous;
    };

  }
}
