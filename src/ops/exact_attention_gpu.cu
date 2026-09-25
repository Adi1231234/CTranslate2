#ifndef CT2_USE_HIP

#include "cuda/exact_attention.h"

#include <cstdint>

#include "exact_attention_launch.cuh"
#include "cuda/persistent.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    static bool aligned4(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 4 == 0;
    }

    bool exact_attention_applies(dim_t m, dim_t n, dim_t depth, const void* q, const void* k, const void* v) {
      return depth == at::native::ea_depth && m == 1500 && n == 1500
        && aligned4(q) && aligned4(k) && aligned4(v) && hmma_replicas_verified();
    }

    size_t exact_attention_workspace_bytes(dim_t batch, dim_t n) {
      return at::native::exact_attention_workspace(static_cast<int>(batch), static_cast<int>(n));
    }

    // CT2_EA_BLOCKS=<n>: persistent, n blocks per SM (cuda/persistent.h).
    void exact_attention(const float16_t* q, const float16_t* k, const float16_t* v, void* workspace,
                         float16_t* o, dim_t batch, dim_t heads, dim_t m, dim_t n, float alpha) {
      static const int per_sm = persistent_blocks_per_sm("CT2_EA_BLOCKS");
      cudaStream_t stream = get_cuda_stream();
      at::native::exact_attention(reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
                                  reinterpret_cast<const __half*>(v), workspace,
                                  reinterpret_cast<__half*>(o), static_cast<int>(batch), static_cast<int>(heads),
                                  static_cast<int>(m), static_cast<int>(n), alpha, stream,
                                  per_sm > 0 ? work_counter(stream) : nullptr, per_sm * sm_count());
    }

    bool exact_attention_qkv_applies(dim_t n, dim_t depth, const void* x, const void* bias) {
      return depth == at::native::ea_depth && n == 1500 && aligned4(x) && (!bias || aligned4(bias))
        && hmma_replicas_verified();
    }

    void exact_attention_qkv(const float16_t* x, const float16_t* bias, void* workspace, float16_t* o,
                             dim_t clips, dim_t heads, dim_t n, float alpha) {
      static const int per_sm = persistent_blocks_per_sm("CT2_EA_BLOCKS");
      cudaStream_t stream = get_cuda_stream();
      at::native::exact_attention_qkv(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(bias),
                                      workspace, reinterpret_cast<__half*>(o), static_cast<int>(clips),
                                      static_cast<int>(heads), static_cast<int>(n), alpha, stream,
                                      per_sm > 0 ? work_counter(stream) : nullptr, per_sm * sm_count());
    }

  }
}

#endif
