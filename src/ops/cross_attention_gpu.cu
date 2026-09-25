#ifndef CT2_USE_HIP

#include "cuda/cross_attention.h"

#include <algorithm>

#include "cross_attention.cuh"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    // cuBLAS 12.9.2's key tile for the output product on sm_120, by queries (2..8) and batch / 20 (1..8):
    // 32 or 64 keys (residue 1500 % 32 = 28) or 128 (residue 92). kernels/cross_sweep.cu, 400k outputs per
    // shape, no mismatch.
    static const unsigned char residues[7][8] = {
      {92, 28, 92, 92, 92, 28, 28, 92},   // m 2
      {92, 28, 92, 92, 28, 28, 28, 92},   // m 3
      {92, 28, 92, 92, 28, 28, 28, 28},   // m 4
      {92, 28, 92, 28, 28, 28, 28, 28},   // m 5
      {92, 28, 28, 28, 28, 28, 28, 28},   // m 6
      {92, 28, 28, 28, 28, 28, 28, 28},   // m 7
      {92, 28, 28, 28, 28, 28, 28, 28},   // m 8
    };

    int cross_attention_residue(dim_t m, dim_t batch, dim_t keys, dim_t depth) {
      static const bool enabled = read_bool_from_env("CT2_CROSS_ATTN", true);
      if (!enabled || keys != at::native::ca_keys || depth != at::native::ca_depth || m < 2 || m > 8
          || batch % 20 != 0 || batch < 20 || batch > 160 || !hmma_replicas_verified())
        return -1;
      return residues[m - 2][batch / 20 - 1];
    }

    // hmma_gemm_recipe.h: cuBLAS 12.9.2 runs the 1280 x 1280 Dense layer at 2..48 rows as one chain over k.
    // Off by default (CT2_CROSS_Q=1 enables it): exact, but the store PC's 150 clips took 21.8 s with it and
    // 21.5 s without (the projection's chain waits on L2 inside the attention kernel; cuBLAS's GEMM is faster).
    bool cross_attention_projects(dim_t rows, dim_t n, dim_t k) {
      static const bool enabled = read_bool_from_env("CT2_CROSS_Q", false);
      return enabled && rows >= 2 && rows <= 48 && n == 1280 && k == 1280 && hmma_replicas_verified();
    }

    void cross_attention(const float16_t* q, const float16_t* k, const float16_t* v, float16_t* o,
                         dim_t clips, dim_t heads, dim_t m, float alpha, int residue,
                         const float16_t* x, const float16_t* w, const float16_t* bias, dim_t k_inputs) {
      const int rows = static_cast<int>(std::min<dim_t>(m, 8));   // queries per pass (the mma's n)
      const int smem = rows * (at::native::ca_pitch + (x ? at::native::ca_qpitch : 0))
        * static_cast<int>(sizeof (__half));
      auto h = [](const float16_t* p) { return reinterpret_cast<const __half*>(p); };
      const at::native::CaQueries queries{h(q), h(x), h(w), h(bias), static_cast<int>(k_inputs)};
      // L2 prefetch distance in steps (0 none): 2-4 were 0.3 s faster on the store PC's 150 clips than 0.
      static const int ahead = read_int_from_env("CT2_CROSS_AHEAD", 4);
      at::native::cross_attention_kernel<<<static_cast<unsigned>(clips * heads), at::native::ca_warps * 32, smem,
                                           get_cuda_stream()>>>(
        queries, h(k), h(v), reinterpret_cast<__half*>(o), static_cast<int>(heads), static_cast<int>(m), rows,
        residue, alpha, ahead);
    }

  }
}

#endif
