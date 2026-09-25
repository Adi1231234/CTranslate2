#ifndef CT2_USE_HIP

#include "cuda/encoder_gemm.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "cuda/encoder_gemm_kernel.cuh"
#include "cuda/persistent.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static bool aligned16(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 16 == 0;
    }

    // CT2_ENC_GEMM_MIN_ROWS (default 1500, one 30 s window): smaller products keep cuBLAS, which may pick
    // another kernel (and another arithmetic) for them. k a multiple of 64: no k tile has a residue, in any
    // configuration below (a residue tile would change where the chain starts).
    bool encoder_gemm_applies(dim_t m, dim_t n, dim_t k, const void* a, const void* w, const void* c) {
      static const bool enabled = read_string_from_env("CT2_ENC_GEMM", "cutlass") != "cublas";
      static const dim_t min_rows = read_int_from_env("CT2_ENC_GEMM_MIN_ROWS", 1500);
      return enabled && m >= min_rows && n % 8 == 0 && k % 64 == 0
        && aligned16(a) && aligned16(w) && aligned16(c) && hmma_replicas_verified();
    }

    template <int Tile, int KTile, int Stages, typename Op>
    static void enc_gemm_launch(const float16_t* a, const float16_t* w, float16_t* c, int m, int n, int k,
                                const float16_t* bias) {
      using K = EncGemmKernel<Tile, KTile, Stages, Op>;
      using RefA = typename K::Mma::IteratorA::TensorRef;
      using RefB = typename K::Mma::IteratorB::TensorRef;
      using RefC = typename K::Epilogue::OutputTileIterator::TensorRef;
      auto half_ptr = [](const float16_t* p) { return reinterpret_cast<EncHalf*>(const_cast<float16_t*>(p)); };
      const cutlass::gemm::GemmCoord problem(m, n, k), tiles((m + Tile - 1) / Tile, (n + Tile - 1) / Tile, 1);
      const typename K::Params params(problem, tiles,
                                      RefA(half_ptr(a), cutlass::layout::RowMajor(k)),
                                      RefB(half_ptr(w), cutlass::layout::ColumnMajor(k)),
                                      RefC(half_ptr(bias ? bias : c), cutlass::layout::RowMajor(bias ? 0 : n)),
                                      RefC(half_ptr(c), cutlass::layout::RowMajor(n)));
      const int items = 8 * tiles.m() * ((tiles.n() + 7) / 8);
      const int smem = int(sizeof (typename K::SharedStorage));
      static const bool configured = cudaFuncSetAttribute(enc_gemm_kernel<K>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      static const int per_sm = persistent_blocks_per_sm("CT2_ENC_GEMM_BLOCKS");
      cudaStream_t stream = get_cuda_stream();
      if (per_sm > 0)
        enc_gemm_kernel<K><<<per_sm * sm_count(), K::kThreadCount, smem, stream>>>(params, work_counter(stream),
                                                                                  items);
      else
        enc_gemm_kernel<K><<<items, K::kThreadCount, smem, stream>>>(params, nullptr, items);
    }

    using EncLaunch = void (*)(const float16_t*, const float16_t*, float16_t*, int, int, int, const float16_t*);
    struct EncConfig {
      const char* name;                                  // <tile>x<k tile>s<stages>
      EncLaunch plain, gelu;
    };

#define CT2_ENC_CONFIG(T, KT, S) \
    {#T "x" #KT "s" #S, &enc_gemm_launch<T, KT, S, EncPlainOp>, &enc_gemm_launch<T, KT, S, EncBiasGeluOp>}

    // Shared memory: Tile * KTile * 4 bytes per stage (at most 96 KB per block here).
    static const EncConfig enc_configs[] = {
      CT2_ENC_CONFIG(64, 32, 6), CT2_ENC_CONFIG(64, 32, 4), CT2_ENC_CONFIG(64, 32, 3),
      CT2_ENC_CONFIG(64, 64, 4), CT2_ENC_CONFIG(64, 64, 3), CT2_ENC_CONFIG(64, 64, 6),
      CT2_ENC_CONFIG(128, 32, 4), CT2_ENC_CONFIG(128, 32, 3), CT2_ENC_CONFIG(128, 64, 3),
    };

    // CT2_ENC_GEMM_CFG picks the configuration (default 64x32s6, cuBLAS's own); an unknown name keeps it.
    static const EncConfig& enc_config() {
      static const EncConfig& config = [] () -> const EncConfig& {
        const std::string name = read_string_from_env("CT2_ENC_GEMM_CFG", "64x32s6");
        for (const EncConfig& c : enc_configs)
          if (name == c.name)
            return c;
        return enc_configs[0];
      }();
      return config;
    }

    void encoder_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k,
                      const float16_t* gelu_bias) {
      const EncConfig& config = enc_config();
      (gelu_bias ? config.gelu : config.plain)(a, w, c, int(m), int(n), int(k), gelu_bias);
    }

  }
}

#endif
