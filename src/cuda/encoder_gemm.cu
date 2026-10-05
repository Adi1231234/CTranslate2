#ifndef CT2_USE_HIP

#include "cuda/encoder_gemm.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "cuda/encoder_gemm_kernel.cuh"
#include "cuda/persistent.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static bool aligned16(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 16 == 0;
    }

    // The width of the chain cuBLAS 12.9.2 runs this encoder product in on this device, 0 where no replica was
    // verified: sm_120, every encoder Dense layer in 16-wide steps (cutlass_80_tensorop_f16_s16816gemm..._64x64);
    // sm_89, the first feed-forward (5120 x 1280) in 8-wide steps (ampere_fp16_s1688gemm_fp16_128x128...,
    // tools/turing/kernels/encoder_ffn1_check.cu); its other products stay on cuBLAS (s16816 kernels near the
    // tensor pipe's peak, nothing to gain).
    static int replica_width(dim_t n, dim_t k) {
      if (cublas_verified_on(12, 0))
        return 16;
      if (cublas_verified_on(8, 9) && n == 5120 && k == 1280)
        return 8;
      return 0;
    }

    // CT2_ENC_GEMM_MIN_ROWS (default 1500, one 30 s window): smaller products keep cuBLAS, which may pick
    // another kernel (and another arithmetic) for them. k a multiple of 64: no k tile has a residue, in any
    // configuration below (a residue tile would change where the chain starts).
    bool encoder_gemm_applies(dim_t m, dim_t n, dim_t k, const void* a, const void* w, const void* c) {
      static const bool enabled = read_string_from_env("CT2_ENC_GEMM", "cublas") == "cutlass";
      static const dim_t min_rows = read_int_from_env("CT2_ENC_GEMM_MIN_ROWS", 1500);
      return enabled && m >= min_rows && n % 8 == 0 && k % 64 == 0
        && aligned16(a) && aligned16(w) && aligned16(c) && replica_width(n, k) > 0;
    }

    // enc_gemm_launch with CT2_ENC_GEMM_BLOCKS=<n>: persistent, n blocks per SM (cuda/persistent.h).
    template <int Tile, int KTile, int Stages, typename Op, int InstK>
    static void enc_gemm_run(const float16_t* a, const float16_t* w, float16_t* c, int m, int n, int k,
                             const float16_t* bias) {
      static const int per_sm = persistent_blocks_per_sm("CT2_ENC_GEMM_BLOCKS");
      cudaStream_t stream = get_cuda_stream();
      auto half_ptr = [](const float16_t* p) { return reinterpret_cast<const __half*>(p); };
      enc_gemm_launch<Tile, KTile, Stages, Op, InstK>(half_ptr(a), half_ptr(w), reinterpret_cast<__half*>(c), m, n,
                                                       k, half_ptr(bias), stream,
                                                       per_sm > 0 ? work_counter(stream) : nullptr,
                                                       per_sm * sm_count());
    }

    using EncLaunch = void (*)(const float16_t*, const float16_t*, float16_t*, int, int, int, const float16_t*);
    struct EncConfig {
      const char* name;                                  // <tile>x<k tile>s<stages>
      EncLaunch plain16, gelu16, plain8, gelu8;          // by the chain's width
    };

#define CT2_ENC_CONFIG(T, KT, S) \
    {#T "x" #KT "s" #S, &enc_gemm_run<T, KT, S, EncPlainOp, 16>, &enc_gemm_run<T, KT, S, EncBiasGeluOp, 16>, \
     &enc_gemm_run<T, KT, S, EncPlainOp, 8>, &enc_gemm_run<T, KT, S, EncBiasGeluOp, 8>}

    // Shared memory: Tile * KTile * 4 bytes per stage (at most 96 KB per block here).
    static const EncConfig enc_configs[] = {
      CT2_ENC_CONFIG(64, 32, 6), CT2_ENC_CONFIG(64, 32, 4), CT2_ENC_CONFIG(64, 32, 3),
      CT2_ENC_CONFIG(64, 64, 4), CT2_ENC_CONFIG(64, 64, 3), CT2_ENC_CONFIG(64, 64, 6),
      CT2_ENC_CONFIG(128, 32, 4), CT2_ENC_CONFIG(128, 32, 3), CT2_ENC_CONFIG(128, 64, 3),
    };

    // CT2_ENC_GEMM_CFG picks the configuration (default 64x32s6, cuBLAS's own on sm_120); an unknown name keeps it.
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
      const bool narrow = replica_width(n, k) == 8;
      const EncLaunch launch = gelu_bias ? (narrow ? config.gelu8 : config.gelu16)
                                         : (narrow ? config.plain8 : config.plain16);
      launch(a, w, c, int(m), int(n), int(k), gelu_bias);
    }

  }
}

#endif
