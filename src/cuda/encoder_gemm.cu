#ifndef CT2_USE_HIP

#include "cuda/encoder_gemm.h"

#include <cstdint>

#include "cuda/encoder_gemm_kernel.cuh"
#include "cuda/persistent.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static bool aligned16(const void* p) {
      return reinterpret_cast<uintptr_t>(p) % 16 == 0;
    }

    // CT2_ENC_GEMM_MIN_ROWS (default 1500, one 30 s window): smaller products keep cuBLAS, which may pick
    // another kernel (and another arithmetic) for them.
    bool encoder_gemm_applies(dim_t m, dim_t n, dim_t k, const void* a, const void* w, const void* c) {
      static const bool enabled = read_string_from_env("CT2_ENC_GEMM", "cutlass") != "cublas";
      static const dim_t min_rows = read_int_from_env("CT2_ENC_GEMM_MIN_ROWS", 1500);
      return enabled && m >= min_rows && n % 8 == 0 && k % 8 == 0
        && aligned16(a) && aligned16(w) && aligned16(c) && hmma_replicas_verified();
    }

    template <int Tile, int Stages, typename Op>
    static void enc_gemm_launch(const float16_t* a, const float16_t* w, float16_t* c, int m, int n, int k,
                                const float16_t* bias) {
      using K = EncGemmKernel<Tile, Stages, Op>;
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

    template <int Tile, typename Op>
    static void enc_gemm_stages(const float16_t* a, const float16_t* w, float16_t* c, int m, int n, int k,
                                const float16_t* bias) {
      static const int stages = read_int_from_env("CT2_ENC_GEMM_STAGES", 6);
      if (stages == 3)
        enc_gemm_launch<Tile, 3, Op>(a, w, c, m, n, k, bias);
      else if (stages == 4)
        enc_gemm_launch<Tile, 4, Op>(a, w, c, m, n, k, bias);
      else
        enc_gemm_launch<Tile, 6, Op>(a, w, c, m, n, k, bias);
    }

    // CT2_ENC_GEMM_TILE=128: 128 x 128 output tiles (half the operand traffic of 64 x 64).
    template <typename Op>
    static void enc_gemm_tiles(const float16_t* a, const float16_t* w, float16_t* c, int m, int n, int k,
                               const float16_t* bias) {
      static const int tile = read_int_from_env("CT2_ENC_GEMM_TILE", 64);
      if (tile == 128)
        enc_gemm_stages<128, Op>(a, w, c, m, n, k, bias);
      else
        enc_gemm_stages<64, Op>(a, w, c, m, n, k, bias);
    }

    void encoder_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k,
                      const float16_t* gelu_bias) {
      if (gelu_bias)
        enc_gemm_tiles<EncBiasGeluOp>(a, w, c, int(m), int(n), int(k), gelu_bias);
      else
        enc_gemm_tiles<EncPlainOp>(a, w, c, int(m), int(n), int(k), nullptr);
    }

  }
}

#endif
