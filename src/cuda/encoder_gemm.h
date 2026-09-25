#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // c = a w^T for the Whisper encoder's Dense layers (fp16, a [m, k], w [n, k], c [m, n], fp32 accumulation),
    // with the arithmetic of the cuBLAS kernel they run on sm_120 with cuBLAS 12.9.2
    // (cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn): each output one mma.sync m16n8k16 chain
    // over k from zero, activations as the A operand, then rounded to fp16. The kernel is CUTLASS's own 64x64x32
    // tile of that configuration, so every output is computed as cuBLAS computes it, whatever the grid.
    // CT2_ENC_GEMM=cublas keeps cuBLAS; CT2_ENC_GEMM_BLOCKS=<n> runs it persistent with n blocks per SM
    // (cuda/persistent.h); CT2_ENC_GEMM_STAGES=3|4|6 (default 6, cuBLAS's) sets the shared-memory pipeline and
    // CT2_ENC_GEMM_TILE=64|128 the output tile (4 warps either way).
    bool encoder_gemm_applies(dim_t m, dim_t n, dim_t k, const void* a, const void* w, const void* c);

    // With gelu_bias, c = gelu(bias + c) as BiasAdd's fp16 GELU mode computes it from the rounded product
    // (ops/bias_add_vec.cuh), fused in the kernel's epilogue.
    void encoder_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k,
                      const float16_t* gelu_bias = nullptr);

  }
}
