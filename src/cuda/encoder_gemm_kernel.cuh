#pragma once

// The kernel of encoder_gemm.h: CUTLASS 2.x's threadblock-level main loop and epilogue for one 64x64 output tile
// (the body of cutlass::gemm::kernel::Gemm without split-K), run for one tile per block or, persistent, for the
// tiles a block takes from the work counter. Tiles are numbered as cuBLAS's grid orders them
// (GemmIdentityThreadblockSwizzle<8>: 8 column tiles per row tile, all row tiles, then the next 8 columns).

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm.h"

#include "cuda/helpers.h"
#include "cuda/persistent.cuh"

namespace ctranslate2 {
  namespace cuda {

    using EncHalf = cutlass::half_t;
    using EncPlainOp = cutlass::epilogue::thread::LinearCombination<
      EncHalf, 8, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;   // c = half_rn(acc)

    // c = half(gelu(float(__hadd(bias, half_rn(acc))))): the product rounded as above, then BiasAdd's GELU mode.
    // The source operand is the bias, read with a row stride of 0.
    struct EncBiasGeluOp : EncPlainOp {
      using EncPlainOp::EncPlainOp;
      CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
      CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc, FragmentSource const& bias) const {
        FragmentOutput out;
        const gelu_func<__half> gelu;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
          const EncHalf b = bias[i];
          const __half s = __hadd(b.to_half(), __float2half_rn(acc[i]));
          out[i] = EncHalf(__float2half_rn(gelu(__half2float(s))));
        }
        return out;
      }
      CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const {
        return EncPlainOp::operator()(acc);
      }
    };

    // Tile x Tile outputs per block, 4 warps of Tile/2 x Tile/2 (any tile keeps each output's chain over k).
    template <int Tile, int Stages, typename Op>
    using EncGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      EncHalf, cutlass::layout::RowMajor, 8, EncHalf, cutlass::layout::ColumnMajor, 8,
      EncHalf, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<Tile, Tile, 32>, cutlass::gemm::GemmShape<Tile / 2, Tile / 2, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>, Op, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      Stages, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;

    template <typename K>
    __device__ __forceinline__ void enc_gemm_tile(typename K::Params const& params,
                                                  typename K::SharedStorage& shared, int tm, int tn) {
      using Mma = typename K::Mma;
      using Epilogue = typename K::Epilogue;
      const int k = params.problem_size.k(), thread = threadIdx.x;
      const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x % 32;
      typename Mma::IteratorA iterator_a(params.params_A, params.ref_A.data(), {params.problem_size.m(), k},
                                         thread, cutlass::MatrixCoord{tm * Mma::Shape::kM, 0});
      typename Mma::IteratorB iterator_b(params.params_B, params.ref_B.data(), {k, params.problem_size.n()},
                                         thread, cutlass::MatrixCoord{0, tn * Mma::Shape::kN});
      Mma mma(shared.main_loop, thread, warp, lane);
      typename Mma::FragmentC acc;
      acc.clear();
      mma((k + Mma::Shape::kK - 1) / Mma::Shape::kK, acc, iterator_a, iterator_b, acc);
      const cutlass::MatrixCoord offset(tm * Mma::Shape::kM, tn * Mma::Shape::kN);
      typename Epilogue::OutputTileIterator iterator_c(params.params_C, params.ref_C.data(),
                                                       params.problem_size.mn(), thread, offset);
      typename Epilogue::OutputTileIterator iterator_d(params.params_D, params.ref_D.data(),
                                                       params.problem_size.mn(), thread, offset);
      typename Epilogue::OutputOp op(params.output_op);
      Epilogue epilogue(shared.epilogue, thread, warp, lane);
      epilogue(op, iterator_d, acc, iterator_c);
    }

    // items = 8 * row tiles * column groups; without a counter, block b computes item b.
    template <typename K>
    __global__ void __launch_bounds__(K::kThreadCount)
    enc_gemm_kernel(typename K::Params params, unsigned* counter, int items) {
      extern __shared__ __align__(16) unsigned char enc_smem[];
      auto& shared = *reinterpret_cast<typename K::SharedStorage*>(enc_smem);
      __shared__ int slot;
      const int tiles_m = params.grid_tiled_shape.m(), tiles_n = params.grid_tiled_shape.n();
      for (int item = counter ? next_work_item(counter, items, slot) : int(blockIdx.x); item < items;
           item = counter ? next_work_item(counter, items, slot) : items) {
        const int group = item / (8 * tiles_m), r = item % (8 * tiles_m);
        const int tn = 8 * group + r % 8;
        if (tn < tiles_n)
          enc_gemm_tile<K>(params, shared, r / 8, tn);
      }
    }

  }
}
