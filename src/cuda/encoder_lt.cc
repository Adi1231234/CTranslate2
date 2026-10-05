#include "cuda/encoder_lt.h"

#include <cublasLt.h>

#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    struct LtChoice {
      dim_t n, k;
      int algo, tile, stages, swizzle;
    };

    // The exact algorithms with less energy than cuBLAS's pick, by product (encoder_lt.h). Rows: 1500 and up (one
    // 30 s window a clip), where cuBLAS runs the product as one chain over k (encoder_ffn1_check.cu).
    static const LtChoice* choice(dim_t m, dim_t n, dim_t k) {
      static const LtChoice choices[] = {{5120, 1280, 21, 24, 9, 1}};
      static const bool enabled = read_bool_from_env("CT2_ENC_LT", true) && cublas_verified_on(8, 9);
      if (!enabled || m < 1500)
        return nullptr;
      for (const LtChoice& c : choices)
        if (c.n == n && c.k == k)
          return &c;
      return nullptr;
    }

    static cublasLtHandle_t lt_handle() {
      static thread_local cublasLtHandle_t handle = [] {
        cublasLtHandle_t h = nullptr;
        CUBLAS_CHECK(cublasLtCreate(&h));
        return h;
      }();
      return handle;
    }

    // The operands as the cuBLAS call has them (column-major: C^T [n, m] = W^T [n, k] A^T [k, m]), the algorithm
    // configured, its workspace (a thread's buffer of the size it asks).
    bool encoder_lt_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k) {
      const LtChoice* pick = choice(m, n, k);
      if (!pick)
        return false;
      cublasLtMatmulDesc_t desc;
      CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof op_t));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof op_n));
      cublasLtMatrixLayout_t lw, la, lc;
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lw, CUDA_R_16F, k, n, k));
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, k, m, k));
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, n, m, n));
      cublasLtMatmulAlgo_t algo;
      CUBLAS_CHECK(cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F,
                                          CUDA_R_16F, CUDA_R_16F, pick->algo, &algo));
      const int split = 1;
      const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE, swizzle = pick->swizzle;
      CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &pick->tile,
                                                        sizeof pick->tile));
      CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &pick->stages,
                                                        sizeof pick->stages));
      CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof split));
      CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
                                                        sizeof reduction));
      CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle,
                                                        sizeof swizzle));
      cublasLtMatmulHeuristicResult_t check;
      const bool usable = cublasLtMatmulAlgoCheck(lt_handle(), desc, lw, la, lc, lc, &algo, &check)
        == CUBLAS_STATUS_SUCCESS;
      if (usable) {
        static thread_local void* workspace = nullptr;
        static thread_local size_t workspace_size = 0;
        if (check.workspaceSize > workspace_size) {
          if (workspace)
            CUDA_CHECK(cudaFree(workspace));
          CUDA_CHECK(cudaMalloc(&workspace, check.workspaceSize));
          workspace_size = check.workspaceSize;
        }
        const float one = 1.f, zero = 0.f;
        CUBLAS_CHECK(cublasLtMatmul(lt_handle(), desc, &one, w, lw, a, la, &zero, c, lc, c, lc, &algo, workspace,
                                    workspace_size, get_cuda_stream()));
      }
      cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatrixLayoutDestroy(la);
      cublasLtMatrixLayoutDestroy(lw);
      cublasLtMatmulDescDestroy(desc);
      return usable;
    }

  }
}
