// The cuBLASLt functions the library calls (cuda/encoder_lt.cc), loaded from the cuBLASLt library at run time as
// cublas_stub.cc loads cuBLAS (CUDA_DYNAMIC_LOADING: no link-time dependency on the CUDA libraries).
#include <stdexcept>
#include <string>

#include <cublasLt.h>

#define LT_STR_HELPER(x) #x
#define LT_STR(x) LT_STR_HELPER(x)

#ifdef _WIN32
#  include <windows.h>
#  define CUBLASLT_LIBNAME "cublasLt64_" LT_STR(CUBLAS_VER_MAJOR) ".dll"
#else
#  include <dlfcn.h>
#  define CUBLASLT_LIBNAME "libcublasLt.so." LT_STR(CUBLAS_VER_MAJOR)
#endif

namespace ctranslate2 {

  static void* cublaslt_handle() {
    static void* handle = [] {
#ifdef _WIN32
      void* h = static_cast<void*>(LoadLibraryA(CUBLASLT_LIBNAME));
#else
      void* h = dlopen(CUBLASLT_LIBNAME, RTLD_LAZY);
#endif
      if (!h)
        throw std::runtime_error("Library " + std::string(CUBLASLT_LIBNAME) + " is not found or cannot be loaded");
      return h;
    }();
    return handle;
  }

  template <typename Signature>
  static Signature cublaslt_symbol(const char* name) {
#ifdef _WIN32
    void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(cublaslt_handle()), name));
#else
    void* symbol = dlsym(cublaslt_handle(), name);
#endif
    if (!symbol)
      throw std::runtime_error("Cannot load symbol " + std::string(name) + " from " + CUBLASLT_LIBNAME);
    return reinterpret_cast<Signature>(symbol);
  }

}

#define CT2_LT_FORWARD(name, params, args)                                              \
  cublasStatus_t name params {                                                         \
    static const auto f = ctranslate2::cublaslt_symbol<cublasStatus_t (*) params>(#name); \
    return f args;                                                                     \
  }

extern "C" {

  CT2_LT_FORWARD(cublasLtCreate, (cublasLtHandle_t* handle), (handle))
  CT2_LT_FORWARD(cublasLtMatmulDescCreate,
                 (cublasLtMatmulDesc_t* desc, cublasComputeType_t compute, cudaDataType_t scale),
                 (desc, compute, scale))
  CT2_LT_FORWARD(cublasLtMatmulDescDestroy, (cublasLtMatmulDesc_t desc), (desc))
  CT2_LT_FORWARD(cublasLtMatmulDescSetAttribute,
                 (cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t size),
                 (desc, attr, buf, size))
  CT2_LT_FORWARD(cublasLtMatrixLayoutCreate,
                 (cublasLtMatrixLayout_t* layout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld),
                 (layout, type, rows, cols, ld))
  CT2_LT_FORWARD(cublasLtMatrixLayoutDestroy, (cublasLtMatrixLayout_t layout), (layout))
  CT2_LT_FORWARD(cublasLtMatmulAlgoInit,
                 (cublasLtHandle_t handle, cublasComputeType_t compute, cudaDataType_t scale, cudaDataType_t a,
                  cudaDataType_t b, cudaDataType_t c, cudaDataType_t d, int id, cublasLtMatmulAlgo_t* algo),
                 (handle, compute, scale, a, b, c, d, id, algo))
  CT2_LT_FORWARD(cublasLtMatmulAlgoConfigSetAttribute,
                 (cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void* buf,
                  size_t size),
                 (algo, attr, buf, size))
  CT2_LT_FORWARD(cublasLtMatmulAlgoCheck,
                 (cublasLtHandle_t handle, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t a,
                  cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
                  const cublasLtMatmulAlgo_t* algo, cublasLtMatmulHeuristicResult_t* result),
                 (handle, desc, a, b, c, d, algo, result))
  CT2_LT_FORWARD(cublasLtMatmul,
                 (cublasLtHandle_t handle, cublasLtMatmulDesc_t desc, const void* alpha, const void* a,
                  cublasLtMatrixLayout_t la, const void* b, cublasLtMatrixLayout_t lb, const void* beta,
                  const void* c, cublasLtMatrixLayout_t lc, void* d, cublasLtMatrixLayout_t ld,
                  const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspace_size, cudaStream_t stream),
                 (handle, desc, alpha, a, la, b, lb, beta, c, lc, d, ld, algo, workspace, workspace_size, stream))

}
