// Which cuBLAS kernels run the Whisper decoder's self-attention of one window (CTranslate2's two strided batched
// products a step: scores = alpha q k^T, one query of 64 dims against t keys, and output = p v, batch 100 = 5 beams x
// 20 heads), for every t 1..448: one call of each, an NVTX range "t<t>" around them, to be read with Nsight Systems
// (nsys profile -t cuda,nvtx; nsys stats --report cuda_gpu_trace,nvtx_gpu_proj_trace). The kernels a fused exact
// replacement would have to reproduce, and where cuBLAS switches between them.
// "slots" as argument: the keys and values at the stream's slot stride (448 x 64, layers/slot_cache.h), not t x 64.
// usage: nsys profile -t cuda,nvtx -o sa ./selfattn_kernels [slots]
#include <cstdio>
#include <string>
#include <nvtx3/nvToolsExt.h>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kD = 64, kE = 100, kT = 448;

int main(int argc, char** argv) {
  const bool slots = argc > 1 && std::string(argv[1]) == "slots";
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *Q, *K, *V, *P, *C, *O;
  CK(cudaMalloc(&Q, 2ull * kE * kD)); CK(cudaMalloc(&K, 2ull * kE * kT * kD)); CK(cudaMalloc(&V, 2ull * kE * kT * kD));
  CK(cudaMalloc(&P, 2ull * kE * kT)); CK(cudaMalloc(&C, 2ull * kE * kT)); CK(cudaMalloc(&O, 2ull * kE * kD));
  fill<<<1024, 256>>>(Q, (size_t)kE * kD, 3u, -6, 1);
  fill<<<1024, 256>>>(K, (size_t)kE * kT * kD, 7u, -7, 1);
  fill<<<1024, 256>>>(V, (size_t)kE * kT * kD, 11u, -6, 2);
  fill<<<1024, 256>>>(P, (size_t)kE * kT, 13u, -14, -2);
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  for (int t = 1; t <= kT; ++t) {
    nvtxRangePushA(("t" + std::to_string(t)).c_str());
    const long long stride = (long long)(slots ? kT : t) * kD;
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, t, 1, kD, &scale, K, CUDA_R_16F, kD, stride,
                                  Q, CUDA_R_16F, kD, kD, &zero, C, CUDA_R_16F, t, t, kE, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, t, &one, V, CUDA_R_16F, kD, stride,
                                  P, CUDA_R_16F, t, t, &zero, O, CUDA_R_16F, kD, kD, kE, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());
    nvtxRangePop();
  }
  printf("ran t 1..%d\n", kT);
  return 0;
}
