// Does cuBLAS give the same bits when a batched product reads its batch entries through a pointer array
// (cublasGemmBatchedEx) as through strides (cublasGemmStridedBatchedEx, as CTranslate2 calls it)? For the Whisper
// decoder's cross-attention of sampled hypotheses (one query a row against 1500 keys of 64 dims, batch = rows x 20
// heads): scores C = alpha q k^T and output O = p v. If so, the hypotheses of a clip can read one copy of the clip's
// keys and values (pointers) instead of one copy each. 3 fills per batch size; also the batch sizes' run times.
// usage: ptrbatch_probe
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int kKeys = 1500, kD = 64;

template <typename F> float time_us(F run, int reps = 50) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const int batches[] = {20, 40, 60, 80, 100, 160, 200, 300, 400, 500, 800};
  const int maxb = 800;
  __half *Q, *K, *V, *P, *C1, *C2, *O1, *O2;
  CK(cudaMalloc(&Q, 2ull * maxb * kD)); CK(cudaMalloc(&K, 2ull * maxb * kKeys * kD));
  CK(cudaMalloc(&V, 2ull * maxb * kKeys * kD)); CK(cudaMalloc(&P, 2ull * maxb * kKeys));
  CK(cudaMalloc(&C1, 2ull * maxb * kKeys)); CK(cudaMalloc(&C2, 2ull * maxb * kKeys));
  CK(cudaMalloc(&O1, 2ull * maxb * kD)); CK(cudaMalloc(&O2, 2ull * maxb * kD));
  const void **pa, **pb; void** pc;
  CK(cudaMalloc(&pa, sizeof(void*) * maxb)); CK(cudaMalloc(&pb, sizeof(void*) * maxb)); CK(cudaMalloc(&pc, sizeof(void*) * maxb));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  auto diff = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  auto set_ptrs = [&](const __half* a, size_t sa, const __half* b, size_t sb, __half* c, size_t sc, int n) {
    std::vector<const void*> ha(n), hb(n); std::vector<void*> hc(n);
    for (int i = 0; i < n; ++i) { ha[i] = a + i * sa; hb[i] = b + i * sb; hc[i] = c + i * sc; }
    CK(cudaMemcpy(pa, ha.data(), sizeof(void*) * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pb, hb.data(), sizeof(void*) * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pc, hc.data(), sizeof(void*) * n, cudaMemcpyHostToDevice));
  };
  const float scale = 0.125f, one = 1.f, zero = 0.f;
  unsigned long long total = 0;
  for (int batch : batches) {
    unsigned long long qbad = 0, vbad = 0;
    for (int f = 0; f < 3; ++f) {
      fill<<<1024, 256>>>(Q, (size_t)batch * kD, 3u + f, -6 + f, 1 + f);
      fill<<<1024, 256>>>(K, (size_t)batch * kKeys * kD, 7u + f, -7 + f, 1 + f);
      fill<<<1024, 256>>>(V, (size_t)batch * kKeys * kD, 11u + f, -6 + f, 2 + f);
      fill<<<1024, 256>>>(P, (size_t)batch * kKeys, 13u + f, -14 + f, -6 + f);
      // scores, as Probe::cublas_run: C[b] (1 x 1500) = scale * Q[b] (1 x 64) . K[b]^T
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kKeys, 1, kD, &scale, K, CUDA_R_16F, kD,
                                    (long long)kKeys * kD, Q, CUDA_R_16F, kD, kD, &zero, C1, CUDA_R_16F, kKeys,
                                    kKeys, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      set_ptrs(K, (size_t)kKeys * kD, Q, kD, C2, kKeys, batch);
      CK(cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kKeys, 1, kD, &scale, pa, CUDA_R_16F, kD, pb, CUDA_R_16F,
                             kD, &zero, pc, CUDA_R_16F, kKeys, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      qbad += diff(C1, C2, (size_t)batch * kKeys);
      // output, as MatMul(attn, values): O[b] (1 x 64) = P[b] (1 x 1500) . V[b] (1500 x 64)
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, V, CUDA_R_16F, kD,
                                    (long long)kKeys * kD, P, CUDA_R_16F, kKeys, kKeys, &zero, O1, CUDA_R_16F, kD,
                                    kD, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      set_ptrs(V, (size_t)kKeys * kD, P, kKeys, O2, kD, batch);
      CK(cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, pa, CUDA_R_16F, kD, pb, CUDA_R_16F,
                             kKeys, &zero, pc, CUDA_R_16F, kD, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      vbad += diff(O1, O2, (size_t)batch * kD);
    }
    total += qbad + vbad;
    const float ts = time_us([&] {
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, V, CUDA_R_16F, kD,
                                    (long long)kKeys * kD, P, CUDA_R_16F, kKeys, kKeys, &zero, O1, CUDA_R_16F, kD,
                                    kD, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); });
    const float tp = time_us([&] {
      CK(cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, pa, CUDA_R_16F, kD, pb, CUDA_R_16F,
                             kKeys, &zero, pc, CUDA_R_16F, kD, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); });
    printf("batch %4d: scores %llu, output %llu mismatched; output strided %.1f us, pointers %.1f us\n",
           batch, qbad, vbad, ts, tp);
  }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
