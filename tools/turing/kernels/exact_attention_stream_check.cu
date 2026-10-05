// Bit-for-bit check and timing of the streamed encoder attention (src/ops/exact_attention_stream.cuh) in each
// block shape, against the three steps production replaces (the cuBLAS MatMul(trans_b, alpha 1/8), the library's
// softmax_rows, the cuBLAS MatMul of the probabilities and the values; as exact_attention_check.cu) for 1, 4 and 8
// clips of 20 heads, 1500 x 1500 x 64, three fills each; then from the fused projection and its bias (read by the
// streamed kernel itself) against exact_attention_qkv, 1..8 clips. Must end with TOTAL 0.
// usage: exact_attention_stream_check [timing repetitions, default 10]
#include <cstdio>
#include "probe_common.h"
#include "probe_data.cuh"
#include "ops/exact_attention_launch.cuh"
#include "ops/exact_attention_stream.cuh"

template <typename F> float time_us(F run, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  run(); CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) run();
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms * 1000 / reps;
}

struct Shape { const char* name; at::native::EasItems run; };
#define SHAPE(W, R, S) {#W " warps x " #R " tiles, " #S " stages", at::native::exact_attention_stream_items<W, R, S>}
static const Shape shapes[] = {SHAPE(8, 1, 3), SHAPE(8, 1, 2), SHAPE(4, 2, 3), SHAPE(4, 1, 4), SHAPE(8, 2, 2)};

unsigned long long diff(const __half* a, const __half* b, size_t count, unsigned long long* dc) {
  CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, count, dc);
  unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 10, m = 1500, n = 1500, d = 64, heads = 20;
  const float alpha = 0.125f;
  Probe p(160, m, n, d);                                   // dQ, dK, and dC for the scores
  __half *V, *O, *F, *X, *B; void* W;
  const size_t vs = 160ull * n * d;
  CK(cudaMalloc(&V, 2 * vs)); CK(cudaMalloc(&O, 2 * vs)); CK(cudaMalloc(&F, 2 * vs));
  CK(cudaMalloc(&W, at::native::exact_attention_workspace(160, n, true)));
  unsigned long long* dc; CK(cudaMalloc(&dc, 8));
  unsigned long long total = 0;
  for (int clips : {1, 4, 8}) {
    const int batch = clips * heads;
    auto three_steps = [&] {
      p.cublas_run(alpha, batch);
      at::native::softmax_rows<__half, at::native::SoftMaxForwardEpilogue>(0, p.dC, p.dC, batch * m, n, nullptr, true);
      const float one = 1.f, zero = 0.f;
      CK(cublasGemmStridedBatchedEx(p.h, CUBLAS_OP_N, CUBLAS_OP_N, d, m, n, &one, V, CUDA_R_16F, d, (long long)n * d,
                                    p.dC, CUDA_R_16F, n, (long long)m * n, &zero, O, CUDA_R_16F, d, (long long)m * d,
                                    batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    auto fused = [&] { at::native::exact_attention(p.dQ, p.dK, V, W, F, batch, 1, m, n, alpha, 0); };
    unsigned long long bad[8] = {};
    for (int f = 0; f < 3; ++f) {
      fill<<<1024, 256>>>(p.dQ, (size_t)batch * m * d, 23u * f + batch, -6 + f, 1 + f);
      fill<<<1024, 256>>>(p.dK, (size_t)batch * n * d, 211u * f + batch, -7 + f, 1 + f);
      fill<<<1024, 256>>>(V, (size_t)batch * n * d, 307u * f + batch, -8 + 2 * f, 2 + f);
      three_steps();
      for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s) {
        CK(cudaMemset(F, 0xff, 2 * vs));
        at::native::exact_attention_stream(shapes[s].run, p.dQ, p.dK, V, F, batch, 1, alpha, 0);
        CK(cudaGetLastError());
        bad[s] += diff(O, F, (size_t)batch * m * d, dc);
      }
    }
    printf("%d clips: MatMul + SoftMax + MatMul %8.1f us, exact_attention %8.1f us\n", clips,
           time_us(three_steps, reps), time_us(fused, reps));
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s) {
      total += bad[s];
      const float ts = time_us([&] { at::native::exact_attention_stream(shapes[s].run, p.dQ, p.dK, V, F, batch, 1,
                                                                         alpha, 0); }, reps);
      printf("  %-28s %llu of %llu mismatched, %8.1f us\n", shapes[s].name, bad[s], 3ull * batch * m * d, ts);
    }
  }
  // From the fused projection x [clips, n, 3 * heads * 64] and its bias (the key part zero, as Whisper's).
  const int row = 3 * heads * d;
  CK(cudaMalloc(&X, 2ull * 8 * n * row)); CK(cudaMalloc(&B, 2ull * row));
  for (int clips = 1; clips <= 8; ++clips) {
    fill<<<1024, 256>>>(X, (size_t)clips * n * row, 97u * clips, -6, 2);
    fill<<<16, 256>>>(B, (size_t)row, 131u * clips, -8, 0);
    set_bits<<<4, 256>>>(B + heads * d, (size_t)heads * d, 0);
    const size_t count = (size_t)clips * heads * n * d;
    auto qkv_path = [&] { at::native::exact_attention_qkv(X, B, W, O, clips, heads, n, alpha, 0); };
    auto stream_path = [&](const Shape& s) {
      at::native::exact_attention_stream_qkv(s.run, X, B, F, clips, heads, n, alpha, 0);
    };
    qkv_path();
    printf("qkv %d clips: exact_attention_qkv %8.1f us;", clips, time_us(qkv_path, reps));
    for (const Shape& s : shapes) {
      CK(cudaMemset(F, 0xff, 2 * vs));
      stream_path(s);
      CK(cudaGetLastError());
      const unsigned long long x = diff(O, F, count, dc);
      total += x;
      printf(" %llu mismatched %7.1f us |", x, time_us([&] { stream_path(s); }, reps));
    }
    printf("\n");
  }
  printf("TOTAL %llu mismatches\n", total);
  return 0;
}
