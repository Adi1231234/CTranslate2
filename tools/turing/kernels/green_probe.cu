// Can the store PC's GPU be split between the encoder and the decoder? In pipe8 the two run side by side and
// slow each other down (encoder GEMMs 9.5 vs 6.8 s alone, the decoder's small GEMMs ~2x). Green contexts
// (CUDA 12.4+ driver API) give a stream a fixed set of SMs. Times a decoder-like chain (cuBLAS 40 x 1280 x
// 1280 GEMMs, each on other weights, so read from DRAM as in production) and an encoder-like GEMM
// (12000 x 3840 x 1280), alone and together: on the whole GPU (two streams, the decoder's at high priority,
// as pipe8 runs them) and on disjoint SM partitions.
// usage: green_probe [decoder SMs, default 12]   (build for sm_120; links the driver API)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "probe_common.h"
#pragma comment(lib, "cuda")

#define CU(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) { const char* _s = nullptr; cuGetErrorString(_r, &_s); \
  fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x, (int)_r, _s ? _s : ""); exit(1); } } while (0)

struct Gemm { const __half *A, *W; __half* C; int m, n, k; };

static __half* zeros(size_t count) {
  __half* p = nullptr;
  CK(cudaMalloc(&p, sizeof(__half) * count));
  CK(cudaMemset(p, 0, sizeof(__half) * count));
  return p;
}

static void run(cublasHandle_t h, const Gemm& g) {
  const float alpha = 1.f, beta = 0.f;
  CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, g.n, g.m, g.k, &alpha, g.W, CUDA_R_16F, g.k, g.A, CUDA_R_16F,
                  g.k, &beta, g.C, CUDA_R_16F, g.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct Side { cudaStream_t stream; cublasHandle_t handle; };

static Side side(CUstream stream) {
  Side s{(cudaStream_t)stream, nullptr};
  CK(cublasCreate(&s.handle));
  CK(cublasSetStream(s.handle, s.stream));
  return s;
}

// A stream on a green context made of the given SM resources.
static CUstream green_stream(CUdevice dev, const CUdevResource* sms, unsigned count, int priority) {
  CUdevResourceDesc desc;
  CU(cuDevResourceGenerateDesc(&desc, const_cast<CUdevResource*>(sms), count));
  CUgreenCtx ctx;
  CU(cuGreenCtxCreate(&ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CUstream stream;
  CU(cuGreenCtxStreamCreate(&stream, ctx, CU_STREAM_NON_BLOCKING, priority));
  return stream;
}

static const int dec_weights = 64, dec_count = 3000, enc_count = 24;
static __half *dA, *dW, *dC, *eA, *eW, *eC;

static void queue_decoder(const Side& s) {
  for (int i = 0; i < dec_count; ++i)
    run(s.handle, {dA, dW + (size_t)(i % dec_weights) * 1280 * 1280, dC, 40, 1280, 1280});
}
static void queue_encoder(const Side& s) {
  for (int i = 0; i < enc_count; ++i)
    run(s.handle, {eA, eW, eC, 12000, 3840, 1280});
}

// Queues the decoder and/or encoder work and returns each side's time (ms) from a common start.
static void measure(const char* label, const Side* dec, const Side* enc) {
  cudaEvent_t start, dec_end, enc_end;
  CK(cudaEventCreate(&start)); CK(cudaEventCreate(&dec_end)); CK(cudaEventCreate(&enc_end));
  const Side& first = dec ? *dec : *enc;
  CK(cudaEventRecord(start, first.stream));
  if (dec && enc) CK(cudaStreamWaitEvent(enc->stream, start, 0));
  if (enc) { queue_encoder(*enc); CK(cudaEventRecord(enc_end, enc->stream)); }
  if (dec) { queue_decoder(*dec); CK(cudaEventRecord(dec_end, dec->stream)); }
  CK(cudaDeviceSynchronize());
  float d = 0, e = 0;
  if (dec) CK(cudaEventElapsedTime(&d, start, dec_end));
  if (enc) CK(cudaEventElapsedTime(&e, start, enc_end));
  printf("%-28s decoder %7.1f ms (%5.1f us/GEMM)   encoder %7.1f ms (%6.3f ms/GEMM)\n", label, d,
         dec ? 1000 * d / dec_count : 0.f, e, enc ? e / enc_count : 0.f);
}

int main(int argc, char** argv) {
  const unsigned dec_sms = argc > 1 ? atoi(argv[1]) : 12;
  CK(cudaFree(nullptr));                                        // the primary context, current
  dA = zeros(40 * 1280); dW = zeros((size_t)dec_weights * 1280 * 1280); dC = zeros(40 * 1280);
  eA = zeros(12000ull * 1280); eW = zeros(3840ull * 1280); eC = zeros(12000ull * 3840);

  int lo = 0, hi = 0;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t ds, es;
  CK(cudaStreamCreateWithPriority(&ds, cudaStreamNonBlocking, hi));
  CK(cudaStreamCreateWithPriority(&es, cudaStreamNonBlocking, lo));
  const Side dec_all = side((CUstream)ds), enc_all = side((CUstream)es);
  measure("warm-up", &dec_all, &enc_all);
  measure("whole GPU: decoder alone", &dec_all, nullptr);
  measure("whole GPU: encoder alone", nullptr, &enc_all);
  measure("whole GPU: together", &dec_all, &enc_all);

  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUdevResource all, part, rest;
  CU(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
  unsigned groups = 1;
  CU(cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, dec_sms));
  printf("SMs: %u = decoder %u + encoder %u\n", all.sm.smCount, part.sm.smCount, rest.sm.smCount);
  const Side dec_part = side(green_stream(dev, &part, 1, hi)), enc_part = side(green_stream(dev, &rest, 1, lo));
  measure("partitions: warm-up", &dec_part, &enc_part);
  measure("partitions: decoder alone", &dec_part, nullptr);
  measure("partitions: encoder alone", nullptr, &enc_part);
  measure("partitions: together", &dec_part, &enc_part);

  // Only the encoder confined, to 8-SM groups (and the 4 SMs left over): the decoder keeps the whole GPU.
  CUdevResource eight[4], left;
  unsigned n8 = 4;
  CU(cuDevSmResourceSplitByCount(eight, &n8, &all, &left, 0, 8));
  CUdevResource with_left[5] = {eight[0], eight[1], eight[2], eight[3], left};
  for (unsigned g = 3; g <= n8; ++g) {
    for (int extra = 0; extra <= 1; ++extra) {
      CUdevResource set[5];
      for (unsigned i = 0; i < g; ++i) set[i] = with_left[i];
      if (extra) set[g] = left;
      const Side enc = side(green_stream(dev, set, g + extra, lo));
      char label[64];
      snprintf(label, sizeof label, "encoder on %u SMs: alone", 8 * g + (extra ? left.sm.smCount : 0));
      measure(label, nullptr, &enc);
      snprintf(label, sizeof label, "  + decoder whole GPU");
      measure(label, &dec_all, &enc);
    }
  }
  return 0;
}
