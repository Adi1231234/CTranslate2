// The cross-attention of a long recording's sampled ladder rows (src/cuda/shared_memory_rows.cu: each row one query a
// head against its clip's 1500 keys of 64 dims, through pointer arrays, batch = rows x 20 heads, 1..5 rows a call):
// long13's profile put these products at ~13% of the GPU's time, a clip's memory read once per row. A kernel reading
// it once for all the rows needs cuBLAS's arithmetic for these calls on this GPU. For batches 20..100: one call each of
// the scores (alpha q k^T) and the output (p v) in an NVTX range ("s<batch>", "v<batch>") to name the kernels with
// Nsight Systems (box/nsys_probe.sh), the same entries through strides (the stock layout: every row its own copy),
// and candidate arithmetics against cuBLAS, mismatches counted over 3 fills.
//   scores: S0 sequential over the 64 dims; S1 sm_75's (attention_scores_k64.cuh); S2 an mma.sync m16n8k16 chain
//   over 16-dim groups, the query a row of A; S3 the same with the keys rows of A; S4 each group from zero, the
//   groups added in order
//   output: V0 sequential over the keys; V1 an mma chain over 16-key groups (the probabilities a row of A); V2/V3
//   T threads each summing keys r, r + T, ... then a tree (V2) or in order (V3), T = 16..256; V4 T contiguous chunks
//   then a tree
// usage: ladder_cross_probe
#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include "probe_common.h"
#include "probe_data.cuh"
#include "gemv_candidates.cuh"

constexpr int kKeys = 1500, kD = 64, kHeads = 20, kMaxE = 100;
constexpr float kScale = 0.125f;

// Scores, one thread per (entry, key): S0, S1, S4 (S4 emulates per-group chains with sequential sums: a guide only).
__global__ void scores_scalar(const __half* K, const __half* Q, __half* C, int entries, int mode) {
  const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kKeys || e >= entries) return;
  const __half* k = K + ((size_t)(e % kHeads) * kKeys + i) * kD;
  const __half* q = Q + (size_t)e * kD;
  float sum = 0.f;
  if (mode == 0) {
    for (int d = 0; d < kD; ++d) sum = fmaf(f(k[d]), f(q[d]), sum);
  } else if (mode == 1) {
    for (int r = 0; r < 16; ++r) {
      float p = f(k[r]) * f(q[r]);
      p = fmaf(f(k[r + 16]), f(q[r + 16]), p); p = fmaf(f(k[r + 32]), f(q[r + 32]), p);
      p = fmaf(f(k[r + 48]), f(q[r + 48]), p);
      sum += p;
    }
  } else {
    for (int g = 0; g < 4; ++g) {
      float part = 0.f;
      for (int d = 16 * g; d < 16 * g + 16; ++d) part = fmaf(f(k[d]), f(q[d]), part);
      sum += part;
    }
  }
  C[(size_t)e * kKeys + i] = __float2half_rn(kScale * sum);
}

// Scores by mma: a warp per (entry, 8 keys). mode 2: A = the query (row 0, the rest zero), B = 8 keys' columns;
// mode 3: A = 16 keys' rows, B = the query (column 0).
__global__ void scores_mma(const __half* K, const __half* Q, __half* C, int entries, int mode) {
  const int e = blockIdx.y, lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const __half* kb = K + (size_t)(e % kHeads) * kKeys * kD;
  const __half* q = Q + (size_t)e * kD;
  const __half zero = __float2half(0.f);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (mode == 2) {
    const int i0 = blockIdx.x * 8;
    auto kk = [&](int i, int d) { return i < kKeys ? kb[(size_t)i * kD + d] : zero; };
    for (int d0 = 0; d0 < kD; d0 += 16) {
      const unsigned a0 = g == 0 ? pack(q[d0 + 2 * t], q[d0 + 2 * t + 1]) : 0u;
      const unsigned a2 = g == 0 ? pack(q[d0 + 8 + 2 * t], q[d0 + 8 + 2 * t + 1]) : 0u;
      const unsigned b0 = pack(kk(i0 + g, d0 + 2 * t), kk(i0 + g, d0 + 2 * t + 1));
      const unsigned b1 = pack(kk(i0 + g, d0 + 8 + 2 * t), kk(i0 + g, d0 + 8 + 2 * t + 1));
      mma16816(acc, a0, 0u, a2, 0u, b0, b1);
    }
    if (g == 0)
      for (int c = 0; c < 2; ++c) {
        const int i = i0 + 2 * t + c;
        if (i < kKeys) C[(size_t)e * kKeys + i] = __float2half_rn(kScale * acc[c]);
      }
  } else {
    const int i0 = blockIdx.x * 16;
    auto kk = [&](int i, int d) { return i < kKeys ? kb[(size_t)i * kD + d] : zero; };
    for (int d0 = 0; d0 < kD; d0 += 16) {
      const unsigned a0 = pack(kk(i0 + g, d0 + 2 * t), kk(i0 + g, d0 + 2 * t + 1));
      const unsigned a1 = pack(kk(i0 + g + 8, d0 + 2 * t), kk(i0 + g + 8, d0 + 2 * t + 1));
      const unsigned a2 = pack(kk(i0 + g, d0 + 8 + 2 * t), kk(i0 + g, d0 + 8 + 2 * t + 1));
      const unsigned a3 = pack(kk(i0 + g + 8, d0 + 8 + 2 * t), kk(i0 + g + 8, d0 + 8 + 2 * t + 1));
      const unsigned b0 = g == 0 ? pack(q[d0 + 2 * t], q[d0 + 2 * t + 1]) : 0u;
      const unsigned b1 = g == 0 ? pack(q[d0 + 8 + 2 * t], q[d0 + 8 + 2 * t + 1]) : 0u;
      mma16816(acc, a0, a1, a2, a3, b0, b1);
    }
    if (t == 0) {                                            // column 0: acc[0] row g, acc[2] row g + 8
      if (i0 + g < kKeys) C[(size_t)e * kKeys + i0 + g] = __float2half_rn(kScale * acc[0]);
      if (i0 + g + 8 < kKeys) C[(size_t)e * kKeys + i0 + g + 8] = __float2half_rn(kScale * acc[2]);
    }
  }
}

// Output, one thread per (entry, dim): V0, V2/V3 (strided, T), V4 (contiguous chunks, T).
__global__ void output_scalar(const __half* V, const __half* P, __half* O, int entries, int mode, int T) {
  const int e = blockIdx.x, d = threadIdx.x;
  if (e >= entries) return;
  const __half* v = V + (size_t)(e % kHeads) * kKeys * kD;
  const __half* p = P + (size_t)e * kKeys;
  float part[256];
  float sum = 0.f;
  if (mode == 0) {
    for (int i = 0; i < kKeys; ++i) sum = fmaf(f(p[i]), f(v[(size_t)i * kD + d]), sum);
  } else {
    const int chunk = (kKeys + T - 1) / T;
    for (int r = 0; r < T; ++r) {
      float s = 0.f;
      if (mode == 4) {
        for (int i = r * chunk; i < min(kKeys, (r + 1) * chunk); ++i) s = fmaf(f(p[i]), f(v[(size_t)i * kD + d]), s);
      } else {
        for (int i = r; i < kKeys; i += T) s = fmaf(f(p[i]), f(v[(size_t)i * kD + d]), s);
      }
      part[r] = s;
    }
    if (mode == 3) {
      for (int r = 0; r < T; ++r) sum += part[r];
    } else {
      for (int off = T / 2; off > 0; off /= 2)
        for (int r = 0; r < off; ++r) part[r] += part[r + off];
      sum = part[0];
    }
  }
  O[(size_t)e * kD + d] = __float2half_rn(sum);
}

// Output by an mma chain over 16-key groups: a warp per (entry, 8 dims), the probabilities row 0 of A.
__global__ void output_mma(const __half* V, const __half* P, __half* O, int entries) {
  const int e = blockIdx.y, lane = threadIdx.x, g = lane >> 2, t = lane & 3, d0 = blockIdx.x * 8;
  const __half* v = V + (size_t)(e % kHeads) * kKeys * kD;
  const __half* p = P + (size_t)e * kKeys;
  const __half zero = __float2half(0.f);
  auto pp = [&](int i) { return i < kKeys ? p[i] : zero; };
  auto vv = [&](int i, int d) { return i < kKeys ? v[(size_t)i * kD + d] : zero; };
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i0 = 0; i0 < kKeys; i0 += 16) {
    const unsigned a0 = g == 0 ? pack(pp(i0 + 2 * t), pp(i0 + 2 * t + 1)) : 0u;
    const unsigned a2 = g == 0 ? pack(pp(i0 + 8 + 2 * t), pp(i0 + 8 + 2 * t + 1)) : 0u;
    const unsigned b0 = pack(vv(i0 + 2 * t, d0 + g), vv(i0 + 2 * t + 1, d0 + g));
    const unsigned b1 = pack(vv(i0 + 8 + 2 * t, d0 + g), vv(i0 + 8 + 2 * t + 1, d0 + g));
    mma16816(acc, a0, 0u, a2, 0u, b0, b1);
  }
  if (g == 0)
    for (int c = 0; c < 2; ++c) O[(size_t)e * kD + d0 + 2 * t + c] = __float2half_rn(acc[c]);
}

__global__ void scores_general(const __half* K, const __half* Q, __half* C, int entries, int T, int w, bool contiguous,
                               int tree) {
  const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kKeys || e >= entries) return;
  const float sum = reduce_partials(K + ((size_t)(e % kHeads) * kKeys + i) * kD, 1, Q + (size_t)e * kD, kD, T, w,
                                    contiguous, tree);
  C[(size_t)e * kKeys + i] = __float2half_rn(kScale * sum);
}

__global__ void output_general(const __half* V, const __half* P, __half* O, int entries, int T, bool contiguous,
                               int tree) {
  const int e = blockIdx.x, d = threadIdx.x;
  if (e >= entries) return;
  const float sum = reduce_partials(V + (size_t)(e % kHeads) * kKeys * kD + d, kD, P + (size_t)e * kKeys, kKeys, T, 1,
                                    contiguous, tree);
  O[(size_t)e * kD + d] = __float2half_rn(sum);
}

// Entry e's pointers as shared_memory_rows.cu makes them: the keys or values of its head, its own query / row.
__global__ void pointers(const __half* KV, const __half* B, size_t b_stride, __half* C, size_t c_stride, int entries,
                         const void** pa, const void** pb, void** pc) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= entries) return;
  pa[e] = KV + (size_t)(e % kHeads) * kKeys * kD;
  pb[e] = B + e * b_stride;
  pc[e] = C + e * c_stride;
}

// The stock layout: every entry its own copy of its head's keys or values.
__global__ void replicate(const __half* KV, __half* R, int entries) {
  const size_t per = (size_t)kKeys * kD;
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < entries * per; x += (size_t)gridDim.x * blockDim.x)
    R[x] = KV[(x / per % kHeads) * per + x % per];
}

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *K, *V, *Q, *P, *S, *S2, *O, *O2, *R; void** ptr; unsigned long long* dc;
  CK(cudaMalloc(&K, 2ull * kHeads * kKeys * kD)); CK(cudaMalloc(&V, 2ull * kHeads * kKeys * kD));
  CK(cudaMalloc(&Q, 2ull * kMaxE * kD)); CK(cudaMalloc(&P, 2ull * kMaxE * kKeys));
  CK(cudaMalloc(&S, 2ull * kMaxE * kKeys)); CK(cudaMalloc(&S2, 2ull * kMaxE * kKeys));
  CK(cudaMalloc(&O, 2ull * kMaxE * kD)); CK(cudaMalloc(&O2, 2ull * kMaxE * kD));
  CK(cudaMalloc(&R, 2ull * kMaxE * kKeys * kD)); CK(cudaMalloc(&ptr, 3 * kMaxE * sizeof(void*)));
  CK(cudaMalloc(&dc, 8));
  const float scale = kScale, one = 1.f, zero = 0.f;
  auto differ = [&](const __half* a, const __half* b, size_t n) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, dc);
    unsigned long long x; CK(cudaMemcpy(&x, dc, 8, cudaMemcpyDeviceToHost)); return x;
  };
  const void** pa = const_cast<const void**>(ptr);
  const void** pb = const_cast<const void**>(ptr + kMaxE);
  void** pc = ptr + 2 * kMaxE;
  const char* sname[] = {"S0 sequential", "S1 sm_75", "S2 mma query row", "S3 mma keys rows", "S4 groups added"};
  const int Ts[] = {16, 32, 64, 128, 256};
  unsigned long long sbad[5][5] = {}, vbad[5][1 + 3 * 5 + 1] = {}, layout[5][2] = {};
  const int gT[] = {2, 4, 8, 16, 32, 64}, gW[] = {1, 2, 4, 8};
  unsigned long long sgen[5][6][5][3] = {}, vgen[5][6][2][3] = {};   // [batch][T][w, or contiguous][tree]
  for (int fill_no = 0; fill_no < 3; ++fill_no) {
    fill<<<1024, 256>>>(K, (size_t)kHeads * kKeys * kD, 7u + fill_no, -7, 1);
    fill<<<1024, 256>>>(V, (size_t)kHeads * kKeys * kD, 11u + fill_no, -6, 2);
    fill<<<1024, 256>>>(Q, (size_t)kMaxE * kD, 3u + fill_no, -6, 1);
    fill<<<1024, 256>>>(P, (size_t)kMaxE * kKeys, 13u + fill_no, -14, -6);
    for (int b = 0; b < 5; ++b) {
      const int E = 20 * (b + 1);
      char name[16];
      // scores through pointers (cuBLAS's view: column-major keys x 1 = K^T q)
      pointers<<<1, 128>>>(K, Q, kD, S, kKeys, E, pa, pb, pc);
      snprintf(name, sizeof name, "s%d", E); nvtxRangePushA(name);
      CK(cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kKeys, 1, kD, &scale, pa, CUDA_R_16F, kD, pb, CUDA_R_16F,
                             kD, &zero, pc, CUDA_R_16F, kKeys, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CK(cudaDeviceSynchronize()); nvtxRangePop();
      replicate<<<1024, 256>>>(K, R, E);
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kKeys, 1, kD, &scale, R, CUDA_R_16F, kD,
                                    (long long)kKeys * kD, Q, CUDA_R_16F, kD, kD, &zero, S2, CUDA_R_16F, kKeys, kKeys,
                                    E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      layout[b][0] += differ(S, S2, (size_t)E * kKeys);
      for (int mode = 0; mode < 5; ++mode) {
        if (mode == 2) scores_mma<<<dim3((kKeys + 7) / 8, E), 32>>>(K, Q, S2, E, 2);
        else if (mode == 3) scores_mma<<<dim3((kKeys + 15) / 16, E), 32>>>(K, Q, S2, E, 3);
        else scores_scalar<<<dim3((kKeys + 127) / 128, E), 128>>>(K, Q, S2, E, mode == 4 ? 4 : mode);
        CK(cudaGetLastError());
        sbad[b][mode] += differ(S, S2, (size_t)E * kKeys);
      }
      // output through pointers (column-major dims x 1 = V p)
      pointers<<<1, 128>>>(V, P, kKeys, O, kD, E, pa, pb, pc);
      snprintf(name, sizeof name, "v%d", E); nvtxRangePushA(name);
      CK(cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, pa, CUDA_R_16F, kD, pb, CUDA_R_16F,
                             kKeys, &zero, pc, CUDA_R_16F, kD, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CK(cudaDeviceSynchronize()); nvtxRangePop();
      replicate<<<1024, 256>>>(V, R, E);
      CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kD, 1, kKeys, &one, R, CUDA_R_16F, kD,
                                    (long long)kKeys * kD, P, CUDA_R_16F, kKeys, kKeys, &zero, O2, CUDA_R_16F, kD, kD,
                                    E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      layout[b][1] += differ(O, O2, (size_t)E * kD);
      for (int ti = 0; ti < 6; ++ti)
        for (int wi = 0; wi < 5; ++wi)
          for (int tr = 0; tr < 3; ++tr) {
            if (wi < 4 && gT[ti] * gW[wi] > kD) continue;
            scores_general<<<dim3((kKeys + 127) / 128, E), 128>>>(K, Q, S2, E, gT[ti], wi < 4 ? gW[wi] : 1, wi == 4,
                                                                    tr);
            CK(cudaGetLastError());
            sgen[b][ti][wi][tr] += differ(S, S2, (size_t)E * kKeys);
          }
      int c = 0;
      output_scalar<<<E, kD>>>(V, P, O2, E, 0, 1); vbad[b][c++] += differ(O, O2, (size_t)E * kD);
      for (int mode = 2; mode <= 4; ++mode)
        for (int T : Ts) {
          output_scalar<<<E, kD>>>(V, P, O2, E, mode, T); CK(cudaGetLastError());
          vbad[b][c++] += differ(O, O2, (size_t)E * kD);
        }
      output_mma<<<dim3(kD / 8, E), 32>>>(V, P, O2, E); CK(cudaGetLastError());
      vbad[b][c++] += differ(O, O2, (size_t)E * kD);
      for (int ti = 0; ti < 6; ++ti)
        for (int ct = 0; ct < 2; ++ct)
          for (int tr = 0; tr < 3; ++tr) {
            output_general<<<E, kD>>>(V, P, O2, E, gT[ti], ct == 1, tr); CK(cudaGetLastError());
            vgen[b][ti][ct][tr] += differ(O, O2, (size_t)E * kD);
          }
    }
  }
  for (int b = 0; b < 5; ++b) {
    const int E = 20 * (b + 1);
    printf("batch %3d: pointers vs strides: scores %llu, output %llu mismatched\n", E, layout[b][0], layout[b][1]);
    for (int mode = 0; mode < 5; ++mode)
      printf("  scores %-18s %8llu of %llu\n", sname[mode], sbad[b][mode], 3ull * E * kKeys);
    int c = 0;
    printf("  output V0 sequential      %8llu of %llu\n", vbad[b][c++], 3ull * E * kD);
    for (int mode = 2; mode <= 4; ++mode)
      for (int T : Ts)
        printf("  output V%d T=%-3d           %8llu\n", mode, T, vbad[b][c++]);
    printf("  output V1 mma chain       %8llu\n", vbad[b][c++]);
    for (int ti = 0; ti < 6; ++ti)
      for (int wi = 0; wi < 5; ++wi)
        for (int tr = 0; tr < 3; ++tr)
          if ((wi == 4 || gT[ti] * gW[wi] <= kD) && sgen[b][ti][wi][tr] == 0)
            printf("  MATCH scores T=%d %s%d %s\n", gT[ti], wi == 4 ? "contiguous" : "strided w=", wi == 4 ? 0 : gW[wi],
                   tr == 2 ? "tree from neighbours" : tr ? "tree" : "in order");
    for (int ti = 0; ti < 6; ++ti)
      for (int ct = 0; ct < 2; ++ct)
        for (int tr = 0; tr < 3; ++tr)
          if (vgen[b][ti][ct][tr] == 0)
            printf("  MATCH output T=%d %s %s\n", gT[ti], ct ? "contiguous" : "strided",
                   tr == 2 ? "tree from neighbours" : tr ? "tree" : "in order");
  }
  return 0;
}
