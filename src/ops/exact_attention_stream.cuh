#pragma once

// exact_attention.cuh's encoder self-attention (1500 queries and keys of 64 dims) with the same bits, without
// keeping a score row in shared memory: a block of WARPS warps takes WARPS x RT x 16 queries of one batch entry, and
// the keys and values stream through shared memory once per pass, each staged chunk serving every query of the
// block (exact_attention.cuh reads the entry's keys and values once per 16 queries). The scores are recomputed in
// each of three passes, every time the same chain (one mma.sync m16n8k16 over the 64 dims from zero, then
// s = half(alpha * acc)):
//   1. the row maximum (order-free);
//   2. the row sum in rows1024_row's order: legacy thread i of lane L adds keys 32L + i and 1024 + 32L + i, each
//      lane adds its 32 threads in order, then the 32 lanes in order; a step stages both chunks of a lane L;
//   3. p = half(exp(x - max) / sum) (rows1024_quotient where it is exact, the division elsewhere: the same value),
//      and the output, one m16n8k16 chain per output over the keys in cuBLAS's 16-key groups: [0, 16), [16, 28)
//      with zero values to 32, then from 28 on; a step stages 32 keys and their values, two groups.
// Rows past 1500 compute on zero queries and are not stored. q, k and v are head-split [batch, 1500, 64]; o is
// [clip, query, head, dim] as exact_attention's.

#include <string>

#include "exact_attention_parts.cuh"
#include "cuda/split_gemm_common.cuh"

namespace at {
  namespace native {

    constexpr int eas_n = 1500, eas_residue = eas_n % 64, eas_pitch = ea_depth + 8, eas_rows = 64;
    constexpr int eas_pass1 = (eas_n + 63) / 64, eas_pass2 = 32, eas_pass3 = 1 + (eas_n - eas_residue) / 32;
    constexpr int eas_steps = eas_pass1 + eas_pass2 + eas_pass3;

    template <int WARPS, int RT>
    constexpr int eas_queries = WARPS * RT * ea_rows;

    template <int WARPS, int RT, int STAGES>
    constexpr int eas_smem_bytes = STAGES * eas_rows * eas_pitch * 2 + WARPS * RT * ea_rows * 33 * 4;

    // Stage rows of step s: key rows (or, in pass 3, rows 32..63, the values) into rows 0..63, zeros where a row
    // is past the keys (or a value past the 28 of the first output chunk).
    template <int THREADS>
    __device__ __forceinline__ void eas_load(__half* dst, const __half* kb, const __half* vb, int s) {
      for (int p = threadIdx.x; p < eas_rows * 8; p += THREADS) {
        const int row = p / 8, piece = p % 8;
        int key;
        bool valid, values = false;
        if (s < eas_pass1) {
          key = 64 * s + row;
          valid = key < eas_n;
        } else if (s < eas_pass1 + eas_pass2) {
          const int L = s - eas_pass1;
          key = row < 32 ? 32 * L + row : 1024 + 32 * L + row - 32;
          valid = key < eas_n;
        } else {
          const int c = s - eas_pass1 - eas_pass2;
          key = (c == 0 ? 0 : eas_residue + 32 * (c - 1)) + row % 32;
          values = row >= 32;
          valid = key < eas_n && !(values && c == 0 && key >= eas_residue);
        }
        ctranslate2::cuda::gsg_cp16(dst + row * eas_pitch + piece * 8,
                                    (values ? vb : kb) + (size_t)(valid ? key : 0) * ea_depth + piece * 8, valid);
      }
    }

    // The scores of 8 keys (stage rows r0..r0 + 7) for the warp's 16-query tile `a`: x[0..1] row g, x[2..3] row g + 8,
    // keys 2t, 2t + 1, as the softmax reads them (float of the half).
    __device__ __forceinline__ void eas_scores(const unsigned (&a)[4][4], const unsigned (&b)[8], float alpha,
                                               float* x) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int c = 0; c < 4; ++c)                               // 16-dim groups in increasing order
        ea_mma(d, a[c][0], a[c][1], a[c][2], a[c][3], make_uint2(b[2 * c], b[2 * c + 1]));
      const __half2 lo = __floats2half2_rn(alpha * d[0], alpha * d[1]);
      const __half2 hi = __floats2half2_rn(alpha * d[2], alpha * d[3]);
      x[0] = __low2float(lo); x[1] = __high2float(lo); x[2] = __low2float(hi); x[3] = __high2float(hi);
    }

    // The B fragments of the 8 keys at stage rows r0..r0 + 7, all 64 dims (b[2c], b[2c + 1]: dims 16c..16c + 15).
    __device__ __forceinline__ void eas_key_fragments(const __half* buf, int r0, int lane, unsigned (&b)[8]) {
      const __half* row = buf + (r0 + lane % 8) * eas_pitch + 8 * (lane / 8);
      ctranslate2::cuda::gsg_ldm4(b, row);
      ctranslate2::cuda::gsg_ldm4(b + 4, row + 32);
    }

    __device__ __forceinline__ void eas_ldm4_trans(unsigned* r, const __half* p) {
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }

    __device__ __forceinline__ unsigned eas_pack(float a, float b) {
      const __half2 h = __halves2half2(static_cast<__half>(a), static_cast<__half>(b));
      return *reinterpret_cast<const unsigned*>(&h);
    }

    // rows1024_store's value of e / sum: the quotient where rows1024_quotient is exact, else the division.
    __device__ __forceinline__ float eas_probability(float e, float sum, float y) {
      return sum >= 1.f && sum <= 2048.f && e >= 0x1p-60f ? rows1024_quotient(e, sum, y) : e / sum;
    }

    template <int WARPS, int RT, int STAGES>
    __global__ void __launch_bounds__(WARPS * C10_WARP_SIZE)
    exact_attention_stream_kernel(const __half* q, const __half* k, const __half* v, __half* o, int heads,
                                  float alpha) {
#if __CUDA_ARCH__ >= 800
      constexpr int THREADS = WARPS * C10_WARP_SIZE, stage_halves = eas_rows * eas_pitch;
      extern __shared__ __align__(16) unsigned char eas_smem[];
      __half* stages = reinterpret_cast<__half*>(eas_smem);
      const int warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE, g = lane / 4, t = lane % 4;
      float* tbuf = reinterpret_cast<float*>(stages + STAGES * stage_halves) + warp * RT * ea_rows * 33;
      const int entry = blockIdx.y, row0 = blockIdx.x * eas_queries<WARPS, RT> + warp * RT * ea_rows;
      const __half* kb = k + (size_t)entry * eas_n * ea_depth;
      const __half* vb = v + (size_t)entry * eas_n * ea_depth;
      const __half* qb = q + (size_t)entry * eas_n * ea_depth;

      #pragma unroll
      for (int s = 0; s < STAGES - 1; ++s) {
        eas_load<THREADS>(stages + s * stage_halves, kb, vb, s);
        asm volatile("cp.async.commit_group;");
      }
      int step = 0;
      auto next = [&]() {                                       // wait for `step`, start step + STAGES - 1
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        __syncthreads();
        if (step + STAGES - 1 < eas_steps)
          eas_load<THREADS>(stages + ((step + STAGES - 1) % STAGES) * stage_halves, kb, vb, step + STAGES - 1);
        asm volatile("cp.async.commit_group;");
        return stages + (step++ % STAGES) * stage_halves;
      };

      unsigned a[RT][4][4];                                     // the warp's RT x 16 queries, 4 groups of 16 dims
      #pragma unroll
      for (int r = 0; r < RT; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c)
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = row0 + 16 * r + g + 8 * (e % 2);
            a[r][c][e] = j < eas_n ? *reinterpret_cast<const unsigned*>(qb + (size_t)j * ea_depth + 16 * c
                                                                        + 8 * (e / 2) + 2 * t) : 0u;
          }

      // 1. The row maxima: rows g and g + 8 of each tile r, in every lane of the quad.
      float mx[RT][2];
      #pragma unroll
      for (int r = 0; r < RT; ++r)
        mx[r][0] = mx[r][1] = -max_float;
      for (int s = 0; s < eas_pass1; ++s) {
        const __half* buf = next();
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int key = 64 * s + 8 * j + 2 * t;
          if (64 * s + 8 * j >= eas_n)
            break;
          unsigned b[8];
          eas_key_fragments(buf, 8 * j, lane, b);
          #pragma unroll
          for (int r = 0; r < RT; ++r) {
            float x[4];
            eas_scores(a[r], b, alpha, x);
            #pragma unroll
            for (int e = 0; e < 4; ++e)
              if (key + e % 2 < eas_n)
                mx[r][e / 2] = MaxFloat<float, float>()(mx[r][e / 2], x[e]);
          }
        }
      }
      #pragma unroll
      for (int r = 0; r < RT; ++r)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
          #pragma unroll
          for (int offset = 1; offset < 4; offset *= 2)
            mx[r][h] = Max<float>()(mx[r][h], __shfl_xor_sync(0xffffffff, mx[r][h], offset));

      // 2. The row sums: lane 16 (r % 2) + row adds row `row` of tile r (RT <= 2).
      static_assert(RT <= 2, "two query tiles a warp at most");
      float sum = 0.f;
      const int sum_tile = lane / 16, sum_row = lane % 16;
      for (int L = 0; L < eas_pass2; ++L) {
        const __half* buf = next();
        #pragma unroll
        for (int r = 0; r < RT; ++r) {
          float T[4][4];                                        // [8-key tile][x index] of keys 32L + 8j + 2t (+1)
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            unsigned b[8];
            eas_key_fragments(buf, 8 * j, lane, b);
            float x[4];
            eas_scores(a[r], b, alpha, x);
            #pragma unroll
            for (int e = 0; e < 4; ++e)
              T[j][e] = 0.f + std::exp(x[e] - mx[r][e / 2]);
          }
          if (1024 + 32 * L < eas_n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
              unsigned b[8];
              eas_key_fragments(buf, 32 + 8 * j, lane, b);
              float x[4];
              eas_scores(a[r], b, alpha, x);
              #pragma unroll
              for (int e = 0; e < 4; ++e)
                if (1024 + 32 * L + 8 * j + 2 * t + e % 2 < eas_n)
                  T[j][e] = T[j][e] + std::exp(x[e] - mx[r][e / 2]);
            }
          }
          float* tb = tbuf + r * ea_rows * 33;
          #pragma unroll
          for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
              tb[(g + 8 * (e / 2)) * 33 + 8 * j + 2 * t + e % 2] = T[j][e];
        }
        __syncwarp();
        if (sum_tile < RT) {
          const float* tb = tbuf + (sum_tile * ea_rows + sum_row) * 33;
          float lane_sum = 0.f;                                 // legacy threads 32L .. 32L + 31 in order
          #pragma unroll
          for (int i = 0; i < 32; ++i)
            lane_sum = Add<float>()(lane_sum, tb[i]);
          sum = Add<float>()(sum, lane_sum);                    // the lanes in order
        }
        __syncwarp();
      }

      // 3. The probabilities and the output.
      float rs[RT][2], ry[RT][2];
      #pragma unroll
      for (int r = 0; r < RT; ++r)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          rs[r][h] = __shfl_sync(0xffffffff, sum, 16 * r + g + 8 * h);
          ry[r][h] = __frcp_rn(rs[r][h]);
        }
      float acc[RT][8][4] = {};
      for (int c = 0; c < eas_pass3; ++c) {
        const __half* buf = next();
        #pragma unroll
        for (int G = 0; G < 2; ++G) {                           // the chunk's 16-key groups in order
          unsigned bv[4][4];                                    // dims 16u..16u + 15: b0 b1 of tile 2u, then 2u + 1
          #pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int m = lane / 8;
            eas_ldm4_trans(bv[u], buf + (32 + 16 * G + lane % 8 + 8 * (m % 2)) * eas_pitch + 16 * u + 8 * (m / 2));
          }
          unsigned b0[8], b1[8];
          eas_key_fragments(buf, 16 * G, lane, b0);
          eas_key_fragments(buf, 16 * G + 8, lane, b1);
          #pragma unroll
          for (int r = 0; r < RT; ++r) {
            float x0[4], x1[4], p0[4], p1[4];
            eas_scores(a[r], b0, alpha, x0);
            eas_scores(a[r], b1, alpha, x1);
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
              p0[e] = eas_probability(std::exp(x0[e] - mx[r][e / 2]), rs[r][e / 2], ry[r][e / 2]);
              p1[e] = eas_probability(std::exp(x1[e] - mx[r][e / 2]), rs[r][e / 2], ry[r][e / 2]);
            }
            const unsigned pa0 = eas_pack(p0[0], p0[1]), pa1 = eas_pack(p0[2], p0[3]);
            const unsigned pa2 = eas_pack(p1[0], p1[1]), pa3 = eas_pack(p1[2], p1[3]);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
              ea_mma(acc[r][2 * u], pa0, pa1, pa2, pa3, make_uint2(bv[u][0], bv[u][1]));
              ea_mma(acc[r][2 * u + 1], pa0, pa1, pa2, pa3, make_uint2(bv[u][2], bv[u][3]));
            }
          }
        }
      }

      const int clip = entry / heads, head = entry % heads;     // o is [clip, query, head, dim]
      #pragma unroll
      for (int r = 0; r < RT; ++r)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = row0 + 16 * r + g + 8 * h;
          if (j < eas_n)
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt)
              *reinterpret_cast<__half2*>(o + (((size_t)clip * eas_n + j) * heads + head) * ea_depth + 8 * dt + 2 * t)
                = __floats2half2_rn(acc[r][dt][2 * h], acc[r][dt][2 * h + 1]);
        }
#endif
    }

    // q, k and v head-split, each from its source (with its bias: exact_attention_layout's add), into dst: three
    // [batch, n, 64] tensors one after the other.
    static __global__ void exact_attention_stream_split(EalSource qs, EalSource ks, EalSource vs, __half* dst,
                                                        int heads, int batch, int n) {
      const size_t count = (size_t)batch * n * (ea_depth / 8);
      for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < 3 * count; i += (size_t)gridDim.x * blockDim.x) {
        const int part = int(i / count);
        eal_query_vector(part == 0 ? qs : part == 1 ? ks : vs, dst + part * count * 8, heads, n, ea_depth, i % count);
      }
    }

    template <int WARPS, int RT, int STAGES>
    inline void exact_attention_stream_items(const __half* q, const __half* k, const __half* v, __half* o, int batch,
                                             int heads, float alpha, cudaStream_t stream) {
      constexpr int smem = eas_smem_bytes<WARPS, RT, STAGES>;
      static const bool configured = cudaFuncSetAttribute(exact_attention_stream_kernel<WARPS, RT, STAGES>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      constexpr int queries = eas_queries<WARPS, RT>;
      exact_attention_stream_kernel<WARPS, RT, STAGES><<<dim3((eas_n + queries - 1) / queries, batch),
                                                         WARPS * C10_WARP_SIZE, smem, stream>>>(q, k, v, o, heads,
                                                                                                alpha);
    }

    // The attention of head-split q, k, v [batch, 1500, 64] in one block shape.
    using EasItems = void (*)(const __half*, const __half*, const __half*, __half*, int, int, float, cudaStream_t);

    // A block shape by name, <warps>x<tiles>s<stages> (null for any other name).
    inline EasItems exact_attention_stream_shape(const std::string& name) {
      if (name == "8x1s3") return exact_attention_stream_items<8, 1, 3>;
      if (name == "8x1s2") return exact_attention_stream_items<8, 1, 2>;
      if (name == "4x2s3") return exact_attention_stream_items<4, 2, 3>;
      if (name == "4x1s4") return exact_attention_stream_items<4, 1, 4>;
      if (name == "8x2s2") return exact_attention_stream_items<8, 2, 2>;
      return nullptr;
    }

    // From the fused projection x [clips, n, 3 * heads * 64] without its bias, and that bias (or null): q, k and v
    // head-split with their bias into the workspace (3 x clips x heads x n x 64 halves, which
    // exact_attention_workspace(clips * heads, n, true) holds), then `items` on them.
    inline void exact_attention_stream_qkv(EasItems items, const __half* x, const __half* bias, void* workspace,
                                           __half* o, int clips, int heads, int n, float alpha, cudaStream_t stream) {
      __half* split = static_cast<__half*>(workspace);
      const size_t count = (size_t)clips * heads * n * ea_depth;
      exact_attention_stream_split<<<1024, 256, 0, stream>>>(eal_qkv_part(x, bias, heads, n, ea_depth, 0),
                                                             eal_qkv_part(x, bias, heads, n, ea_depth, 1),
                                                             eal_qkv_part(x, bias, heads, n, ea_depth, 2), split,
                                                             heads, clips * heads, n);
      items(split, split + count, split + 2 * count, o, clips * heads, heads, alpha, stream);
    }

  }
}
