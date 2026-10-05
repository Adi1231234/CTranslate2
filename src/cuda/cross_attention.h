#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Whisper decoder cross-attention in one kernel (ops/cross_attention.cuh), bit for bit the MatMul, SoftMax and
    // MatMul it replaces where their cuBLAS kernels were matched (tools/turing/kernels/cross_sweep.cu): sm_120 with
    // cuBLAS 12.9.2, fp16, 64-dim heads, 1500 encoder positions, 2..8 queries, batch (clips x heads) 20..160 in
    // steps of 20. Returns the residue of the key tile cuBLAS uses for the output product, or -1 where the kernel
    // does not apply. CT2_CROSS_ATTN=0 keeps the three ops.
    int cross_attention_residue(dim_t m, dim_t batch, dim_t keys, dim_t depth);

    // Several batches' clips in one launch (cuda/clip_groups.h), each group with its own batch's residue: clips
    // [clip_end[g - 1], clip_end[g]) use residue[g]. count 0: every clip uses the launch's residue.
    struct CrossResidues {
      static constexpr int max_groups = 16;
      int count = 0;
      int clip_end[max_groups] = {};
      int residue[max_groups] = {};
    };

    // Whether the kernel can also project the queries (cross_attention_q.cuh): rows = clips x m of K = heads x 64
    // inputs, where cuBLAS runs that Dense layer as one chain (2..48 rows, 1280 x 1280). CT2_CROSS_Q=0 keeps it.
    bool cross_attention_projects(dim_t rows, dim_t n, dim_t k);

    // q: [clips][heads][m][64] (or null with x: [clips * m][k], w: [heads * 64][k], bias: [heads * 64], the
    // queries' Dense layer); k, v: [clips][heads][1500][64] (with slot, on the device: clip c's are the cache's
    // entry slot[c], cuda/memory_slots.h; with kv, on the device, k and v unused: clip c's start at kv[2c] and
    // kv[2c + 1], the clips of several searches, layers/joint_step.h); o: [clips][m][heads][64]. With dense (q and
    // x null): the queries' Dense output [clips * m][heads * 64] without its bias, which is bias (or null).
    void cross_attention(const float16_t* q, const float16_t* k, const float16_t* v, float16_t* o,
                         dim_t clips, dim_t heads, dim_t m, float alpha, int residue,
                         const float16_t* x = nullptr, const float16_t* w = nullptr,
                         const float16_t* bias = nullptr, dim_t k_inputs = 0, const int32_t* slot = nullptr,
                         const CrossResidues& residues = CrossResidues(), const float16_t* const* kv = nullptr,
                         const float16_t* dense = nullptr);

  }
}
