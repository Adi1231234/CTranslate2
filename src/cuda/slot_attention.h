#pragma once

#include <cstdint>

namespace ctranslate2 {
  namespace cuda {

    // CT2_SLOT_ATTENTION=1: the self-attention products of every slot part of a joint decoding step (layers/slot_cache.h)
    // in one launch per product and layer, instead of a cuBLAS call per part and product (long13: ~28% of the GPU's
    // time on long recordings, small calls at half the bandwidth). Every value has the arithmetic cuBLAS gives the
    // part's own call at its length t (selfattn_recipes.h, recovered by tools/turing/kernels/selfattn_recipe_probe.cu
    // on the L40S with cuBLAS 12.9.2), and positions every slot holds alike (the prompt, copied into each at the
    // part's first slot step) are read from slot 0 for all the beams: the same values, a fifth of the reads.
    struct SlotAttention {
      const void* keys;            // the part's slots this layer, [rows, heads, capacity, depth] fp16, slot order
      const void* values;
      void* scores;                // [rows, heads, time] fp16, slot order: the scores, then the probabilities
      int32_t rows, time, shared, row_begin;   // time: positions attended; [0, shared) alike in every slot
      int32_t scores_recipe, output_recipe;    // selfattn_recipes.h's codes for this time
      int32_t scores_mma, output_mma;          // whether they are mma chains (their own kernels)
    };

    // On this device and cuBLAS build, with CT2_SLOT_ATTENTION=1.
    bool slot_attention_enabled();
    // A part of `rows` beams, `heads` heads of `depth` dims, attending `time` positions has recovered recipes.
    bool slot_attention_applies(int rows, int heads, int depth, int time);
    int slot_scores_recipe(int time);
    int slot_output_recipe(int time);
    bool slot_scores_mma(int time);
    bool slot_output_mma(int time);

    // scores = scale q k^T for every part (queries [rows, heads, depth] in slot order from row_begin).
    void slot_attention_scores(const SlotAttention* parts, int count, const void* queries, int heads, int max_time,
                               float scale);
    // out = p v for every part, into out's rows (slot order) from row_begin.
    void slot_attention_output(const SlotAttention* parts, int count, void* out, int heads);

  }
}
