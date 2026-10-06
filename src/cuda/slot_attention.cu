#include "cuda/slot_attention.h"

#include <cuda_fp16.h>

#include "cuda/slot_attention.cuh"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    bool slot_attention_enabled() {
      static const bool on = read_bool_from_env("CT2_SLOT_ATTENTION") && cublas_verified_on(8, 9);
      return on;
    }

    // The recipes were recovered for one window's 5 beams x 20 heads of 64 dims, t up to the slots' 448.
    bool slot_attention_applies(int rows, int heads, int depth, int time) {
      return rows == sa_rows && heads == sa_heads && depth == sa_depth && time >= 1 && time <= sa_capacity;
    }

    int slot_scores_recipe(int time) {
      return selfattn_scores_recipe_of_t[time];
    }

    int slot_output_recipe(int time) {
      return selfattn_output_recipe_of_t[time];
    }

    bool slot_scores_mma(int time) {
      return selfattn_scores_recipes[selfattn_scores_recipe_of_t[time]].kind == 1;
    }

    bool slot_output_mma(int time) {
      return selfattn_output_recipes[selfattn_output_recipe_of_t[time]].kind == 1;
    }

    void slot_attention_scores(const SlotAttention* parts, int count, const void* queries, int heads, int max_time,
                               float scale) {
      sa_scores_launch(parts, count, static_cast<const __half*>(queries), heads, max_time, scale, get_cuda_stream());
    }

    void slot_attention_output(const SlotAttention* parts, int count, void* out, int heads) {
      sa_output_launch(parts, count, static_cast<__half*>(out), heads, get_cuda_stream());
    }

  }
}
