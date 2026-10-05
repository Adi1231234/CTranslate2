#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Sampled hypotheses that share their clip's encoder memory (GreedySearch with num_hypotheses > 1): CTranslate2
    // repeats the memory keys and values once per hypothesis, and each hypothesis attends to its copy as a batch entry
    // of its own (one query a row). While a SharedMemoryRowsScope is active on a thread, the decoder state keeps one
    // copy a clip and the cross-attention reads it through pointers (cublasGemmBatchedEx): the same products on the
    // same values, so the same bits (tools/turing/kernels/ptrbatch_probe.cu), without the copies nor their reads.
    struct SharedMemoryRows {
      const int32_t* row_clip = nullptr;   // on the device: each row's clip in the memory (rows entries)
      dim_t rows = 0;
      dim_t clips = 0;
    };

    // The calling thread's mapping, or nullptr outside a scope.
    const SharedMemoryRows* shared_memory_rows();

    // True where sharing was verified (ptrbatch_probe: sm_89 with cuBLAS 12.9.2) and not disabled
    // (CT2_SHARED_MEMORY_ROWS=0).
    bool shared_memory_rows_enabled();

    class SharedMemoryRowsScope {
    public:
      explicit SharedMemoryRowsScope(const SharedMemoryRows& rows);
      ~SharedMemoryRowsScope();
      SharedMemoryRowsScope(const SharedMemoryRowsScope&) = delete;
      SharedMemoryRowsScope& operator=(const SharedMemoryRowsScope&) = delete;
    private:
      const SharedMemoryRows* _previous;
    };

    // scores[r][h] (1 x keys) = alpha q[r][h] . k[clip(r)][h]^T, as MatMul(queries, keys, trans_b, alpha) computes
    // it for the repeated keys; q [rows, heads, 1, depth], k [clips, heads, keys, depth].
    void shared_rows_scores(const float16_t* q, const float16_t* k, float16_t* scores, dim_t heads, dim_t keys,
                            dim_t depth, float alpha);
    // out[r][h] (1 x depth) = p[r][h] . v[clip(r)][h], as MatMul(attention, values) for the repeated values.
    void shared_rows_output(const float16_t* p, const float16_t* v, float16_t* out, dim_t heads, dim_t keys,
                            dim_t depth);

  }
}
