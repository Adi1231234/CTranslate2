#pragma once

#include <cstdint>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Self-attention caches in slots that stay in place (layers/slot_cache.h): the kernels, pure data movement. The
    // descriptors are arrays on the device, one entry a part (packed by the caller as int32 words).

    // A part's slot maps and its step's plan.
    struct SlotPlan {
      const int32_t* order;        // this step's beams' order (row r continues row order[r]), or null: unchanged
      int32_t* slot_of_row;        // in: the previous step's; out: this step's
      int32_t* row_of_slot;        // out
      int32_t* fork_src;           // out: this step's copies, slot fork_src[i] into slot fork_dst[i]
      int32_t* fork_dst;
      int32_t* fork_count;         // out: [1]
      int32_t rows;                // this step's rows, the part's slots
      int32_t old_rows;            // the previous step's rows (fewer at the beams' expansion)
    };
    // A thread a part: row r keeps its parent's slot when it is the parent's first child, else takes a slot no row
    // keeps, which gets a copy of the parent's.
    void slot_plan(const SlotPlan* plans, int parts);

    // One layer of a part.
    struct SlotAppend {
      void* keys;                  // [rows, heads, capacity, depth] fp16
      void* values;
      const int32_t* slot_of_row;
      const int32_t* fork_src;
      const int32_t* fork_dst;
      const int32_t* fork_count;
      int32_t rows;
      int32_t time;                // positions cached before this step's
      int32_t row_begin;           // the part's first row in the step's keys and values
      int32_t shared;              // positions [0, shared) alike in every slot (the prompt, layers/slot_cache.h)
    };
    // Every part: the forks' positions [shared, time) copied (the prompt's are in every slot already), then the
    // step's keys and values ([rows, heads, 1, depth] fp16 from row_begin) written at position `time` of their rows'
    // slots.
    void slot_append(const SlotAppend* parts, int count, const void* fresh_keys, const void* fresh_values,
                     int heads, int depth, int capacity, int max_rows, int max_time);

    // Rows of `row_bytes` (a multiple of 16) of every part: dst[row_begin + i] = src[row_begin + map[i]].
    struct SlotPermute {
      const int32_t* map;
      int32_t rows;
      int32_t row_begin;
    };
    void slot_permute(const SlotPermute* parts, int count, const void* src, void* dst, int row_bytes, int max_rows);

  }
}
