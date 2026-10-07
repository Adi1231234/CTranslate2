#pragma once

// slot_attention.h's kernels and their launches (headers of their own so that tools/turing/kernels/
// slot_attention_check.cu runs these very kernels against cuBLAS).

#include "cuda/slot_attention_scores.cuh"
#include "cuda/slot_attention_output.cuh"
