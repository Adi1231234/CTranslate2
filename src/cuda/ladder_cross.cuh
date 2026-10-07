#pragma once

// The kernels of ladder_cross.h (a long recording's sampled ladder rows against their clip's memory), in headers
// so that tools/turing/kernels/ladder_cross_check.cu runs these very kernels against cuBLAS.

#include "cuda/ladder_cross_scores.cuh"
#include "cuda/ladder_cross_output.cuh"
