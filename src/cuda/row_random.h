#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "ctranslate2/types.h"
#include "cuda/random.h"

namespace ctranslate2 {
  namespace cuda {

    // Sampling where every row draws from a random stream of its own, seeded by its input (DecodingOptions::
    // sampling_seeds): a row's draws then depend on neither the other rows of the call, nor its place in the batch,
    // nor what its thread drew before, so an input sampled in a batch with others draws what it draws alone, and a
    // run repeats. While a RowRandomScope is active on a thread, the multinomial kernel takes row i's random number
    // from states[state_of_row[i]] (the shared per-thread states otherwise, cuda/random.h).
    struct RowRandom {
      curandStatePhilox4_32_10_t* states = nullptr;   // one per row of the search, on the device
      const int32_t* state_of_row = nullptr;          // on the device: the state of each row of the call
      dim_t rows = 0;                                 // rows of the call
    };

    const RowRandom* row_random();

    class RowRandomScope {
    public:
      explicit RowRandomScope(const RowRandom& random);
      ~RowRandomScope();
      RowRandomScope(const RowRandomScope&) = delete;
      RowRandomScope& operator=(const RowRandomScope&) = delete;
    private:
      const RowRandom* _previous;
    };

    // Philox states on the device, the i-th from seeds[i] = (seed, subsequence).
    class RowStates {
    public:
      explicit RowStates(const std::vector<std::pair<uint64_t, uint64_t>>& seeds);
      ~RowStates();
      RowStates(const RowStates&) = delete;
      RowStates& operator=(const RowStates&) = delete;
      curandStatePhilox4_32_10_t* states() {
        return _states;
      }
    private:
      curandStatePhilox4_32_10_t* _states = nullptr;
    };

  }
}
