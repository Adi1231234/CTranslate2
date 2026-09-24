// Bit-for-bit check of src/cuda/timestamp_rules.cuh (two segmented reductions, one read-back) against the
// per-row thrust::reduce calls of primitives<CUDA>::max and ::logsumexp that should_sample_timestamp
// makes, on fp16 rows shaped like Whisper log-probs (51866 tokens, timestamps 50365..51865).
// usage: ts_check      -> mismatching text maxima, timestamp maxima, exp sums and decisions; must be 0
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "probe_common.h"
#include "probe_data.cuh"
#include "cuda/timestamp_rules.cuh"

using namespace ctranslate2::cuda;

// primitives.cu's functor: the max comes by value from the host.
template <typename T>
struct exp_minus_max_func {
  const float _max_value;
  exp_minus_max_func(const float max_value) : _max_value(max_value) {}
  __device__ float operator()(T x) { return expf(float(x) - _max_value); }
};

int main() {
  const int vocab = 51866, begin = 50365, end = 51865, rows = 3000, nts = end - begin + 1;
  const __half lowest = __float2half(-65504.f);                     // numeric_limits<half>::lowest()
  __half* lp;
  CK(cudaMalloc(&lp, sizeof(__half) * vocab * rows));
  for (int part = 0; part < 3; ++part)                              // three value ranges, 1000 rows each
    fill<<<1024, 256>>>(lp + (size_t)part * 1000 * vocab, (size_t)1000 * vocab, 777u + part,
                        -14 + 4 * part, 2 + 2 * part);
  // Old path: one thrust::reduce per value per row, as primitives<CUDA> does.
  std::vector<__half> old_text(rows), old_ts(rows);
  std::vector<float> old_sum(rows);
  for (int r = 0; r < rows; ++r) {
    const __half* row = lp + (size_t)r * vocab;
    auto pol = thrust::cuda::par_nosync.on(0);
    old_text[r] = thrust::reduce(pol, row, row + begin, lowest, maximum<__half>());
    old_ts[r] = thrust::reduce(pol, row + begin, row + begin + nts, lowest, maximum<__half>());
    auto it = thrust::make_transform_iterator(thrust::device_pointer_cast(row + begin),
                                              exp_minus_max_func<__half>(__half2float(old_ts[r])));
    old_sum[r] = thrust::reduce(pol, it, it + nts);
  }
  // New path.
  std::vector<int32_t> ids(rows);
  for (int r = 0; r < rows; ++r) ids[r] = (r * 7) % rows;             // rows in a shuffled order
  const size_t temp_bytes = timestamp_mass_temp_bytes<__half>(rows, begin, end, lowest, 0);
  __half* maxima;
  float* ts_sum;
  void* temp;
  CK(cudaMalloc(&maxima, sizeof(__half) * 2 * rows));
  CK(cudaMalloc(&ts_sum, sizeof(float) * rows));
  CK(cudaMalloc(&temp, temp_bytes));
  timestamp_mass_enqueue<__half>(lp, vocab, ids, begin, end, lowest, maxima, ts_sum, temp, temp_bytes, 0);
  std::vector<__half> new_text(rows), new_ts(rows);
  std::vector<float> new_sum(rows);
  CK(cudaMemcpy(new_text.data(), maxima, sizeof(__half) * rows, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(new_ts.data(), maxima + rows, sizeof(__half) * rows, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(new_sum.data(), ts_sum, sizeof(float) * rows, cudaMemcpyDeviceToHost));
  int bad_text = 0, bad_ts = 0, bad_sum = 0, bad_decision = 0, timestamps = 0;
  for (int i = 0; i < rows; ++i) {
    const int r = (int)ids[i];
    bad_text += half_bits(new_text[i]) != half_bits(old_text[r]);
    bad_ts += half_bits(new_ts[i]) != half_bits(old_ts[r]);
    uint32_t a, b;
    memcpy(&a, &new_sum[i], 4); memcpy(&b, &old_sum[r], 4);
    bad_sum += a != b;
    const bool old_d = std::log(old_sum[r]) + __half2float(old_ts[r]) > __half2float(old_text[r]);
    const bool new_d = std::log(new_sum[i]) + __half2float(new_ts[i]) > __half2float(new_text[i]);
    bad_decision += old_d != new_d;
    timestamps += new_d;
  }
  printf("rows %d: text max mismatches %d, timestamp max %d, exp sums %d, decisions %d "
         "(timestamp chosen in %d rows)\n", rows, bad_text, bad_ts, bad_sum, bad_decision, timestamps);
  printf("TOTAL %d mismatches\n", bad_text + bad_ts + bad_sum + bad_decision);
  return 0;
}
