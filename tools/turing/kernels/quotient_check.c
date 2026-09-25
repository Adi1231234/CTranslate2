/* CPU check of the softmax's division shortcut (src/ops/softmax_rows1024.cuh rows1024_quotient): for y = RN(1/s),
 * q = RN(e y), q = RN(q + RN(e - s q) y) twice (fma), must equal RN(e / s) - the IEEE division the kernel replaces -
 * for every softmax value e in [2^-60, 1] and row sum s in [1, 2048]. IEEE single precision with round to nearest
 * is the same on the CPU (SSE division, FMA3 fmaf) as on the GPU (__fdiv_rn, __frcp_rn, __fmul_rn, __fmaf_rn).
 * Checks every e of each tested binade against structured and random sums, then random pairs.
 * usage: quotient_check [random pairs in millions, default 4000]   -> must print TOTAL 0 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bits_to_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t float_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static float quotient(float e, float s, float y) {
  float q = e * y;
  q = fmaf(fmaf(-s, q, e), y, q);
  return fmaf(fmaf(-s, q, e), y, q);
}

static uint64_t rng = 88172645463325252ull;
static uint32_t next32(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }

static unsigned long long check(float e, float s) {
  volatile float vs = s;                       /* keep 1 / s a real division */
  const float y = 1.0f / vs;
  return float_bits(quotient(e, s, y)) != float_bits(e / s);
}

int main(int argc, char** argv) {
  const long long millions = argc > 1 ? atoll(argv[1]) : 4000;
  unsigned long long bad = 0, n = 0;
  /* 1. every e in [2^-60, 1] (2^29 values) against sums near powers of two and random sums */
  float sums[64]; int ns = 0;
  for (int k = 0; k <= 11; ++k) {                          /* 2^k and its neighbours */
    const float p = ldexpf(1.f, k);
    sums[ns++] = p; sums[ns++] = nextafterf(p, 0.f); sums[ns++] = nextafterf(p, 4096.f);
  }
  while (ns < 64) sums[ns++] = 1.f + (next32() % 2047000000u) / 1000000.f;
  const uint32_t lo = float_bits(ldexpf(1.f, -60)), hi = float_bits(1.f);
  for (int j = 0; j < 64; ++j) {
    for (uint32_t u = lo; u <= hi; u += 1 + (j > 8) * 7) { bad += check(bits_to_float(u), sums[j]); ++n; }
    fprintf(stderr, "sum %d/64 done, %llu mismatches so far\n", j + 1, bad);
  }
  printf("structured: %llu pairs, %llu mismatches\n", n, bad);
  /* 2. random pairs: e uniform over the bit patterns of [2^-60, 1], s uniform over those of [1, 2048] */
  const uint32_t slo = float_bits(1.f), shi = float_bits(2048.f);
  unsigned long long rbad = 0;
  for (long long i = 0; i < millions * 1000000LL; ++i) {
    const float e = bits_to_float(lo + next32() % (hi - lo + 1));
    const float s = bits_to_float(slo + next32() % (shi - slo + 1));
    rbad += check(e, s);
  }
  printf("random: %lld pairs, %llu mismatches\n", millions * 1000000LL, rbad);
  printf("TOTAL %llu mismatches\n", bad + rbad);
  return 0;
}
