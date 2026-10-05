#include "cuda/shared_memory_rows.h"

#include "ctranslate2/allocator.h"
#include "cuda/clip_groups.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local const SharedMemoryRows* active = nullptr;

    const SharedMemoryRows* shared_memory_rows() {
      return active;
    }

    bool shared_memory_rows_enabled() {
      static const bool enabled = read_bool_from_env("CT2_SHARED_MEMORY_ROWS", true) && cublas_verified_on(8, 9);
      return enabled;
    }

    SharedMemoryRowsScope::SharedMemoryRowsScope(const SharedMemoryRows& rows)
      : _previous(active) {
      active = &rows;
    }

    SharedMemoryRowsScope::~SharedMemoryRowsScope() {
      active = _previous;
    }

    // Pointer arrays of a batched product over entries e = row * heads + h: a (the shared memory, by the row's clip),
    // b and c (per entry, strided).
    __global__ void shared_rows_pointers(const int32_t* row_clip, const __half* a, size_t a_stride, const __half* b,
                                         size_t b_stride, __half* c, size_t c_stride, int heads, int entries,
                                         const void** pa, const void** pb, void** pc) {
      for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < entries; e += gridDim.x * blockDim.x) {
        const int row = e / heads, h = e % heads;
        pa[e] = a + (static_cast<size_t>(row_clip[row]) * heads + h) * a_stride;
        pb[e] = b + static_cast<size_t>(e) * b_stride;
        pc[e] = c + static_cast<size_t>(e) * c_stride;
      }
    }

    static void batched(bool trans_a, int m, int n, int k, float alpha, const __half* a, size_t a_stride, int lda,
                        const __half* b, size_t b_stride, int ldb, __half* c, size_t c_stride, int ldc, dim_t heads) {
      const SharedMemoryRows& rows = *shared_memory_rows();
      // Rows [first, first + count) as one product of count x heads entries.
      const auto product = [&](dim_t first, dim_t count) {
        const int entries = static_cast<int>(count * heads);
        Allocator& allocator = get_allocator<Device::CUDA>();
        void** pointers = static_cast<void**>(allocator.allocate(3 * entries * sizeof (void*)));
        cudaStream_t stream = get_cuda_stream();
        shared_rows_pointers<<<(entries + 127) / 128, 128, 0, stream>>>(
          rows.row_clip + first, a, a_stride, b + first * heads * b_stride, b_stride, c + first * heads * c_stride,
          c_stride, static_cast<int>(heads), entries, const_cast<const void**>(pointers),
          const_cast<const void**>(pointers + entries), pointers + 2 * entries);
        const float beta = 0;
        CUBLAS_CHECK(cublasGemmBatchedEx(get_cublas_handle(), trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, &alpha, const_cast<const void**>(pointers), CUDA_R_16F, lda,
                                         const_cast<const void**>(pointers + entries), CUDA_R_16F, ldb, &beta,
                                         pointers + 2 * entries, CUDA_R_16F, ldc, entries, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT));
        allocator.free(pointers);                              // stream-ordered, after the product
      };
      // Inputs decoded together (cuda/clip_groups.h, rows grouped by input): each group's entries as that input's
      // own product would have them (cuBLAS's kernel for these products depends on the entry count).
      if (!for_each_clip_group(rows.rows, product))
        product(0, rows.rows);
    }

    // primitives<CUDA>::gemm_batch_strided's cuBLAS call for MatMul(queries, keys, trans_b): column-major
    // (keys x 1) = K^T q, A = K (trans), B = q.
    void shared_rows_scores(const float16_t* q, const float16_t* k, float16_t* scores, dim_t heads, dim_t keys,
                            dim_t depth, float alpha) {
      const auto h = [](const float16_t* p) { return reinterpret_cast<const __half*>(p); };
      batched(true, static_cast<int>(keys), 1, static_cast<int>(depth), alpha, h(k), keys * depth,
              static_cast<int>(depth), h(q), depth, static_cast<int>(depth), reinterpret_cast<__half*>(scores),
              keys, static_cast<int>(keys), heads);
    }

    // ... and for MatMul(attention, values): (depth x 1) = V p, A = V, B = p.
    void shared_rows_output(const float16_t* p, const float16_t* v, float16_t* out, dim_t heads, dim_t keys,
                            dim_t depth) {
      const auto h = [](const float16_t* x) { return reinterpret_cast<const __half*>(x); };
      batched(false, static_cast<int>(depth), 1, static_cast<int>(keys), 1.f, h(v), keys * depth,
              static_cast<int>(depth), h(p), keys, static_cast<int>(keys), reinterpret_cast<__half*>(out), depth,
              static_cast<int>(depth), heads);
    }

  }
}
