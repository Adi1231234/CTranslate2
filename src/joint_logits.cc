#include "joint_logits.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "ctranslate2/ops/softmax.h"
#include "layers/joint_step.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/disable_tokens.h"
#  include "cuda/pinned_buffer.h"
#  include "cuda/timestamp_rules.h"
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {

  static thread_local JointLogits* current_joint = nullptr;

  // Every check's maxima and sums on the host (cuda/timestamp_rules.h), read once the stream has synchronized.
  struct JointLogits::Answers {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    cuda::PinnedBuffer host;
#endif
    DataType dtype = DataType::FLOAT32;
    size_t count = 0;
    bool flushed = false;
    bool read = false;
    std::vector<bool> all;

    const std::vector<bool>& read_all() {
      if (!flushed)
        throw std::logic_error("JointLogits: answers read before the joint work was flushed");
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      if (!read) {
        switch (dtype) {
        case DataType::FLOAT32: all = cuda::read_sample_timestamps<float>(host, count); break;
        case DataType::FLOAT16: all = cuda::read_sample_timestamps<float16_t>(host, count); break;
        case DataType::BFLOAT16: all = cuda::read_sample_timestamps<bfloat16_t>(host, count); break;
        default: throw std::invalid_argument("JointLogits: unsupported logits type");
        }
        read = true;
      }
#endif
      return all;
    }
  };

  JointLogits::JointLogits(StorageView& logits)
    : _logits(logits)
    , _previous(current_joint)
    , _answers(std::make_shared<Answers>()) {
    current_joint = this;
  }

  JointLogits::~JointLogits() {
    current_joint = _previous;
  }

  JointLogits* JointLogits::current() {
    return current_joint;
  }

  bool JointLogits::accepts(const StorageView& part) const {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    if (cuda::use_stock_kernels() || part.device() != Device::CUDA || part.dtype() != _logits.dtype()
        || part.rank() != 2 || _logits.rank() != 2 || part.dim(1) != _logits.dim(1) || part.empty())
      return false;
    const auto* base = static_cast<const char*>(_logits.buffer());
    const auto* first = static_cast<const char*>(part.buffer());
    const auto row_bytes = static_cast<std::ptrdiff_t>(_logits.dim(1) * _logits.item_size());
    return first >= base && (first - base) % row_bytes == 0
      && (first - base) / row_bytes + part.dim(0) <= _logits.dim(0);
#else
    (void)part;
    return false;
#endif
  }

  std::function<std::vector<bool>()> JointLogits::add(const StorageView& part, DisableTokens& disable_tokens,
                                                      const std::vector<dim_t>& check_rows, dim_t timestamp_begin,
                                                      dim_t timestamp_end) {
    if (!accepts(part) || _answers->flushed
        || (_timestamp_begin >= 0 && (timestamp_begin != _timestamp_begin || timestamp_end != _timestamp_end)))
      throw std::logic_error("JointLogits: a search's logits it does not take");
    _timestamp_begin = timestamp_begin;
    _timestamp_end = timestamp_end;
    const auto row_bytes = static_cast<std::ptrdiff_t>(_logits.dim(1) * _logits.item_size());
    const dim_t row = (static_cast<const char*>(part.buffer()) - static_cast<const char*>(_logits.buffer()))
      / row_bytes;
    _parts.push_back({row, part.dim(0), &disable_tokens});
    const size_t start = _check_rows.size();
    for (const dim_t r : check_rows)
      _check_rows.push_back(row + r);
    return [answers = _answers, start, count = check_rows.size()]() {
      const std::vector<bool>& all = answers->read_all();
      return std::vector<bool>(all.begin() + start, all.begin() + start + count);
    };
  }

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
  // Every search's disabled tokens in one launch, then the log-softmax of the block of their rows and the
  // timestamp checks of all of them (rows: indices into the block).
  template <typename T>
  static void queue_joint(StorageView& logits, const std::vector<int32_t>& ranges,
                          const std::vector<int32_t>& singles, float value, StorageView& block,
                          const std::vector<dim_t>& rows, dim_t begin, dim_t end, cuda::PinnedBuffer& host) {
    if (!ranges.empty() || !singles.empty())
      cuda::disable_tokens(logits.data<T>(), static_cast<T>(value), ranges, singles, {},
                           static_cast<int32_t>(logits.dim(0)), static_cast<int32_t>(logits.dim(1)));
    StorageView log_probs(block.dtype(), block.device());
    ops::LogSoftMax()(block, log_probs);
    cuda::queue_sample_timestamps(log_probs.data<T>(), log_probs.dim(-1), rows, begin, end, host);
  }
#endif

  void JointLogits::flush() {
    if (_parts.empty())
      return;
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    std::vector<int32_t> ranges, singles;
    const float value = _parts.front().disable_tokens->disable_value();
    dim_t first = _logits.dim(0), end = 0;
    for (const Part& part : _parts) {
      if (part.disable_tokens->disable_value() != value)
        throw std::logic_error("JointLogits: searches disabling tokens with different values");
      part.disable_tokens->move_to(part.row, ranges, singles);
      first = std::min(first, part.row);
      end = std::max(end, part.row + part.rows);
    }
    StorageView block = layers::rows_view(_logits, first, end - first);
    std::vector<dim_t> rows(_check_rows);
    for (dim_t& r : rows)
      r -= first;
    Answers& answers = *_answers;
    answers.dtype = _logits.dtype();
    answers.count = rows.size();
    switch (_logits.dtype()) {
    case DataType::FLOAT32:
      queue_joint<float>(_logits, ranges, singles, value, block, rows, _timestamp_begin, _timestamp_end,
                         answers.host);
      break;
    case DataType::FLOAT16:
      queue_joint<float16_t>(_logits, ranges, singles, value, block, rows, _timestamp_begin, _timestamp_end,
                             answers.host);
      break;
    case DataType::BFLOAT16:
      queue_joint<bfloat16_t>(_logits, ranges, singles, value, block, rows, _timestamp_begin, _timestamp_end,
                              answers.host);
      break;
    default:
      throw std::invalid_argument("JointLogits: unsupported logits type");
    }
    answers.flushed = true;
#endif
  }

}
