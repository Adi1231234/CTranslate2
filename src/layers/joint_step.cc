#include "joint_step.h"

namespace ctranslate2 {
  namespace layers {

    static thread_local const JointStep* active = nullptr;

    const JointStep* joint_step() {
      return active;
    }

    const void* const* JointStep::memory(size_t l) const {
      const auto* table = reinterpret_cast<const void* const*>(memory_table.data<int32_t>());
      return table + l * 2 * clips;
    }

    JointStepScope::JointStepScope(const JointStep& step)
      : _previous(active) {
      active = &step;
    }

    JointStepScope::~JointStepScope() {
      active = _previous;
    }

    StorageView rows_view(StorageView& x, dim_t begin, dim_t count) {
      Shape shape = x.shape();
      const dim_t row = x.size() / shape[0];
      shape[0] = count;
      StorageView view(x.dtype(), x.device());
      view.view(static_cast<char*>(x.buffer()) + begin * row * x.item_size(), std::move(shape));
      return view;
    }

  }
}
