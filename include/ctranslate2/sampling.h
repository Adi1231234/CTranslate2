#pragma once

#include "storage_view.h"

namespace ctranslate2 {

  // Base class for sampling from a score distribution.
  class Sampler {
  public:
    virtual ~Sampler() = default;

    // sample_ids and sampled_scores should be on CPU device.
    void operator()(const StorageView& scores,
                    StorageView& sampled_ids,
                    StorageView& sampled_scores,
                    dim_t num_samples = 1) const;
  protected:
    virtual void sample(const StorageView& scores,
                        dim_t num_samples,
                        StorageView& sampled_ids,
                        StorageView& sampled_scores) const = 0;
  };


  class BestSampler : public Sampler {
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores) const final;
  };


  // Rows sampled at temperatures of their own (GreedySearch's temperature variants): while a RowScalesScope is
  // active on a thread, RandomSampler multiplies each row of its scores by that row's value of `scales` (one a row,
  // 1 / its temperature in the scores' type and device), as it otherwise multiplies them all by its own.
  class RowScalesScope {
  public:
    explicit RowScalesScope(const StorageView* scales);
    ~RowScalesScope();
    RowScalesScope(const RowScalesScope&) = delete;
    RowScalesScope& operator=(const RowScalesScope&) = delete;
  private:
    const StorageView* _previous;
  };

  const StorageView* row_scales();


  class RandomSampler : public Sampler {
  public:
    RandomSampler(dim_t from_topk = 0, float topp = 1, float temperature = 1);
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores) const final;
  private:
    dim_t _from_topk;
    float _topp;
    float _temperature;
  };

}
