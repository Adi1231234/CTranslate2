"""The batched pipeline's log-mel features computed ahead on a thread pool. faster-whisper's
BatchedInferencePipeline.transcribe computes every clip's features one after the other before any GPU work of
the call (0.6 s of 150 clips on the store PC, the GPU idle meanwhile); numpy releases the GIL, so a pool does the
same computations about 5x sooner. FeatureCache wraps model.feature_extractor: a call with a clip that was
prefetched (same samples, checked with np.array_equal) returns that clip's precomputed features, anything else
(e.g. a fallback's own call) is computed as before. Same function, same input: the same features."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class FeatureCache:
    def __init__(self, extractor, threads=6):
        self.extractor, self.pool, self.pending = extractor, ThreadPoolExecutor(threads), {}

    def __getattr__(self, name):                   # sampling_rate, chunk_length, ... of the real extractor
        return getattr(self.extractor, name)

    def prefetch(self, waves):
        self.pending.clear()                        # a call's leftovers never match another call
        for w in waves:
            self.pending.setdefault(len(w), []).append((w, self.pool.submit(self.extractor, w)))

    def __call__(self, audio, *args, **kwargs):
        if not args and not kwargs:
            entries = self.pending.get(len(audio), [])
            for i, (w, future) in enumerate(entries):
                if np.array_equal(w, audio):
                    del entries[i]
                    return future.result()
        return self.extractor(audio, *args, **kwargs)


def feature_cache(model):
    """model.feature_extractor as a FeatureCache (installed once)."""
    if not isinstance(model.feature_extractor, FeatureCache):
        model.feature_extractor = FeatureCache(model.feature_extractor)
    return model.feature_extractor
