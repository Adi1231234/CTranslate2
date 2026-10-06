"""Where a process's threads spend their time (LONG_STATS=1, longform_run.py): once a second every thread's stack
(sys._current_frames), each counted at its innermost frame in our code or faster-whisper (the place it waits or
works) and at its innermost frame of all (the blocking call itself). top() lists the most frequent, with the share of
thread-seconds each holds. Pure Python: no ptrace, no profiler in the container."""
import collections, os, sys, threading, time

OURS = ("longform", "long_broker", "fallback_batch", "longform_run", "chunked_features", "engine", "transcribe",
        "feature_extractor", "tokenizer", "audio")


def _where(frame):
    inner = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"
    f = frame
    while f is not None:
        name = os.path.splitext(os.path.basename(f.f_code.co_filename))[0]
        if name in OURS:
            return f"{name}:{f.f_code.co_name}:{f.f_lineno}", inner
        f = f.f_back
    return "other", inner


class ThreadSampler:
    def __init__(self, every=1.0):
        self.ours, self.inner, self.samples = collections.Counter(), collections.Counter(), 0
        self.lock, self.every = threading.Lock(), every
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        me = threading.get_ident()
        while True:
            time.sleep(self.every)
            frames = sys._current_frames()
            with self.lock:
                self.samples += 1
                for ident, frame in frames.items():
                    if ident != me:
                        ours, inner = _where(frame)
                        self.ours[ours] += 1
                        self.inner[inner] += 1

    def top(self, n=12):
        with self.lock:
            total = sum(self.ours.values()) or 1
            lines = [f"THREADS {self.samples} samples, {total / max(self.samples, 1):.0f} threads on average"]
            lines += [f"  {100 * c / total:5.1f}% at {k}" for k, c in self.ours.most_common(n)]
            lines += [f"  {100 * c / total:5.1f}% inside {k}" for k, c in self.inner.most_common(6)]
        return "\n".join(lines)
