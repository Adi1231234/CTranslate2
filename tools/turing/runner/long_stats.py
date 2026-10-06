"""Where a long-recording run's time goes (LONG_STATS=1, longform.py): every encoder call's wait and the batch it ran
in, every window's time in the stream and how many windows decode at once (time-weighted), every ladder call's
time, and where the threads are (thread_sampler.py). A report every LONG_STATS_S seconds (default 60) and one at the
end (report()), to stdout."""
import os, threading, time
from thread_sampler import ThreadSampler


class LongStats:
    def __init__(self):
        self.on = os.environ.get("LONG_STATS") == "1"
        self.lock, self.t0 = threading.Lock(), time.monotonic()
        self.sums, self.counts = {}, {}
        self.in_stream, self.area, self.last = 0, 0.0, self.t0
        self.sampler = ThreadSampler() if self.on else None
        if self.on:
            threading.Thread(target=self._report, daemon=True).start()

    def report(self):
        return self.line() + ("\n" + self.sampler.top() if self.sampler else "")

    def add(self, name, value=1.0):
        if self.on:
            with self.lock:
                self.sums[name] = self.sums.get(name, 0.0) + value
                self.counts[name] = self.counts.get(name, 0) + 1

    def stream(self, delta):
        """A window entering (+1) or leaving (-1) the stream."""
        if self.on:
            with self.lock:
                now = time.monotonic()
                self.area += self.in_stream * (now - self.last)
                self.in_stream, self.last = self.in_stream + delta, now

    def line(self):
        with self.lock:
            now = time.monotonic()
            span = now - self.t0
            area = self.area + self.in_stream * (now - self.last)
            mean = lambda k: self.sums.get(k, 0.0) / max(self.counts.get(k, 0), 1)
            return (f"STATS {span:.0f} s: windows {self.counts.get('stream_s', 0)} ({self.counts.get('stream_s', 0) / span:.2f}/s),"
                    f" in the stream {area / span:.1f} on average, {mean('stream_s'):.2f} s each;"
                    f" encoder calls {self.counts.get('encode_batch', 0)} of {mean('encode_batch'):.1f} windows,"
                    f" a window's wait {mean('encode_wait_s'):.2f} s;"
                    f" ladder calls {self.counts.get('ladder_s', 0)}, {self.sums.get('ladder_s', 0.0):.0f} s"
                    f" ({mean('ladder_s'):.2f} s each), {self.sums.get('ladder_wait_s', 0.0):.0f} s waiting for a worker")

    def _report(self):
        every = float(os.environ.get("LONG_STATS_S", "60"))
        while True:
            time.sleep(every)
            print(self.report(), flush=True)
