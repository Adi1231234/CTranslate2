"""Where the fallback clips (the full temperature ladder) run, with each clip's ladder time in the log.
'inline' (default): right away on the caller's thread, so a ladder never needs GPU memory at the same time as a
batch; 'async' (RUN_FALLBACK=async): a side thread and its own CTranslate2 worker, next to the batched path. On the
store PC's 8 GB GPU async paged 760-860 MB to system memory and ran 30 real units at 21.1x, inline 26.1x (26.9)."""
import time
from concurrent.futures import Future, ThreadPoolExecutor


def _timed(log, fn, model, uuid, wav):
    t = time.time()
    r = fn(model, uuid, wav)
    top = max((s["temperature"] for s in r["segments"]), default=None)
    log(f"fallback {len(wav) / 16000:.1f}s audio took {time.time() - t:.1f}s, final T {top}")
    return r


class AsyncPool(ThreadPoolExecutor):
    def __init__(self, log):
        super().__init__(max_workers=1)
        self.log = log

    def submit(self, fn, model, uuid, wav):
        return super().submit(_timed, self.log, fn, model, uuid, wav)


class InlinePool:
    def __init__(self, log):
        self.log = log

    def submit(self, fn, model, uuid, wav):
        done = Future()
        done.set_result(_timed(self.log, fn, model, uuid, wav))
        return done


def make_pool(kind, log):
    """The pool for RUN_FALLBACK=kind, and how many CTranslate2 workers it needs of its own."""
    return (InlinePool(log), 0) if kind == "inline" else (AsyncPool(log), 1)
