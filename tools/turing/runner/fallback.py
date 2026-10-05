"""Where the fallback clips (the full temperature ladder) run, with each clip's ladder time in the log.
'inline' (default): right away on the caller's thread, so a ladder never needs GPU memory at the same time as a
batch; 'async' (RUN_FALLBACK=async): a side thread and its own CTranslate2 worker, next to the batched path. On the
store PC's 8 GB GPU async paged 760-860 MB to system memory and ran 30 real units at 21.1x, inline 26.1x (26.9).
'skip' (measurement only, never production): no ladder at all, the row says so (path "fallback_skipped"), to time
the batched path alone. 'batched': several ladders at once whose GPU calls are joined (fallback_batch.py)."""
import copy, os, threading, time
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


class BatchedPool:
    """Ladders on RUN_FALLBACK_THREADS threads (default 8) whose models' CTranslate2 calls go through one broker.
    Fallback clips come one at a time, so ladders wait until RUN_FALLBACK_GATHER of them are queued (default: the
    threads), RUN_FALLBACK_GATHER_S seconds pass after the first (default 600) or flush() (the batched path is
    done), and then start together; the writer holds their units meanwhile."""
    def __init__(self, log):
        self.threads = int(os.environ.get("RUN_FALLBACK_THREADS", "8"))
        self.gather = int(os.environ.get("RUN_FALLBACK_GATHER", str(self.threads)))
        self.max_wait = float(os.environ.get("RUN_FALLBACK_GATHER_S", "600"))
        self.executor = ThreadPoolExecutor(max_workers=self.threads)
        self.log, self.proxies, self.waiting, self.lock, self.timer = log, {}, [], threading.Lock(), None

    def submit(self, fn, model, uuid, wav):
        from fallback_batch import Broker, ladder_of
        if id(model) not in self.proxies:
            from engine import EXACT
            proxy = copy.copy(model)
            speculate = ([t for t in EXACT["temperature"] if t > 0]
                         if os.environ.get("RUN_FALLBACK_SPECULATE") == "1" else None)
            proxy.model = Broker(model.model, os.environ.get("RUN_FALLBACK_SAMPLING") == "batched",
                                 os.environ.get("RUN_FALLBACK_SEEDS") == "1", speculate=speculate,
                                 spec_clips=int(os.environ.get("RUN_FALLBACK_SPEC_CLIPS", "2")))
            self.proxies[id(model)] = proxy
        proxy, done = self.proxies[id(model)], Future()

        def ladder():
            try:
                with ladder_of(uuid):
                    done.set_result(_timed(self.log, fn, proxy, uuid, wav))
            except Exception as e:                         # the writer raises it
                done.set_exception(e)
            finally:
                proxy.model.ladder_finished()
        with self.lock:
            self.waiting.append((proxy, ladder))
            if len(self.waiting) >= self.gather:
                self._start()
            elif self.timer is None:                         # the first one waiting: start them all by then
                self.timer = threading.Timer(self.max_wait, self.flush)
                self.timer.daemon = True
                self.timer.start()
        return done

    def _start(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        for proxy, _ in self.waiting:
            proxy.model.ladder_started()                     # before any of them calls the broker
        for _, ladder in self.waiting:
            self.executor.submit(ladder)
        self.waiting = []

    def flush(self):
        with self.lock:
            self._start()


class SkipPool:
    def submit(self, fn, model, uuid, wav):
        done = Future()
        done.set_result({"uuid": uuid, "dur_s": len(wav) / 16000, "text": None, "segments": [],
                         "path": "fallback_skipped"})
        return done


def make_pool(kind, log):
    """The pool for RUN_FALLBACK=kind, and how many CTranslate2 workers it needs of its own."""
    if kind == "skip":
        return SkipPool(), 0
    if kind == "batched":
        return BatchedPool(log), 1
    return (InlinePool(log), 0) if kind == "inline" else (AsyncPool(log), 1)
