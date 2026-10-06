"""LongBroker (longform.py): fallback_batch.Broker for whole recordings. Each window's beam search (T=0) is a batch of
one in a Whisper stream (open_stream, the ladder-probe fork), decoded with the other recordings' windows, each exactly
as generate() alone decodes it; encoder calls are joined as the Broker joins them (every window's arithmetic its own).
LONG_LADDER_WORKERS=<n>: the ladder's sampled attempts run on a second instance of the model with n workers, each call
on its own as the Broker would run it (same model, same bits), on the recording's thread, so a ladder call of seconds
never holds up the windows waiting for the encoder (0: through the Broker's one dispatcher, which runs the encoder
calls and the ladder calls in turn). LONG_STATS=1: where the time goes (long_stats.py). LONG_LADDERS=skip
(measurement only, the rows change): a ladder's sampled attempts return the window's beam result, no decoding.
"""
import os, threading, time
import ctranslate2
from fallback_batch import Broker
from long_stats import LongStats

STREAM_OPTIONS = ("beam_size", "patience", "length_penalty", "repetition_penalty", "no_repeat_ngram_size",
                  "max_length", "return_scores", "return_no_speech_prob", "max_initial_timestamp_index",
                  "suppress_blank", "suppress_tokens")


class LongBroker(Broker):
    def __init__(self, model, windows, pending, ladder_model=None, ladder_workers=0, **kw):
        super().__init__(model, **kw)
        self._windows, self._pending_max = windows, pending
        self._stream = self._options = None
        self._calls, self._tag, self._lock = {}, 0, threading.Lock()
        self._ladder, self._lanes = ladder_model, threading.Semaphore(max(ladder_workers, 1))
        self._skip_ladders, self._last = os.environ.get("LONG_LADDERS") == "skip", threading.local()
        self.stats = LongStats()

    def _idle(self, delta):
        """A recording thread that stops (-1) or starts again (+1) being one the encoder's dispatcher waits for."""
        with self._cv:
            self._busy += delta
            self._cv.notify_all()

    def encode(self, features, to_cpu=False):
        t = time.monotonic()
        out = super().encode(features, to_cpu)
        self.stats.add("encode_wait_s", time.monotonic() - t)
        return out

    def _run(self, group, model=None):
        if group[0].kind == "encode":
            self.stats.add("encode_batch", len(group))
        return super()._run(group, model)

    def _call(self, call, kw=None):
        if call.kind == "encode" or self._ladder is None:
            t = time.monotonic()
            result = super()._call(call, kw)
            if call.kind != "encode":
                self.stats.add("ladder_s", time.monotonic() - t)
            return result
        call.kw = kw
        self._idle(-1)
        try:
            t = time.monotonic()
            with self._lanes:
                self.stats.add("ladder_wait_s", time.monotonic() - t)
                t = time.monotonic()
                self._run([call], model=self._ladder)
                self.stats.add("ladder_s", time.monotonic() - t)
        finally:
            self._idle(+1)
        return call.result

    def generate(self, encoder_output, prompts, **kw):
        if kw.get("beam_size", 1) == 1:                       # a sampled attempt of the ladder
            if self._skip_ladders:                            # measurement only: the window's beam result again
                return [self._last.result]
            return super().generate(encoder_output, prompts, **kw)
        if len(prompts) != 1 or set(kw) - set(STREAM_OPTIONS):
            raise ValueError(f"the stream takes one window's beam search, with {STREAM_OPTIONS}")
        stream = self._open(kw)
        with self._lock:
            tag, self._tag = self._tag, self._tag + 1
            call = self._calls[tag] = threading.Event()
        self._idle(-1)                                       # not an encoder caller while its window decodes
        t = time.monotonic()
        try:
            stream.submit(tag, ctranslate2.StorageView.from_array(encoder_output.array), [list(prompts[0])])
            self.stats.stream(+1)
            call.wait()
            self.stats.stream(-1)
        finally:
            self._idle(+1)
        self.stats.add("stream_s", time.monotonic() - t)
        if call.error:
            raise call.error
        self._last.result = call.result
        return [call.result]

    def _open(self, kw):
        options = {k: (list(v) if k == "suppress_tokens" else v) for k, v in kw.items()}
        with self._lock:
            if self._stream is None:
                self._stream, self._options = self._m.open_stream(
                    max_batches=self._windows, max_rows=self._windows * kw["beam_size"],
                    max_pending=self._pending_max, **options), options
                threading.Thread(target=self._collect, daemon=True).start()
            elif options != self._options:
                raise ValueError("a window's beam search options differ from the stream's")
        return self._stream

    def _collect(self):
        try:
            while (item := self._stream.next()) is not None:
                tag, results = item
                with self._lock:
                    call = self._calls.pop(tag)
                call.result, call.error = results[0], None
                call.set()
        except Exception as e:                               # every waiting window raises it
            with self._lock:
                calls, self._calls = list(self._calls.values()), {}
            for call in calls:
                call.result, call.error = None, e
                call.set()
