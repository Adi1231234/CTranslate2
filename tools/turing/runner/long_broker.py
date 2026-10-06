"""LongBroker (longform.py): fallback_batch.Broker for whole recordings. Each window's beam search (T=0) is a batch of
one in a Whisper stream (open_stream, the ladder-probe fork), decoded with the other recordings' windows, each exactly
as generate() alone decodes it; encoder calls are joined as the Broker joins them (every window's arithmetic its own).
LONG_LADDER_WORKERS=<n>: the ladder's sampled attempts run on n of the model's own workers (longform.py gives it one
a lane), at most n at once, each call on its own as the Broker would run it (same model, same bits), on the
recording's thread, so a ladder call of seconds never holds up the windows waiting for the encoder (0: through the
Broker's one dispatcher, which runs the encoder calls and the ladder calls in turn). LONG_STATS=1: where the time goes (long_stats.py). LONG_LADDERS=skip
(measurement only, the rows change): a ladder's sampled attempts return the window's beam result, no decoding.
LONG_LADDERS=stream: a ladder's speculated attempts (RUN_FALLBACK_SPECULATE, seeded: each what its call alone draws)
are one sampled batch of a stream (WhisperStream.submit_sampled), exactly what the call alone returns. In the windows'
stream (long25 s1: rows identical, but every ladder op then waits its turn in the windows' steps, a ladder ~20 s, and
8 ladders' caches ran the GPU out of memory) it holds LONG_STREAM_LADDERS (default 8) batches besides the windows and
LONG_STREAM_ROWS rows (default 320, the most rows whose products cuda/clip_groups.h proves row-independent).
LONG_LADDER_STREAM=1: in a stream of their own instead, on a worker of its own (longform.workers_needed), beside the
windows' as the lanes were: up to LONG_LADDER_BATCHES ladders (default 2, the lanes' memory) decode together, the
decoder's weights read once a step for them all. LONG_LADDER_PRIORITY=high: that stream's GPU work ahead of the
windows' (a recording waits for its ladder; the windows' stream has many recordings in flight).
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
        self._in_stream = os.environ.get("LONG_LADDERS") == "stream"
        self._own_stream = self._in_stream and os.environ.get("LONG_LADDER_STREAM") == "1"
        self._ladder_stream = None
        if self._in_stream and not (self._speculate and self._seeded):
            raise ValueError("LONG_LADDERS=stream takes speculated, seeded ladders (RUN_FALLBACK_SPECULATE=1, "
                             "RUN_FALLBACK_SEEDS=1)")
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
        if call.kind == "spec" and self._in_stream:
            return self._sampled(call, kw)
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

    def _sampled(self, call, kw):
        """A window's speculated attempts (Broker._speculated: its temperatures from this one on, each with its seed)
        as a sampled batch of the stream; the results one a temperature, as generate() returns them."""
        if self._stream is None:
            raise RuntimeError("a ladder before the stream's first window")
        stream = self._ladders() if self._own_stream else self._stream
        with self._lock:
            tag, self._tag = self._tag, self._tag + 1
            event = self._calls[tag] = threading.Event()
        self._idle(-1)
        t = time.monotonic()
        try:
            stream.submit_sampled(tag, ctranslate2.StorageView.from_array(call.data), [list(call.prompt)],
                                  group_size=1, sampling_temperatures=list(call.key[3]), sampling_seeds=call.seeds,
                                  **kw)
            event.wait()
        finally:
            self._idle(+1)
        self.stats.add("ladder_s", time.monotonic() - t)
        if event.error:
            raise event.error
        return event.result

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
        self._last.result = call.result[0]
        return [call.result[0]]

    def _open(self, kw):
        options = {k: (list(v) if k == "suppress_tokens" else v) for k, v in kw.items()}
        with self._lock:
            if self._stream is None:
                batches, rows = self._windows, self._windows * kw["beam_size"]
                if self._in_stream and not self._own_stream:   # room for the ladders' sampled batches
                    batches += int(os.environ.get("LONG_STREAM_LADDERS", "8"))
                    rows = int(os.environ.get("LONG_STREAM_ROWS", "320"))
                self._stream, self._options = self._m.open_stream(
                    max_batches=batches, max_rows=rows, max_pending=self._pending_max, **options), options
                threading.Thread(target=self._collect, args=(self._stream,), daemon=True).start()
            elif options != self._options:
                raise ValueError("a window's beam search options differ from the stream's")
        return self._stream

    def _ladders(self):
        """The ladders' own stream (LONG_LADDER_STREAM=1), opened with the windows' options (its batches bring
        their own)."""
        with self._lock:
            if self._ladder_stream is None:
                batches = int(os.environ.get("LONG_LADDER_BATCHES", "2"))
                # its kernels ahead of the windows' (only asked for: packages before b21d73cc take no such argument)
                high = {"high_priority": True} if os.environ.get("LONG_LADDER_PRIORITY") == "high" else {}
                self._ladder_stream = self._m.open_stream(max_batches=batches, max_rows=batches * 25,
                                                          max_pending=self._pending_max, **high, **self._options)
                threading.Thread(target=self._collect, args=(self._ladder_stream,), daemon=True).start()
        return self._ladder_stream

    def _collect(self, stream):
        try:
            while (item := stream.next()) is not None:
                tag, results = item
                with self._lock:
                    call = self._calls.pop(tag)
                call.result, call.error = results, None     # a window's: one; a sampled batch's: one a temperature
                call.set()
        except Exception as e:                               # every waiting window raises it
            with self._lock:
                calls, self._calls = list(self._calls.values()), {}
            for call in calls:
                call.result, call.error = None, e
                call.set()
