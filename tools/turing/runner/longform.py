"""Whole recordings (the corpus work lists, corpus-backend RUN_FORMAT.md): faster-whisper's own sequential transcribe
with the crowd-v5 parameters (engine.EXACT, as the fallback runs it: 30 s windows, each conditioned on the text before
it, the temperature ladder), each recording on a thread of its own, LONG_THREADS (default 48) at once. Their
CTranslate2 calls go through LongBroker: encoder calls and sampled attempts as fallback_batch.Broker joins them (every
window's arithmetic its own), and each window's beam search (T=0) a batch of one in a Whisper stream, decoded with the
other recordings' windows, each exactly as generate() alone decodes it (Whisper.open_stream, the ladder-probe fork).
LONG_PENDING (default 2): windows submitted and not yet decoding. RUN_FALLBACK_SEEDS/SAMPLING/SPECULATE/SPEC_CLIPS as
in fallback_batch.py."""
import copy, os, threading
from concurrent.futures import ThreadPoolExecutor
import ctranslate2
import engine
from fallback_batch import Broker, ladder_of

STREAM_OPTIONS = ("beam_size", "patience", "length_penalty", "repetition_penalty", "no_repeat_ngram_size",
                  "max_length", "return_scores", "return_no_speech_prob", "max_initial_timestamp_index",
                  "suppress_blank", "suppress_tokens")


class LongBroker(Broker):
    """fallback_batch.Broker whose beam searches go to a Whisper stream, one window a batch."""

    def __init__(self, model, windows, pending, **kw):
        super().__init__(model, **kw)
        self._windows, self._pending_max = windows, pending
        self._stream = self._options = None
        self._calls, self._tag, self._lock = {}, 0, threading.Lock()

    def generate(self, encoder_output, prompts, **kw):
        if kw.get("beam_size", 1) == 1:                       # a sampled attempt of the ladder
            return super().generate(encoder_output, prompts, **kw)
        if len(prompts) != 1 or set(kw) - set(STREAM_OPTIONS):
            raise ValueError(f"the stream takes one window's beam search, with {STREAM_OPTIONS}")
        stream = self._open(kw)
        with self._lock:
            tag, self._tag = self._tag, self._tag + 1
            call = self._calls[tag] = threading.Event()
        with self._cv:                                       # not an encoder caller while its window decodes
            self._busy -= 1
            self._cv.notify_all()
        try:
            stream.submit(tag, ctranslate2.StorageView.from_array(encoder_output.array), [list(prompts[0])])
            call.wait()
        finally:
            with self._cv:
                self._busy += 1
        if call.error:
            raise call.error
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


class LongEngine:
    """Recordings submitted one by one, transcribed LONG_THREADS at a time; submit(key, load) returns a Future of the
    row, load() (run on the recording's thread) its 16 kHz samples."""

    def __init__(self, model):
        threads = int(os.environ.get("LONG_THREADS", "48"))
        speculate = ([t for t in engine.EXACT["temperature"] if t > 0]
                     if os.environ.get("RUN_FALLBACK_SPECULATE") == "1" else None)
        self.proxy = copy.copy(model)
        self.proxy.model = LongBroker(
            model.model, threads, int(os.environ.get("LONG_PENDING", "2")),
            join_sampled=os.environ.get("RUN_FALLBACK_SAMPLING") == "batched",
            seeded=os.environ.get("RUN_FALLBACK_SEEDS") == "1", speculate=speculate,
            spec_clips=int(os.environ.get("RUN_FALLBACK_SPEC_CLIPS", "2")))
        self.executor = ThreadPoolExecutor(max_workers=threads)

    def submit(self, key, load):
        def run():
            wav = load()
            self.proxy.model.ladder_started()
            try:
                with ladder_of(key):
                    segments, _ = self.proxy.transcribe(wav, **engine.EXACT)
                    return engine._row(key, wav, list(segments), "long")
            finally:
                self.proxy.model.ladder_finished()
        return self.executor.submit(run)
