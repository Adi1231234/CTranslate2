"""Whole recordings (the corpus work lists, corpus-backend RUN_FORMAT.md): faster-whisper's own sequential transcribe
with the crowd-v5 parameters (engine.EXACT, as the fallback runs it: 30 s windows, each conditioned on the text before
it, the temperature ladder), each recording on a thread of its own, LONG_THREADS (default 48) at once, their
CTranslate2 calls through LongBroker (long_broker.py). LONG_PENDING (default 2): windows submitted and not yet
decoding. RUN_FALLBACK_SEEDS/SAMPLING/SPECULATE/SPEC_CLIPS as in fallback_batch.py. The model needs 2 CTranslate2
workers: the stream's decoding loop holds one while the stream is open, and with one only, every encoder call waited
for it forever (long1, 6.10.2026)."""
import copy, os, threading
from concurrent.futures import ThreadPoolExecutor
import ctranslate2
import engine
from chunked_features import ChunkedFeatures
from fallback_batch import ladder_of
from long_broker import LongBroker


class LongEngine:
    """Recordings submitted one by one, transcribed LONG_THREADS at a time; submit(key, load, hours) returns a Future
    of the row, load() (run on the recording's thread) its 16 kHz samples. The features are the original's bytes in
    less memory (chunked_features.py). LONG_MAX_HOURS (default 0, no limit): a recording waits to start while the
    recordings in progress and it would hold more audio than that (a recording alone always starts); each hour holds
    ~0.4 GB of samples and features. model_path: the model's folder, for the ladders' instance (LONG_LADDER_WORKERS)."""

    def __init__(self, model, model_path=None):
        if model.model.num_workers < 2:   # the stream's decoding loop holds a worker as long as the stream is open
            raise ValueError("LongEngine needs a model of 2 workers: the stream's and the encoder's and ladders'")
        threads = int(os.environ.get("LONG_THREADS", "48"))
        ladders = int(os.environ.get("LONG_LADDER_WORKERS", "0"))
        speculate = ([t for t in engine.EXACT["temperature"] if t > 0]
                     if os.environ.get("RUN_FALLBACK_SPECULATE") == "1" else None)
        ladder_model = (ctranslate2.models.Whisper(model_path, device="cuda", compute_type="default",
                                                   inter_threads=ladders, intra_threads=1) if ladders else None)
        self.proxy = copy.copy(model)
        self.proxy.feature_extractor = ChunkedFeatures(model.feature_extractor)
        self.proxy.model = LongBroker(
            model.model, threads, int(os.environ.get("LONG_PENDING", "2")), ladder_model, ladders,
            join_sampled=os.environ.get("RUN_FALLBACK_SAMPLING") == "batched",
            seeded=os.environ.get("RUN_FALLBACK_SEEDS") == "1", speculate=speculate,
            spec_clips=int(os.environ.get("RUN_FALLBACK_SPEC_CLIPS", "2")))
        self.stats = self.proxy.model.stats
        self.executor = ThreadPoolExecutor(max_workers=threads)
        self.budget, self.held, self.room = float(os.environ.get("LONG_MAX_HOURS", "0")), 0.0, threading.Condition()

    def submit(self, key, load, hours=0.0):
        def run():
            with self.room:
                while self.budget and self.held and self.held + hours > self.budget:
                    self.room.wait()
                self.held += hours
            try:
                wav = load()
                self.stats.recording(+1)
                self.proxy.model.ladder_started()
                try:
                    with ladder_of(key):
                        segments, _ = self.proxy.transcribe(wav, **engine.EXACT)
                        return engine._row(key, wav, list(self._progress(segments, len(wav) / 16000)), "long")
                finally:
                    self.proxy.model.ladder_finished()
                    self.stats.recording(-1)
            finally:
                with self.room:
                    self.held -= hours
                    self.room.notify_all()
        return self.executor.submit(run)

    def _progress(self, segments, duration):
        """The segments, each one's advance past the last counted as audio done (LONG_STATS), the rest at the end."""
        done = 0.0
        for segment in segments:
            if segment.end > done:
                self.stats.add("audio_s", min(segment.end, duration) - done)
                done = min(segment.end, duration)
            yield segment
        self.stats.add("audio_s", duration - done)
