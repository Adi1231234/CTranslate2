"""Units decoded through one CTranslate2 Whisper stream (Whisper.open_stream, the ladder-probe fork): the pipelined
engine's batches (the same clips, features, encoder calls, prompts and options, so each batch's results are the
same bits), but a batch starts decoding as soon as there is room, across units, and every decoding step runs all
the batches in flight at once (their weights read once, no batch's long tail decoding alone).
add_unit (the main thread) runs BatchedInferencePipeline.transcribe's setup of a unit and encodes and submits its
batches; a collector thread turns every finished batch into faster-whisper's segments (BatchedInferencePipeline.forward
on its results) and, once all of a unit's batches are back, hands the unit's rows to the writer (engine.unit_rows:
the fallback for the clips that fail). STREAM_BATCHES / STREAM_ROWS / STREAM_PENDING: the stream's limits (8, 320, 2).
"""
import os, threading
import engine
from pipelined import generate_outputs, make_segment
from resume import ResumeCheck


class _Unit:
    def __init__(self, uid, clips, order, bad, ts):
        self.uid, self.clips, self.order, self.bad, self.ts = uid, clips, order, bad, ts
        self.batches, self.unfinished, self.left = {}, [], 0


class _Submit(ResumeCheck):
    """transcribe()'s setup of a unit (features, tokenizer, options), its batches to the stream instead of forward()."""
    def __init__(self, model, submit):
        super().__init__(model)
        self.submit = submit

    def _batched_segments_generator(self, features, tokenizer, chunks_metadata, batch_size, options, log_progress):
        self.submit(features, tokenizer, chunks_metadata, batch_size, options)
        return []


class _Collect(ResumeCheck):
    """BatchedInferencePipeline.forward on a batch the stream decoded: outputs as generate_segment_batched's."""
    outputs = None

    def generate_segment_batched(self, features, tokenizer, options):
        return None, self._keep_tokens(self.outputs)


class StreamEngine:
    def __init__(self, model, pool, done, log, batch_size=8):
        self.model, self.pool, self.done, self.log, self.batch_size = model, pool, done, log, batch_size
        self.submitter, self.collector = _Submit(model, self._submit), _Collect(model)
        self.stream, self.key, self.thread, self.unit = None, None, None, None
        self.tags, self.next_tag, self.lock = {}, 0, threading.Lock()

    def add_unit(self, uid, clips):
        good = [(u, w) for u, w in clips if w is not None and len(w) > 0]
        bad = [{"uuid": u, "dur_s": 0.0, "text": None, "error": "decode"} for u, w in clips if w is None or len(w) == 0]
        if not good:
            self.done.put((uid, bad))
            return
        order = engine.length_order(good)
        ordered = [good[i] for i in order]
        audio, ts = engine.unit_audio(self.model, ordered)
        self.unit = _Unit(uid, ordered, order, bad, ts)
        self.submitter.transcribe(audio, batch_size=self.batch_size, clip_timestamps=ts, **engine.EXACT)
        self.unit = None

    def close(self):
        """Once every unit was added: waits until all their rows are with the writer."""
        if self.stream is not None:
            self.stream.close()
            self.thread.join()

    def _submit(self, features, tokenizer, chunks_metadata, batch_size, options):
        unit = self.unit
        unit.tokenizer, unit.metadata, unit.options, unit.size = tokenizer, chunks_metadata, options, batch_size
        # generate_segment_batched's prompt and length (faster-whisper 1.2.1)
        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(tokenizer.encode(options.initial_prompt) if options.initial_prompt is not None else []),
            without_timestamps=options.without_timestamps, hotwords=options.hotwords)
        max_length = (len(prompt) + options.max_new_tokens if options.max_new_tokens is not None
                      else self.model.max_length)
        self._open(options, max_length)
        starts = list(range(0, len(features), batch_size))
        unit.left = len(starts)
        for k, i in enumerate(starts):
            enc = self.model.encode(features[i:i + batch_size])          # WhisperModel.encode, as the pipeline's
            with self.lock:
                tag, self.next_tag = self.next_tag, self.next_tag + 1
                self.tags[tag] = (unit, k, i)
            self.stream.submit(tag, enc, [list(prompt) for _ in range(len(features[i:i + batch_size]))])

    def _open(self, options, max_length):
        """The stream, with generate_segment_batched's generate() options (the same for every unit)."""
        kw = dict(beam_size=options.beam_size, patience=options.patience, length_penalty=options.length_penalty,
                  max_length=max_length, suppress_blank=options.suppress_blank,
                  suppress_tokens=list(options.suppress_tokens), return_scores=True, return_no_speech_prob=True,
                  repetition_penalty=options.repetition_penalty, no_repeat_ngram_size=options.no_repeat_ngram_size)
        key = (sorted(kw.items(), key=lambda kv: kv[0]), options.temperatures[0], options.multilingual)
        if self.stream is None:
            if options.temperatures[0] != 0 or options.multilingual or options.word_timestamps:
                raise ValueError("The stream decodes the batches' temperature-0 beam search, one language")
            self.stream = self.model.model.open_stream(
                max_batches=int(os.environ.get("STREAM_BATCHES", "8")),
                max_rows=int(os.environ.get("STREAM_ROWS", "320")),
                max_pending=int(os.environ.get("STREAM_PENDING", "2")), **kw)
            self.key = key
            self.thread = threading.Thread(target=self._collect, daemon=True)
            self.thread.start()
        elif key != self.key:
            raise ValueError("A unit's decoding options differ from the stream's")

    def _collect(self):
        try:
            while (item := self.stream.next()) is not None:
                tag, results = item
                with self.lock:
                    unit, k, i = self.tags.pop(tag)
                c = self.collector
                c.outputs = generate_outputs(results, unit.options.length_penalty)
                c.unfinished.clear()
                unit.batches[k] = c.forward(None, unit.tokenizer, unit.metadata[i:i + unit.size], unit.options)
                unit.unfinished += c.unfinished
                unit.left -= 1
                if unit.left == 0:
                    self._finish(unit)
        except Exception as e:                               # never leave the writer waiting on a lost unit
            self.log(f"STREAM COLLECTOR CRASHED: {type(e).__name__}: {e}")
            os._exit(4)

    def _finish(self, unit):
        """_batched_segments_generator's Segments in batch order, then the unit's rows (as engine._batch8's)."""
        segs, seg_idx = [], 0
        for k in range(len(unit.batches)):
            for result in unit.batches[k]:
                for segment in result:
                    seg_idx += 1
                    segs.append(make_segment(segment, seg_idx, unit.options))
        rows = engine.unit_rows(self.model, unit.clips, unit.ts, segs, unit.unfinished, self.pool)
        self.done.put((unit.uid, engine.unsort(unit.order, rows) + unit.bad))
