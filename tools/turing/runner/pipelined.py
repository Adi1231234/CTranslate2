"""BatchedInferencePipeline with the encoder of batch i+1 overlapped with the decoder of batch i.

Per profile, ~31% of wall time is the encoder and ~66% the beam-search decoder, run back to back.
The decoder leaves the GPU partly idle (many small steps), so a background thread encodes the next
batch on a second CTranslate2 worker (model needs num_workers=2) while the main thread decodes.
Per-batch math is unchanged: same batches, same encoder call, same generate() call.
PIPE_ORDER=desc (default) decodes the batches last to first (the longest clips first: while their long decodes
run, the encoder banks later batches, which then decode without waiting; 0.1-0.2 s faster than asc in 3 of 3
alternated pairs, 26.9); PIPE_ORDER=asc keeps the batch order; PIPE_ORDER=interleave alternates
the longest and the shortest left (a long decode, during which the encoder gets ahead, then a short one,
which would otherwise wait for it); the segments still come out in batch order. PIPE_AHEAD=<n>: encoded batches that may wait (default 1). PIPE_LOG=<file>: one line
per batch with its encode and decode start and end, then its generate() call's start and end (seconds).
PIPE_GROUPS=<k> (default 1): k consecutive batches go through one encode() and one generate() call with
group_size = the batch size, so CTranslate2 runs each batch's products as that batch alone would (same
shapes, same kernels, same output) but back to back, and the decoder weights are read once for all k.
"""
import os, queue, threading, time
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, Word, get_ctranslate2_storage
from tqdm import tqdm
from resume import ResumeCheck


def make_segment(segment, seg_idx, options):
    """_batched_segments_generator's Segment of one of forward()'s segments (faster-whisper 1.2.1)."""
    return Segment(
        seek=segment["seek"], id=seg_idx, text=segment["text"],
        start=round(segment["start"], 3), end=round(segment["end"], 3),
        words=None if not options.word_timestamps else [Word(**w) for w in segment["words"]],
        tokens=segment["tokens"], avg_logprob=segment["avg_logprob"],
        no_speech_prob=segment["no_speech_prob"], compression_ratio=segment["compression_ratio"],
        temperature=options.temperatures[0])


def generate_outputs(results, length_penalty):
    """generate_segment_batched's outputs of a batch's generate() results (faster-whisper 1.2.1)."""
    output = []
    for result in results:
        seq_len = len(result.sequences_ids[0])
        cum_logprob = result.scores[0] * (seq_len ** length_penalty)
        output.append(dict(avg_logprob=cum_logprob / (seq_len + 1), no_speech_prob=result.no_speech_prob,
                           tokens=result.sequences_ids[0]))
    return output


class PipelinedBatchedInferencePipeline(ResumeCheck):
    def _encode(self, features, group_size):
        """WhisperModel.encode (faster-whisper 1.2.1) with each group of the batch encoded on its own."""
        whisper = self.model.model                            # the ctranslate2 model
        to_cpu = whisper.device == "cuda" and len(whisper.device_index) > 1
        return whisper.encode(get_ctranslate2_storage(features), to_cpu=to_cpu, group_size=group_size)

    def _batched_segments_generator(self, features, tokenizer, chunks_metadata, batch_size, options, log_progress):
        groups = int(os.environ.get("PIPE_GROUPS", "1"))
        self._group_size = batch_size if groups > 1 else 0
        batch_size *= groups                                 # whole batches of the ungrouped pipeline
        starts = list(range(0, len(features), batch_size))
        mode = os.environ.get("PIPE_ORDER", "desc")
        if mode == "desc":
            order = starts[::-1]
        elif mode == "interleave":                            # longest, shortest, 2nd longest, 2nd shortest, ...
            order = [starts[-1 - k // 2] if k % 2 == 0 else starts[k // 2] for k in range(len(starts))]
        else:
            order = starts
        ahead = queue.Queue(maxsize=int(os.environ.get("PIPE_AHEAD", "1")))
        times = {i: [] for i in starts}                      # encode start, end, decode start, end

        def encode_ahead():
            try:
                for i in order:
                    times[i].append(time.perf_counter())
                    enc = (self._encode(features[i:i + batch_size], self._group_size) if self._group_size
                           else WhisperModel.encode(self.model, features[i:i + batch_size]))
                    times[i].append(time.perf_counter())
                    ahead.put((i, enc))
            except Exception as e:                           # surface encoder errors in the caller
                ahead.put((None, e))

        threading.Thread(target=encode_ahead, daemon=True).start()
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        done = {}
        for i in order:
            j, enc = ahead.get()
            if j is None:
                raise enc
            self._encoder_output = enc
            self._generate_times = []
            times[i].append(time.perf_counter())
            done[i] = self.forward(features[i:i + batch_size], tokenizer, chunks_metadata[i:i + batch_size], options)
            times[i].append(time.perf_counter())
            times[i] += self._generate_times                # generate() call start and end
        if os.environ.get("PIPE_LOG"):
            with open(os.environ["PIPE_LOG"], "a") as f:
                t0 = min(t[0] for t in times.values())
                for i in order:
                    f.write(" ".join([str(i)] + [f"{t - t0:.3f}" for t in times[i]]) + "\n")
        seg_idx = 0
        for i in starts:
            for result in done[i]:
                for segment in result:
                    seg_idx += 1
                    yield make_segment(segment, seg_idx, options)
                pbar.update(1)
        pbar.close()
        self.last_speech_timestamp = 0.0

    def generate_segment_batched(self, features, tokenizer, options):
        """Library body (faster-whisper 1.2.1) with encode() replaced by the pre-computed output."""
        batch_size = features.shape[0]
        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(tokenizer.encode(options.initial_prompt) if options.initial_prompt is not None else []),
            without_timestamps=options.without_timestamps, hotwords=options.hotwords)
        max_length = (len(prompt) + options.max_new_tokens if options.max_new_tokens is not None
                      else self.model.max_length)
        if max_length > self.model.max_length:
            raise ValueError(f"prompt + max_new_tokens = {max_length} exceeds {self.model.max_length}")
        encoder_output = self._encoder_output
        prompts = [prompt.copy() for _ in range(batch_size)]
        if options.multilingual:
            language_tokens = [tokenizer.tokenizer.token_to_id(seg[0][0])
                               for seg in self.model.model.detect_language(encoder_output)]
            idx = prompt.index(tokenizer.language)
            for k, tok in enumerate(language_tokens):
                prompts[k][idx] = tok
        marks = getattr(self, "_generate_times", [])
        marks.append(time.perf_counter())
        results = self.model.model.generate(
            encoder_output, prompts, beam_size=options.beam_size, patience=options.patience,
            length_penalty=options.length_penalty, max_length=max_length,
            suppress_blank=options.suppress_blank, suppress_tokens=options.suppress_tokens,
            return_scores=True, return_no_speech_prob=True, sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty, no_repeat_ngram_size=options.no_repeat_ngram_size,
            **({"group_size": self._group_size} if getattr(self, "_group_size", 0) else {}))
        marks.append(time.perf_counter())
        return encoder_output, self._keep_tokens(generate_outputs(results, options.length_penalty))
