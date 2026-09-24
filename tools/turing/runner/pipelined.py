"""BatchedInferencePipeline with the encoder of batch i+1 overlapped with the decoder of batch i.

Per profile, ~31% of wall time is the encoder and ~66% the beam-search decoder, run back to back.
The decoder leaves the GPU partly idle (many small steps), so a background thread encodes the next
batch on a second CTranslate2 worker (model needs num_workers=2) while the main thread decodes.
Per-batch math is unchanged: same batches, same encoder call, same generate() call.
"""
import queue, threading
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import Segment, Word
from tqdm import tqdm


class PipelinedBatchedInferencePipeline(BatchedInferencePipeline):
    def _batched_segments_generator(self, features, tokenizer, chunks_metadata, batch_size, options, log_progress):
        ahead = queue.Queue(maxsize=1)                       # at most one pre-encoded batch waiting

        def encode_ahead():
            try:
                for i in range(0, len(features), batch_size):
                    ahead.put((i, WhisperModel.encode(self.model, features[i:i + batch_size])))
            except Exception as e:                           # surface encoder errors in the caller
                ahead.put((None, e))

        threading.Thread(target=encode_ahead, daemon=True).start()
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0
        for i in range(0, len(features), batch_size):
            j, enc = ahead.get()
            if j is None:
                raise enc
            self._encoder_output = enc
            results = self.forward(features[i:i + batch_size], tokenizer, chunks_metadata[i:i + batch_size], options)
            for result in results:
                for segment in result:
                    seg_idx += 1
                    yield Segment(
                        seek=segment["seek"], id=seg_idx, text=segment["text"],
                        start=round(segment["start"], 3), end=round(segment["end"], 3),
                        words=None if not options.word_timestamps else [Word(**w) for w in segment["words"]],
                        tokens=segment["tokens"], avg_logprob=segment["avg_logprob"],
                        no_speech_prob=segment["no_speech_prob"], compression_ratio=segment["compression_ratio"],
                        temperature=options.temperatures[0])
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
        results = self.model.model.generate(
            encoder_output, prompts, beam_size=options.beam_size, patience=options.patience,
            length_penalty=options.length_penalty, max_length=max_length,
            suppress_blank=options.suppress_blank, suppress_tokens=options.suppress_tokens,
            return_scores=True, return_no_speech_prob=True, sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty, no_repeat_ngram_size=options.no_repeat_ngram_size)
        output = []
        for result in results:
            seq_len = len(result.sequences_ids[0])
            cum_logprob = result.scores[0] * (seq_len ** options.length_penalty)
            output.append(dict(avg_logprob=cum_logprob / (seq_len + 1), no_speech_prob=result.no_speech_prob,
                               tokens=result.sequences_ids[0]))
        return encoder_output, output
