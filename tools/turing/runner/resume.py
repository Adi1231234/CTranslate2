"""Which clips a batched pass left unfinished. faster-whisper's BatchedInferencePipeline.forward (1.2.1 and master)
splits each clip's tokens with _split_segments_by_timestamps and ignores the frame it returns: when the tokens do not
end in a single timestamp, the segment after the last timestamp pair is dropped and the rest of the clip is never
decoded, while the sequential path (generate_segments) decodes another window from that frame. ResumeCheck records
each clip whose returned frame lies before the clip's end, so the engine can send it to the sequential path.
A subclass that replaces generate_segment_batched hands its outputs to _keep_tokens."""
from math import ceil
from faster_whisper import BatchedInferencePipeline


class ResumeCheck(BatchedInferencePipeline):
    def __init__(self, model):
        super().__init__(model)
        self.unfinished = []                            # offsets (s) of the clips left unfinished; engine clears

    def _keep_tokens(self, outputs):
        self._tokens = [o["tokens"] for o in outputs]
        return outputs

    def generate_segment_batched(self, features, tokenizer, options):
        encoder_output, outputs = super().generate_segment_batched(features, tokenizer, options)
        return encoder_output, self._keep_tokens(outputs)

    def forward(self, features, tokenizer, chunks_metadata, options):
        result = super().forward(features, tokenizer, chunks_metadata, options)
        self._check(tokenizer, chunks_metadata)
        return result

    def _check(self, tokenizer, chunks_metadata):
        """Not in forward(): scale/split_recorder.py tells the library's own calls apart by the caller's name."""
        for meta, tokens in zip(chunks_metadata, self._tokens):
            size = int(ceil(meta["duration"]) * self.model.frames_per_second)     # as forward() computes it
            _, seek, _ = self.model._split_segments_by_timestamps(
                tokenizer=tokenizer, tokens=tokens, time_offset=meta["offset"], segment_size=size,
                segment_duration=meta["duration"], seek=0)
            if seek < size:
                self.unfinished.append(meta["offset"])
