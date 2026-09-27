"""Records every call of faster-whisper's WhisperModel._split_segments_by_timestamps: one decoded 30 s window's
tokens, and the frame the next window should start from. The batched pipeline calls it once per clip from
forward(), which never uses that frame; the sequential path once per window from generate_segments(), which
decodes again from there. When the tokens do not end in a single timestamp, the function drops the segment
after the last timestamp pair (faster-whisper 1.2.1: "ignore the unfinished segment and seek to the last
timestamp"): `dropped` is that text. Importing this module installs the recorder; `calls` holds the records."""
import sys
from faster_whisper import WhisperModel

calls = []
_split = WhisperModel._split_segments_by_timestamps


def render(tokenizer, tokens):
    """Tokens as text with each timestamp token written <|seconds|>."""
    tb, out, run = tokenizer.timestamp_begin, [], []
    for t in list(tokens) + [None]:
        if t is not None and t < tb:
            run.append(t)
            continue
        if run:
            out.append(tokenizer.decode(run)); run = []
        if t is not None:
            out.append(f"<|{(t - tb) * 0.02:.2f}|>")
    return "".join(out)


def _recorded(self, tokenizer, tokens, time_offset, segment_size, segment_duration, seek):
    result = _split(self, tokenizer=tokenizer, tokens=tokens, time_offset=time_offset, segment_size=segment_size,
                    segment_duration=segment_duration, seek=seek)
    _, seek_out, single = result
    tb = tokenizer.timestamp_begin
    pairs = [i for i in range(1, len(tokens)) if tokens[i] >= tb and tokens[i - 1] >= tb]
    dropped = tokens[pairs[-1]:] if pairs and not single else []
    calls.append({"caller": sys._getframe(1).f_code.co_name, "offset": round(time_offset, 3),
                  "duration": segment_duration, "seek_in": seek, "seek_out": seek_out, "size": segment_size,
                  "single_ending": single, "n_tokens": len(tokens), "tokens": render(tokenizer, tokens),
                  "dropped": render(tokenizer, dropped)})
    return result


WhisperModel._split_segments_by_timestamps = _recorded
