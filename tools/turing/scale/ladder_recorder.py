"""Records every attempt of faster-whisper's temperature ladder (WhisperModel.generate_with_fallback, 1.2.1): for
each window, one record per generate() call with its temperature, the prompt length, every hypothesis's token
count, the kept hypothesis's tokens, compression ratio, avg log-prob and no-speech probability (computed as the
library does), whether it passes the thresholds, and the call's wall time; then which attempt the ladder returned.
Importing this module installs the recorder; `calls` holds one record per window. It swaps the model object for
the length of a call, so it is for one decoding thread only (seeded.py: one worker)."""
import time
from faster_whisper import WhisperModel
from faster_whisper.transcribe import get_compression_ratio

calls = []
_ladder = WhisperModel.generate_with_fallback


class _Recording:
    """Stands in for the CTranslate2 Whisper model during one ladder, timing and scoring each generate()."""

    def __init__(self, model, tokenizer, options, prompt_len, attempts):
        self._model, self._tok, self._opt, self._plen, self._attempts = model, tokenizer, options, prompt_len, attempts

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate(self, *args, **kwargs):
        t = time.perf_counter()
        results = self._model.generate(*args, **kwargs)
        secs = time.perf_counter() - t
        r, o = results[0], self._opt
        tokens = r.sequences_ids[0]
        avg_lp = r.scores[0] * (len(tokens) ** o.length_penalty) / (len(tokens) + 1)
        cr = get_compression_ratio(self._tok.decode(tokens).strip())
        low_lp = o.log_prob_threshold is not None and avg_lp < o.log_prob_threshold
        fails = (o.compression_ratio_threshold is not None and cr > o.compression_ratio_threshold) or low_lp
        silence = o.no_speech_threshold is not None and r.no_speech_prob > o.no_speech_threshold and low_lp
        self._attempts.append({"T": kwargs.get("sampling_temperature", 0), "prompt": self._plen,
                               "max_length": kwargs["max_length"], "n": [len(s) for s in r.sequences_ids],
                               "cr": cr, "avg_lp": avg_lp, "nsp": r.no_speech_prob, "passes": silence or not fails,
                               "secs": round(secs, 4), "tokens": list(tokens)})
        return results


def _recorded(self, encoder_output, prompt, tokenizer, options):
    attempts, real = [], self.model
    self.model = _Recording(real, tokenizer, options, len(prompt), attempts)
    try:
        result, avg_lp, temperature, cr = _ladder(self, encoder_output, prompt, tokenizer, options)
    finally:
        self.model = real
    calls.append({"attempts": attempts, "returned": {"avg_lp": avg_lp, "T": temperature, "cr": cr,
                                                     "tokens": list(result.sequences_ids[0])}})
    return result, avg_lp, temperature, cr


WhisperModel.generate_with_fallback = _recorded
