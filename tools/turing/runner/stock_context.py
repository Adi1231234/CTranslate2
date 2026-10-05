"""RUN_STOCK_FULL_CONTEXT=1 (verification only, the stock wheel): stock CTranslate2 decodes as many tokens as the fork.
Stock decodes min(max_length / 2, max_length - start_step) tokens after the prompt (src/models/whisper.cc of 4.8.2),
the fork max_length - start_step (upstream PR #2075, the fork's one intended change of a result). faster-whisper
passes max_length 448, so each generate() call gets the max_length that stock's own rule turns into 448 - start_step:
2 * (448 - start_step) for a start_step up to 224, else 448. The model, the arguments and the code are stock's."""
import os
import ctranslate2


def start_step(prompt, sot_id, no_timestamps_id):
    """CTranslate2's start_step: get_prompt_length (the index past <|startoftranscript|> and the task tokens after
    it) less one, or 0 for a prompt of one token (WhisperReplica::generate)."""
    i = prompt.index(sot_id)
    while i < len(prompt) and sot_id <= prompt[i] <= no_timestamps_id:
        i += 1
    return i - 1 if i > 1 else 0


def stock_max_length(max_length, step):
    """The max_length m for which stock's min(m / 2, m - step) is max_length - step."""
    return 2 * (max_length - step) if 2 * step <= max_length else max_length


class FullContext:
    """The ctranslate2 Whisper model with generate()'s max_length mapped by stock_max_length; all else passes."""

    def __init__(self, whisper, sot_id, no_timestamps_id):
        self._whisper, self._ids = whisper, (sot_id, no_timestamps_id)

    def __getattr__(self, name):
        return getattr(self._whisper, name)

    def generate(self, features, prompts, *args, max_length=448, **kwargs):
        steps = {start_step(list(p), *self._ids) for p in prompts}
        if len(steps) != 1:                        # CTranslate2 itself refuses such a batch (check_prompts)
            raise ValueError("the prompts of one call must have the same start step")
        return self._whisper.generate(features, prompts, *args,
                                      max_length=stock_max_length(max_length, steps.pop()), **kwargs)


def is_fork():
    """The fork's packages carry ctranslate2/BUILD.txt (../linux/build.sh); the PyPI wheel does not."""
    return os.path.exists(os.path.join(os.path.dirname(ctranslate2.__file__), "BUILD.txt"))


def full_context(model):
    """FullContext around a faster-whisper WhisperModel's ctranslate2 model (the stock wheel only)."""
    if is_fork():
        raise SystemExit("RUN_STOCK_FULL_CONTEXT is for the stock wheel: the fork decodes the full context itself")
    tok = model.hf_tokenizer
    model.model = FullContext(model.model, tok.token_to_id("<|startoftranscript|>"),
                              tok.token_to_id("<|notimestamps|>"))
