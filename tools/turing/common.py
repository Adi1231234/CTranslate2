"""Shared setup for the Turing tools: DLL search path, the model, the fixed clip batches, beam-5 generate."""
import os, sys, json, glob, sysconfig


def init(pkg_parent=None):
    """Put a ctranslate2 build first on sys.path and the nvidia-* wheel DLLs on the search path.
    CTranslate2 loads cuBLAS with a plain LoadLibrary, which searches PATH and ignores
    os.add_dll_directory, so both are needed."""
    if pkg_parent:
        sys.path.insert(0, pkg_parent)
    for d in glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def load(sample, batches=4, num_workers=1, first=60):
    """Model, `batches` feature batches of 8 real clips (sorted by length, from index `first`), and a
    beam-5 generate with the production decode parameters."""
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    meta, seen = [], set()
    for m in json.load(open(os.path.join(sample, "meta.json"), encoding="utf-8")):
        if m["key"] not in seen:
            seen.add(m["key"]); meta.append(m)
    model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                         num_workers=num_workers)
    tk = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
    prompt = model.get_prompt(tk, [], without_timestamps=False)
    ws = sorted([np.load(os.path.join(sample, x["key"] + ".npy")) for x in meta[:150]], key=len)
    feats = [np.stack([pad_or_trim(model.feature_extractor(w)[..., :-1]) for w in ws[i:i + 8]])
             for i in range(first, first + 8 * batches, 8)]
    gen = lambda e: model.model.generate(e, [prompt] * e.shape[0], beam_size=5, patience=1,
                                         length_penalty=1, max_length=448, suppress_blank=True,
                                         suppress_tokens=[-1], return_scores=True,
                                         return_no_speech_prob=True)
    return model, feats, gen
