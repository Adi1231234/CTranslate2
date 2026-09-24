"""Shared setup for the Turing tools: DLL search path, the model, the fixed clip batches, beam-5 generate."""
import os, sys, json, glob, sysconfig


def init(pkg_parent=None):
    """Put a ctranslate2 build first on sys.path and the nvidia-* wheel DLLs on the search path.
    CTranslate2 loads cuBLAS with a plain LoadLibrary, which searches PATH and ignores
    os.add_dll_directory, so both are needed."""
    if pkg_parent:
        sys.path.insert(0, pkg_parent)
    # The venv's site-packages, also when a profiler runs the base interpreter with it on PYTHONPATH.
    roots = {sysconfig.get_paths()["purelib"]} | {p for p in sys.path if p.endswith("site-packages")}
    for d in sorted({d for r in roots for d in glob.glob(os.path.join(r, "nvidia", "*", "bin"))}):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


RELEASE_THRESHOLD, RESERVED_HIGH, USED_HIGH = 4, 6, 8     # CUmemPool_attribute


def mempool(device=0):
    """attr(a) reads and attr(a, v) sets an attribute of the device's default CUDA memory pool (the
    pool CTranslate2's cuda_malloc_async allocator draws from), through the driver API."""
    import ctypes
    cu = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    dev, pool = ctypes.c_int(), ctypes.c_void_p()
    assert cu.cuInit(0) == 0 and cu.cuDeviceGet(ctypes.byref(dev), device) == 0
    assert cu.cuDeviceGetDefaultMemPool(ctypes.byref(pool), dev) == 0

    def attr(a, value=None):
        v = ctypes.c_uint64(0 if value is None else value)
        f = cu.cuMemPoolGetAttribute if value is None else cu.cuMemPoolSetAttribute
        assert f(pool, a, ctypes.byref(v)) == 0
        return v.value
    return attr


def sample_waves(sample):
    """The 150 sample clips (first occurrence of each key), shortest first."""
    import numpy as np
    meta, seen = [], set()
    for m in json.load(open(os.path.join(sample, "meta.json"), encoding="utf-8")):
        if m["key"] not in seen:
            seen.add(m["key"]); meta.append(m)
    return sorted([np.load(os.path.join(sample, x["key"] + ".npy")) for x in meta[:150]], key=len)


def whisper(num_workers=1, cpu_threads=0):
    """The production model on the GPU, its Hebrew tokenizer and the production prompt."""
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer
    model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                         num_workers=num_workers, cpu_threads=cpu_threads)
    tk = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
    return model, tk, model.get_prompt(tk, [], without_timestamps=False)


def features(model, waves):
    """A batch of padded log-mel features, as the batched pipeline computes them."""
    import numpy as np
    from faster_whisper.audio import pad_or_trim
    return np.stack([pad_or_trim(model.feature_extractor(w)[..., :-1]) for w in waves])


def load(sample, batches=4, num_workers=1, first=60):
    """Model, `batches` feature batches of 8 real clips (sorted by length, from index `first`), and a
    beam-5 generate with the production decode parameters."""
    model, _, prompt = whisper(num_workers)
    ws = sample_waves(sample)
    feats = [features(model, ws[i:i + 8]) for i in range(first, first + 8 * batches, 8)]
    gen = lambda e: model.model.generate(e, [prompt] * e.shape[0], beam_size=5, patience=1,
                                         length_penalty=1, max_length=448, suppress_blank=True,
                                         suppress_tokens=[-1], return_scores=True,
                                         return_no_speech_prob=True)
    return model, feats, gen
