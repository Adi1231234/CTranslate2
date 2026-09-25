"""Transcribe one unit's clips. exact2 = 2 in-flight model.transcribe calls (byte-identical to
sequential). batch8 = cross-clip batched T=0 + sequential full-ladder fallback (hybrid)."""
import queue, threading
import numpy as np
from faster_whisper import BatchedInferencePipeline
from features import feature_cache
try:
    from faster_whisper.transcribe import get_compression_ratio as gcr
except ImportError:
    from faster_whisper.utils import get_compression_ratio as gcr

EXACT = dict(language="he", beam_size=5, best_of=5, patience=1,
             temperature=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
             compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
             condition_on_previous_text=True, without_timestamps=False, vad_filter=False)
GAP, SR = 8000, 16000
_bp = {}

def _seg(s, offset=0.0):
    return {"start": round(s.start - offset, 3), "end": round(s.end - offset, 3), "text": s.text,
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio, "temperature": s.temperature}

def _row(uuid, wav, segs, path, offset=0.0):
    return {"uuid": uuid, "dur_s": len(wav) / SR, "text": " ".join(s.text for s in segs).strip(),
            "segments": [_seg(s, offset) for s in segs], "path": path}

def _sequential(model, uuid, wav):
    segs, _ = model.transcribe(wav, **EXACT)
    return _row(uuid, wav, list(segs), "seq")

def _fallback(model, uuid, wav):
    r = _sequential(model, uuid, wav); r["path"] = "fallback"
    return r

def _exact2(model, clips):
    out, q = {}, queue.Queue()
    for i, c in enumerate(clips): q.put((i, c))
    def work():
        while True:
            try: i, (uuid, wav) = q.get_nowait()
            except queue.Empty: return
            out[i] = _sequential(model, uuid, wav)
    ts = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return [out[i] for i in range(len(clips))]

def _batch8(model, clips, bs=8, pipelined=False, pool=None):
    """Clips are batched shortest-first (similar lengths decode in lockstep), then put back in order.
    With a pool, fallback clips (full temperature ladder) run there and the row holds a Future."""
    order = sorted(range(len(clips)), key=lambda i: len(clips[i][1]))
    rows = _batch8_ordered(model, [clips[i] for i in order], bs, pipelined, pool)
    out = [None] * len(clips)
    for i, r in zip(order, rows): out[i] = r
    return out

def _batch8_ordered(model, clips, bs, pipelined=False, pool=None):
    if (id(model), pipelined) not in _bp:
        cls = BatchedInferencePipeline
        if pipelined:
            from pipelined import PipelinedBatchedInferencePipeline as cls
        _bp[(id(model), pipelined)] = cls(model=model)
    bp = _bp[(id(model), pipelined)]
    pieces, ts, off = [], [], 0
    for _, w in clips:
        pieces += [w, np.zeros(GAP, "float32")]
        ts.append({"start": off / SR, "end": (off + len(w)) / SR}); off += len(w) + GAP
    audio = np.concatenate(pieces)
    # transcribe's own slices of the clips (int(seconds * rate)), so it gets their precomputed features
    feature_cache(model).prefetch([audio[int(t["start"] * SR):int(t["end"] * SR)] for t in ts])
    segs, _ = bp.transcribe(audio, batch_size=bs, clip_timestamps=ts, **EXACT)
    per = [[] for _ in clips]
    for s in segs:
        per[max(j for j, t in enumerate(ts) if s.start >= t["start"] - 1e-3)].append(s)
    rows = []
    for (uuid, w), t, ss in zip(clips, ts, per):
        text = " ".join(s.text for s in ss).strip()
        lp = min((s.avg_logprob for s in ss), default=-99.0)
        if not ss or gcr(text) > EXACT["compression_ratio_threshold"] or lp < EXACT["log_prob_threshold"]:
            rows.append(pool.submit(_fallback, model, uuid, w) if pool else _fallback(model, uuid, w))
        else:
            rows.append(_row(uuid, w, ss, "batch8", offset=t["start"]))
    return rows

def transcribe_unit(model, clips, mode, pool=None):
    good = [(u, w) for u, w in clips if w is not None and len(w) > 0]
    bad = [{"uuid": u, "dur_s": 0.0, "text": None, "error": "decode"} for u, w in clips if w is None or len(w) == 0]
    if mode == "exact2":
        rows = _exact2(model, good)
    elif mode.startswith("pipe"):                 # "pipe<N>": batch<N> with encoder/decoder overlap
        rows = _batch8(model, good, bs=int(mode[4:]), pipelined=True, pool=pool)
    else:                                         # "batch<N>[-fa]": sorted cross-clip batches of N
        rows = _batch8(model, good, bs=int(mode.split("-")[0][5:]), pool=pool)
    return rows + bad
