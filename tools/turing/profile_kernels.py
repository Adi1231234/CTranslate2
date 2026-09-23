"""Kernel-level profile of CTranslate2 Whisper on this GPU, from CUPTI activity records (ctypes).
Phase A: encoder of one 8-clip batch, then its beam-5 decode. Phase B: next batch's encoder on a
second worker while batch 0 decodes. Per phase: wall, GPU-busy (union of kernel intervals), top
kernels, and how long kernels of different streams actually ran at the same time. -> prof_ct2.log"""
import os, sys, time, ctypes, zipfile, urllib.request, collections, threading
from ctypes import CFUNCTYPE, POINTER, byref, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p
# usage: profile_kernels.py <sample_dir> <out_log> [ctranslate2 package parent dir]
SAMPLE, OUT = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    sys.path.insert(0, sys.argv[3])
import glob, sysconfig
for d in glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "bin")):
    os.add_dll_directory(d)
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
ROOT = os.path.dirname(OUT)

WHL = ("https://files.pythonhosted.org/packages/1c/81/7796f096afaf726796b1b648f3bc80cafc61fe7f77f44a483c89e6c5ef34/"
       "nvidia_cuda_cupti_cu12-12.6.80-py3-none-win_amd64.whl")
DLL_DIR = os.path.join(ROOT, "cupti")
DLL = os.path.join(DLL_DIR, "cupti64_2024.3.2.dll")
if not os.path.exists(DLL):
    os.makedirs(DLL_DIR, exist_ok=True)
    whl = os.path.join(DLL_DIR, "cupti.whl")
    urllib.request.urlretrieve(WHL, whl)
    with zipfile.ZipFile(whl) as z:
        for n in z.namelist():
            if n.endswith(".dll"):
                open(os.path.join(DLL_DIR, os.path.basename(n)), "wb").write(z.read(n))
os.add_dll_directory(DLL_DIR)
cupti = ctypes.CDLL(DLL)

KIND_CONCURRENT_KERNEL, BUF = 10, 8 << 20
bufs, recs = {}, []                     # records: (name, stream, start_ns, end_ns)
REQ = CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(c_size_t))
DONE = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(c_uint8), c_size_t, c_size_t)


def _requested(buf_pp, size_p, maxrec_p):
    b = (c_uint64 * (BUF // 8))()                               # 8-byte aligned, as CUPTI requires
    bufs[ctypes.addressof(b)] = b
    buf_pp[0] = ctypes.cast(b, POINTER(c_uint8)); size_p[0] = BUF; maxrec_p[0] = 0


def _completed(ctx, stream, buf, size, valid):
    rec = c_void_p()
    while cupti.cuptiActivityGetNextRecord(buf, c_size_t(valid), byref(rec)) == 0:
        a = rec.value                                           # CUpti_ActivityKernel9 offsets
        if c_uint32.from_address(a).value == KIND_CONCURRENT_KERNEL:
            name = ctypes.string_at(c_void_p.from_address(a + 104).value).decode(errors="replace")
            recs.append((name, c_uint32.from_address(a + 48).value,
                         c_uint64.from_address(a + 16).value, c_uint64.from_address(a + 24).value))
    bufs.pop(ctypes.addressof(buf.contents), None)


req_cb, done_cb = REQ(_requested), DONE(_completed)
assert cupti.cuptiActivityRegisterCallbacks(req_cb, done_cb) == 0


def now():
    t = c_uint64(); cupti.cuptiGetTimestamp(byref(t)); return t.value


import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
import json
meta, seen = [], set()
for x in json.load(open(os.path.join(SAMPLE, "meta.json"), encoding="utf-8")):
    if x["key"] not in seen:
        seen.add(x["key"]); meta.append(x)
m = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default", num_workers=2)
tk = Tokenizer(m.hf_tokenizer, True, task="transcribe", language="he")
prompt = m.get_prompt(tk, [], without_timestamps=False)
ws = sorted([np.load(os.path.join(SAMPLE, x["key"] + ".npy")) for x in meta[:150]], key=len)
f0, f1 = [np.stack([pad_or_trim(m.feature_extractor(w)[..., :-1]) for w in ws[i:i + 8]]) for i in (60, 68)]
gen = lambda e: m.model.generate(e, [prompt] * 8, beam_size=5, patience=1, length_penalty=1, max_length=448,
                                 suppress_blank=True, suppress_tokens=[-1], return_scores=True,
                                 return_no_speech_prob=True)
gen(m.encode(f0)); gen(m.encode(f1))                            # warmup, outside the trace
assert cupti.cuptiActivityEnable(KIND_CONCURRENT_KERNEL) == 0
t0 = now(); e0 = m.encode(f0); t1 = now(); out = gen(e0); t2 = now()
steps = max(len(r.sequences_ids[0]) for r in out) + 1
th = threading.Thread(target=m.encode, args=(f1,)); t3 = now(); th.start(); gen(e0); th.join(); t4 = now()
cupti.cuptiActivityFlushAll(1)


def busy(iv):
    tot, end = 0, -1
    for s, e in sorted(iv):
        if s > end:
            tot += e - s; end = e
        elif e > end:
            tot += e - end; end = e
    return tot


def short(n):
    n = n.replace("void ", "")
    return (n.split("(")[0].split("<")[0] + ("<" + n.split("<")[1][:40] if "<" in n else ""))[:90]


lines = []
for tag, a, b in (("encoder_bs8", t0, t1), ("decode_bs8_beam5", t1, t2), ("enc_next||decode", t3, t4)):
    rs = [r for r in recs if a <= r[2] <= b]
    tot = collections.Counter(); cnt = collections.Counter()
    for n, s, st, en in rs:
        tot[short(n)] += en - st; cnt[short(n)] += 1
    wall, sumk = (b - a) / 1e6, sum(en - st for _, _, st, en in rs) / 1e6
    streams = sorted(set(r[1] for r in rs))
    per = {s: busy([(st, en) for _, ss, st, en in rs if ss == s]) / 1e6 for s in streams}
    lines.append(f"== {tag}: wall {wall:.1f} ms | kernels {len(rs)} | kernel-time {sumk:.1f} ms | "
                 f"GPU busy {busy([(st, en) for _, _, st, en in rs]) / 1e6:.1f} ms | per-stream busy "
                 + ", ".join(f"s{s}:{v:.1f}" for s, v in per.items())
                 + f" | streams overlapped {sum(per.values()) - busy([(st, en) for _, _, st, en in rs]) / 1e6:.1f} ms"
                 + (f" | decode steps {steps}, kernels/step {len(rs) / steps:.0f}" if tag.startswith("decode") else ""))
    for n, t in tot.most_common(14):
        lines.append(f"   {t / 1e6:8.1f} ms {100 * t / max(1, sum(tot.values())):5.1f}% x{cnt[n]:5d}  {n}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
del m, e0, out                                  # release the model's worker threads while Python is alive
import gc; gc.collect()
