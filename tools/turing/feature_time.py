"""How long the batched pipeline's up-front feature extraction takes (faster-whisper
BatchedInferencePipeline.transcribe computes the log-mel of every clip of a unit, one after the other,
before any GPU work of that unit), serially and on a thread pool.
usage: feature_time.py <sample_dir>"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor

S = sys.argv[1]
meta, seen = [], set()
for m in json.load(open(os.path.join(S, "meta.json"), encoding="utf-8")):
    if m["key"] not in seen:
        seen.add(m["key"]); meta.append(m)
clips = [np.load(os.path.join(S, m["key"] + ".npy")) for m in meta[:150]]
fe = FeatureExtractor(feature_size=128)                  # large-v3
extract = lambda w: fe(w)[..., :-1]                      # as BatchedInferencePipeline.transcribe
extract(clips[0])
t = time.time(); serial = [extract(w) for w in clips]; ts = time.time() - t
for n in (2, 4, 6):
    with ThreadPoolExecutor(n) as pool:
        t = time.time(); par = list(pool.map(extract, clips)); tp = time.time() - t
    same = all(np.array_equal(a, b) for a, b in zip(serial, par))
    print(json.dumps({"threads": n, "seconds": round(tp, 2), "identical": same}))
print(json.dumps({"clips": len(clips), "audio_s": round(sum(len(w) for w in clips) / 16000, 1),
                  "serial_s": round(ts, 2), "ms_per_clip": round(1000 * ts / len(clips), 1)}))
