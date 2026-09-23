"""Whisper throughput + output identity for a CTranslate2 build.
usage: bench_whisper.py <sample_dir> [ctranslate2 package parent dir]   (BENCH_FIRST=<i>: clips from sorted index i)
Encodes and beam-5 decodes 4 batches of 8 real clips (fixed selection; 60 = short clips, 118 = the longest) and prints encoder time E,
decoder time D and a hash of every decoded token + score: the hash must equal the stock wheel's
for a change to count as output-preserving."""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
common.init(sys.argv[2] if len(sys.argv) > 2 else None)
import numpy as np
import ctranslate2
from faster_whisper.transcribe import get_ctranslate2_storage

model, feats, gen = common.load(sys.argv[1], batches=4, first=int(os.environ.get("BENCH_FIRST", "60")))
gen(model.encode(feats[0]))                                      # warmup
t = time.time(); encs = [model.encode(f) for f in feats]; E = time.time() - t
digest, full = hashlib.sha256(), hashlib.sha256()
max_tokens = 0
t = time.time()
for e in encs:
    for r in gen(e):
        max_tokens = max(max_tokens, len(r.sequences_ids[0]))
        digest.update(json.dumps([r.sequences_ids[0], round(r.scores[0], 4)]).encode())
        full.update(repr((r.sequences_ids, r.scores, r.no_speech_prob)).encode())   # every bit of it
D = time.time() - t
enc = hashlib.sha256()                  # bytes of every encoder output: any 1-ulp change shows here
for f in feats:
    enc.update(np.asarray(model.model.encode(get_ctranslate2_storage(f), to_cpu=True)).tobytes())
print(json.dumps({"ctranslate2": ctranslate2.__file__, "stock_kernels": os.environ.get("CT2_CUDA_STOCK_KERNELS", "0"), "first": os.environ.get("BENCH_FIRST", "60"),
                  "max_tokens": max_tokens,
                  "E": round(E, 2), "D": round(D, 2), "tokens_sha": digest.hexdigest()[:16],
                  "full_sha": full.hexdigest()[:16], "enc_sha": enc.hexdigest()[:16]}), flush=True)
del model, encs                                 # release the model's worker threads while Python is alive
import gc; gc.collect()
