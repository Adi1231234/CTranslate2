"""Whisper throughput + output identity for a CTranslate2 build.
usage: bench_whisper.py <sample_dir> [ctranslate2 package parent dir]
Encodes and beam-5 decodes 4 batches of 8 real clips (fixed selection) and prints encoder time E,
decoder time D and a hash of every decoded token + score: the hash must equal the stock wheel's
for a change to count as output-preserving."""
import os, sys, json, time, hashlib
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer

sample = sys.argv[1]
meta, seen = [], set()
for m in json.load(open(os.path.join(sample, "meta.json"), encoding="utf-8")):
    if m["key"] not in seen:
        seen.add(m["key"]); meta.append(m)
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default")
tk = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
prompt = model.get_prompt(tk, [], without_timestamps=False)
ws = sorted([np.load(os.path.join(sample, x["key"] + ".npy")) for x in meta[:150]], key=len)
feats = [np.stack([pad_or_trim(model.feature_extractor(w)[..., :-1]) for w in ws[i:i + 8]]) for i in range(60, 92, 8)]
gen = lambda e: model.model.generate(e, [prompt] * e.shape[0], beam_size=5, patience=1, length_penalty=1,
                                     max_length=448, suppress_blank=True, suppress_tokens=[-1],
                                     return_scores=True, return_no_speech_prob=True)
gen(model.encode(feats[0]))                                      # warmup
t = time.time(); encs = [model.encode(f) for f in feats]; E = time.time() - t
digest = hashlib.sha256()
t = time.time()
for e in encs:
    for r in gen(e):
        digest.update(json.dumps([r.sequences_ids[0], round(r.scores[0], 4)]).encode())
D = time.time() - t
print(json.dumps({"ctranslate2": ctranslate2.__file__, "legacy_softmax": os.environ.get("CT2_CUDA_LEGACY_SOFTMAX", "0"),
                  "E": round(E, 2), "D": round(D, 2), "tokens_sha": digest.hexdigest()[:16]}), flush=True)
