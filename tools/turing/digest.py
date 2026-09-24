"""Fast byte-exactness check of a build: one process, one model load, about a minute. Hashes every byte
the model returns on inputs chosen for coverage and compares each section with a golden file recorded
once from the stock wheel (the reference never has to run again):
  enc_long / enc_mid / enc_6 / enc_4 / enc_1  every encoder output byte, batch sizes of both paths
  beam_long / beam_mid / beam_6       beam 5 exactly as the batched path, all 5 hypotheses (tokens,
                                      full-precision scores, no-speech probability); natural ends, so
                                      the batch shrinks as clips finish
  beam_4_448                          the same on the 4 longest clips with the end token suppressed:
                                      448 steps, every self-attention width a transcription can reach
                                      (8 clips x 5 hypotheses x 448 steps fill the 8 GB of the 2080)
  beam_prompt_448                     batch 1 after a previous-text prompt, as the sequential path
  sample_1 / sample_1_448             the fallback path (T=0.2, best of 5), seeded, and every step's
                                      full-vocabulary logits
Also reports each section's peak memory in the CUDA pool (peak_mb, not part of the digest).
usage: digest.py <sample_dir> [ctranslate2 package parent] (--save <golden.json> | --golden <golden.json>)"""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

args = sys.argv[1:]
opt = lambda name: args[args.index(name) + 1] if name in args else None
save, golden = opt("--save"), opt("--golden")
pos = [a for a in args if not a.startswith("--") and a not in (save, golden)]
common.init(pos[1] if len(pos) > 1 else None)
import numpy as np
import ctranslate2
from faster_whisper.transcribe import get_ctranslate2_storage

ctranslate2.set_random_seed(1234)                  # the same sampling draws in every run
model, tk, prompt = common.whisper(num_workers=1, cpu_threads=1)
ws = common.sample_waves(pos[0])
t0 = time.time()
feats = {"long": common.features(model, ws[-8:]), "mid": common.features(model, ws[70:78])}
feats["6"], feats["4"], feats["1"] = feats["long"][:6], feats["long"][-4:], feats["long"][:1]
out, enc, peak, mem = {}, {}, {}, common.mempool()
for name, f in feats.items():
    cpu = model.model.encode(get_ctranslate2_storage(f), to_cpu=True)
    out[f"enc_{name}"] = hashlib.sha256(np.asarray(cpu).tobytes()).hexdigest()[:16]
    enc[name] = model.encode(f)


def run(name, e, prompts, forced=False, logits=False, **kw):
    """generate() with the production decode options, hashing every result byte into out[name]."""
    mem(common.USED_HIGH, 0)                       # resets the high-water mark to what is in use now
    res = model.model.generate(e, prompts, beam_size=kw.pop("beam_size", 5), patience=1, length_penalty=1,
                               max_length=448, suppress_blank=True, return_scores=True,
                               return_no_speech_prob=True, return_logits_vocab=logits,
                               suppress_tokens=[-1, tk.eot] if forced else [-1], **kw)
    h = hashlib.sha256()
    for r in res:
        h.update(repr((r.sequences_ids, r.scores, r.no_speech_prob)).encode())
        for steps in r.logits:
            for sv in steps:
                h.update(np.asarray(sv.to_device(ctranslate2.Device.cpu)).tobytes())
    out[name] = h.hexdigest()[:16]
    peak[name] = mem(common.USED_HIGH) >> 20
    return res


batched = dict(num_hypotheses=5, sampling_temperature=0.0)     # as the batched pipeline passes them
first = run("beam_long", enc["long"], [prompt] * 8, **batched)
run("beam_mid", enc["mid"], [prompt] * 8, **batched)
run("beam_6", enc["6"], [prompt] * 6, **batched)
run("beam_4_448", enc["4"], [prompt] * 4, forced=True, **batched)
previous = model.get_prompt(tk, first[0].sequences_ids[0], without_timestamps=False)
run("beam_prompt_448", enc["1"], [previous], forced=True, num_hypotheses=5)
fallback = dict(beam_size=1, num_hypotheses=5, sampling_topk=0, sampling_temperature=0.2, logits=True)
run("sample_1", enc["1"], [prompt], **fallback)
run("sample_1_448", enc["1"], [prompt], forced=True, **fallback)
summary = {"ctranslate2": ctranslate2.__file__, "seconds": round(time.time() - t0, 1), "peak_mb": peak,
           "all": hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]}
if save:
    json.dump(out, open(save, "w"), indent=1, sort_keys=True)
if golden:
    ref = json.load(open(golden))
    bad = sorted(k for k in ref.keys() | out.keys() if ref.get(k) != out.get(k))
    summary.update(digest="FAIL" if bad else "PASS", mismatched=bad)
print(json.dumps({**out, **summary}), flush=True)
