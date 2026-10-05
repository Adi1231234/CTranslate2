"""Does the fork's seeded sampling draw exactly what stock CTranslate2 draws? Stock's multinomial takes row i's random
number from a Philox state curand_init(seed, i, 0) that a worker thread creates at its first sampling call
(src/cuda/random.cu, the seed of set_random_seed); the fork's seeded rows take theirs from curand_init(seed,
hypothesis, 0) of their own (src/cuda/row_random.cu). A call of one hypothesis (one row: never compacted) on a fresh
worker thread so gets the same random numbers in both, and if the fork computes every probability with stock's bits,
every token and every score's bits match. Stock: a new model (new worker thread, fresh states) for each case, after
set_random_seed; the fork: one model, sampling_seeds=[seed]. The cases: every clip of a uuid list (the fallback's
clips), its first window as the ladder samples it, at each sampled temperature of the ladder, the full context.
usage: seed_vs_stock.py <runner dir> <cache dir> <units list> <uuid list> <result.json>  (stock or fork: PYTHONPATH)"""
import gc, json, os, sys, zlib
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ctranslate2
from faster_whisper.transcribe import get_ctranslate2_storage, get_suppressed_tokens
from faster_whisper.utils import download_model
import common
from stock_context import is_fork, start_step, stock_max_length

runner, cache, listing, uuids, result = sys.argv[1:6]
wanted = {line.rstrip("\n") for line in open(uuids, encoding="utf-8") if line.strip()}   # uuids hold spaces
clips = [(u, w) for unit in common.cached_units(cache, listing) for u, w in unit if u in wanted]
model, tk, prompt = common.whisper()            # features, tokenizer and prompt (this model never samples)
feats = common.features(model, [w for _, w in clips])
fork = is_fork()
max_length = 448 if fork else stock_max_length(448, start_step(prompt, tk.sot, tk.no_timestamps))
kw = dict(beam_size=1, num_hypotheses=1, sampling_topk=0, length_penalty=1, max_length=max_length,
          return_scores=True, return_no_speech_prob=True, suppress_blank=True,
          suppress_tokens=list(get_suppressed_tokens(tk, [-1])), max_initial_timestamp_index=50)
path = download_model("ivrit-ai/whisper-large-v3-ct2", local_files_only=True)
out = []
for i, (uuid, _) in enumerate(clips):
    x = get_ctranslate2_storage(feats[i:i + 1])
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        seed = zlib.crc32(f"{uuid}|{t}".encode("utf-8")) % 0xFFFFFFFF    # never 2^32 - 1, stock's "unseeded"
        if fork:
            r = model.model.generate(model.model.encode(x, to_cpu=False), [prompt], sampling_temperature=t,
                                     sampling_seeds=[seed], **kw)[0]
        else:
            ctranslate2.set_random_seed(seed)
            fresh = ctranslate2.models.Whisper(path, device="cuda", compute_type="default", inter_threads=1,
                                               intra_threads=1)
            r = fresh.generate(fresh.encode(x, to_cpu=False), [prompt], sampling_temperature=t, **kw)[0]
            del fresh
            gc.collect()
        out.append({"uuid": uuid, "t": t, "seed": seed, "ids": r.sequences_ids,
                    "scores": [float(s).hex() for s in r.scores], "no_speech": float(r.no_speech_prob).hex()})
    print(f"{i + 1}/{len(clips)} clips", flush=True)
json.dump(out, open(result, "w", encoding="utf-8"), ensure_ascii=False)
steps = sum(len(o["ids"][0]) for o in out)
print(f"{'fork' if fork else 'stock'}: {len(out)} sampled calls, {steps} sampled tokens, max_length {max_length}")
