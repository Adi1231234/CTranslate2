"""The fallback ladder's sampled attempts, seeded, as bytes to compare between two runs: every clip of a uuid list
(e.g. a round's fallback clips) is encoded alone and sampled at each temperature of the ladder (best of 5, as
faster-whisper's generate_with_fallback calls CTranslate2), and every hypothesis's tokens and score are written.
Run once with CT2_SHARED_MEMORY_ROWS=0 (the memory repeated a hypothesis, stock) and once without, then compare the
two files: the shared copy must not change a token or a score's bit.
usage: sampled_check.py <runner dir> <cache dir> <units list> <uuid list> <result.json>"""
import glob, json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.transcribe import get_ctranslate2_storage
from faster_whisper.tokenizer import Tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

runner, cache, listing, uuids, result = sys.argv[1:6]
fallback = {line.rstrip("\n") for line in open(uuids, encoding="utf-8") if line.strip()}   # uuids hold spaces
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default", num_workers=1,
                     cpu_threads=1)
tk = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
prompt = model.get_prompt(tk, [], without_timestamps=False)
clips = [(u, w) for unit in common.cached_units(cache, listing) for u, w in unit if u in fallback]
out = []
from faster_whisper.audio import pad_or_trim
feats = np.stack([pad_or_trim(model.feature_extractor(w)[..., :-1]) for _, w in clips]).astype(np.float32)
kw = dict(max_length=448, return_scores=True, return_no_speech_prob=True, suppress_blank=True,
          suppress_tokens=[-1], max_initial_timestamp_index=50)
entry = lambda r: {"ids": r.sequences_ids, "scores": [float(s).hex() for s in r.scores],
                   "no_speech": float(r.no_speech_prob).hex()}
record = lambda uuid, t, r: out.append({"uuid": uuid, "t": t, **entry(r)})
if os.environ.get("CHECK") == "joined":
    # The sampling path without random draws (temperature 0: the best token of each step, 5 hypotheses as sampling
    # repeats them): every clip alone, then all clips in one call
    # with group_size=1 (each clip's arithmetic its own): the joined call must give every clip the same bytes.
    # Likewise the ladder's beam attempt (T=0, beam 5).
    for name, opts in (("sampling", dict(beam_size=1, num_hypotheses=5, sampling_topk=0, sampling_temperature=0)),
                       ("beam", dict(beam_size=5, patience=1))):
        for i, (uuid, _) in enumerate(clips):
            enc = model.model.encode(get_ctranslate2_storage(feats[i:i + 1]), to_cpu=False)
            record(uuid, name + " alone", model.model.generate(enc, [prompt], **opts, **kw)[0])
        enc = model.model.encode(get_ctranslate2_storage(feats), to_cpu=False, group_size=1)
        joined = model.model.generate(enc, [prompt] * len(clips), group_size=1, **opts, **kw)
        for (uuid, _), r in zip(clips, joined):
            record(uuid, name + " joined", r)
        alone = {o["uuid"]: o for o in out if o["t"] == name + " alone"}
        same = sum({**o, "t": 0} == {**alone[o["uuid"]], "t": 0} for o in out if o["t"] == name + " joined")
        print(f"{name}: {same} of {len(clips)} clips joined identical to alone")
    # Sampling with each clip's own seed (sampling_seeds): every temperature of the ladder, each clip alone, then
    # all clips joined with their seeds, then alone again (a repeat must draw the same).
    for t in (0.2, 0.6, 1.0):
        opts = dict(beam_size=1, num_hypotheses=5, sampling_topk=0, sampling_temperature=t)
        seeds = [(int.from_bytes(uuid.encode("utf-8")[:8].ljust(8, b"\0"), "little") + i) % 2 ** 64 for i, (uuid, _) in
                 enumerate(clips)]
        runs = {}
        for name in ("alone", "joined", "again"):
            if name == "joined":
                enc = model.model.encode(get_ctranslate2_storage(feats), to_cpu=False, group_size=1)
                runs[name] = [entry(r) for (u, _), r in zip(
                    clips, model.model.generate(enc, [prompt] * len(clips), group_size=1, sampling_seeds=seeds,
                                                **opts, **kw))]
            else:
                runs[name] = []
                for i, (uuid, _) in enumerate(clips):
                    enc = model.model.encode(get_ctranslate2_storage(feats[i:i + 1]), to_cpu=False)
                    r = model.model.generate(enc, [prompt], sampling_seeds=[seeds[i]], **opts, **kw)[0]
                    runs[name].append(entry(r))
        same = sum(a == b for a, b in zip(runs["joined"], runs["alone"]))
        again = sum(a == b for a, b in zip(runs["again"], runs["alone"]))
        print(f"seeded T={t}: {same} of {len(clips)} clips joined identical to alone, {again} repeat alone")
elif os.environ.get("CHECK") == "variants":
    # The ladder's sampled temperatures in one search (sampling_temperatures, the runner's RUN_FALLBACK_SPECULATE):
    # every clip's attempt at each temperature alone with its own seed, against the clip's attempts in one call, and
    # against 4 clips' in one call (group_size=1): every attempt must keep its tokens, score and no-speech bits.
    import time
    temps = [0.2, 0.4, 0.6, 0.8, 1.0]
    opts = dict(beam_size=1, num_hypotheses=5, sampling_topk=0)
    seed = lambda i, v: (int.from_bytes(clips[i][0].encode("utf-8")[:8].ljust(8, b"\0"), "little") + 7919 * v) % 2 ** 64
    encode = lambda a, b: model.model.encode(get_ctranslate2_storage(feats[a:b]), to_cpu=False, group_size=1)
    runs, took = {}, {}
    for name in ("alone", "clip", "joined"):
        t0, runs[name] = time.time(), []
        if name == "alone":
            for i in range(len(clips)):
                enc = encode(i, i + 1)
                for v, t in enumerate(temps):
                    runs[name].append(entry(model.model.generate(enc, [prompt], sampling_temperature=t,
                                                                 sampling_seeds=[seed(i, v)], **opts, **kw)[0]))
        else:
            step = 1 if name == "clip" else 4
            for a in range(0, len(clips), step):
                b = min(a + step, len(clips))
                rs = model.model.generate(encode(a, b), [prompt] * (b - a), group_size=1, sampling_temperatures=temps,
                                          sampling_seeds=[seed(i, v) for i in range(a, b) for v in range(len(temps))],
                                          **opts, **kw)
                runs[name] += [entry(r) for r in rs]
        took[name] = time.time() - t0
    attempts = len(clips) * len(temps)
    for name in ("clip", "joined"):
        same = sum(x == y for x, y in zip(runs[name], runs["alone"]))
        print(f"variants {name}: {same} of {attempts} attempts identical to alone ({len(runs[name])} returned); "
              f"{took[name]:.1f} s against {took['alone']:.1f} s alone")
else:
    ctranslate2.set_random_seed(1234)
    for i, (uuid, _) in enumerate(clips):
        enc = model.model.encode(get_ctranslate2_storage(feats[i:i + 1]), to_cpu=False)
        for t in (0.2, 0.4, 0.6, 0.8, 1.0):
            record(uuid, t, model.model.generate(enc, [prompt], beam_size=1, num_hypotheses=5, sampling_topk=0,
                                                 sampling_temperature=t, **kw)[0])
json.dump(out, open(result, "w"))
print(f"{len(clips)} fallback clips -> {result}")
