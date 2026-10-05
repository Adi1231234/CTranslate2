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
record = lambda uuid, t, r: out.append({"uuid": uuid, "t": t, "ids": r.sequences_ids,
                                        "scores": [float(s).hex() for s in r.scores],
                                        "no_speech": float(r.no_speech_prob).hex()})
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
else:
    ctranslate2.set_random_seed(1234)
    for i, (uuid, _) in enumerate(clips):
        enc = model.model.encode(get_ctranslate2_storage(feats[i:i + 1]), to_cpu=False)
        for t in (0.2, 0.4, 0.6, 0.8, 1.0):
            record(uuid, t, model.model.generate(enc, [prompt], beam_size=1, num_hypotheses=5, sampling_topk=0,
                                                 sampling_temperature=t, **kw)[0])
json.dump(out, open(result, "w"))
print(f"{len(clips)} fallback clips -> {result}")
