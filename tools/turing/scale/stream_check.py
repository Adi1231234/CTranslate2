"""CTranslate2's Whisper stream (Whisper.open_stream: batches decoded together, each joining as soon as there is
room) against generate() on each batch alone, with the pipelined runner's options (beam 5, timestamps, the
no-speech probability): every clip's tokens, score and no-speech probability must have the same bits. Real clips
of the cached units, each unit's clips sorted by length (as engine.py batches them) in batches of 8, the last one
of a unit shorter. Both ways are timed on the same encoder outputs.
usage: stream_check.py <runner dir> <cache dir> <units list> <result.json> [max clips] [max_batches:max_rows ...]"""
import json, os, sys, threading, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_ctranslate2_storage, get_suppressed_tokens
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

runner, cache, listing, result = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 800
configs = [tuple(map(int, c.split(":"))) for c in sys.argv[6:]] or [(8, 320), (3, 320), (16, 640)]
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default", num_workers=2,
                     cpu_threads=1)
ct2 = model.model
tk = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
prompt = model.get_prompt(tk, [], without_timestamps=False)
kw = dict(beam_size=5, patience=1, length_penalty=1, max_length=model.max_length, suppress_blank=True,
          suppress_tokens=get_suppressed_tokens(tk, [-1]), return_scores=True, return_no_speech_prob=True,
          repetition_penalty=1, no_repeat_ngram_size=0)

batches, seconds = [], 0.0                                  # (clip uuids, features [n, 128, 3000])
for unit in common.cached_units(cache, listing):
    clips = sorted(unit, key=lambda c: len(c[1]))
    for i in range(0, len(clips), 8):
        part = clips[i:i + 8]
        feats = np.stack([pad_or_trim(model.feature_extractor(w)[..., :-1]) for _, w in part]).astype(np.float32)
        batches.append(([u for u, _ in part], feats))
        seconds += sum(len(w) for _, w in part) / 16000
    if sum(len(b[0]) for b in batches) >= limit:
        break
encoded = [ct2.encode(get_ctranslate2_storage(f), to_cpu=False) for _, f in batches]
clips = sum(len(u) for u, _ in batches)
print(f"{clips} clips, {len(batches)} batches, {seconds / 3600:.2f} h", flush=True)

def record(results):
    return [{"ids": r.sequences_ids, "scores": [float(s).hex() for s in r.scores],
             "no_speech": float(r.no_speech_prob).hex()} for r in results]

t0 = time.perf_counter()
reference = [record(ct2.generate(enc, [prompt] * len(u), sampling_temperature=0, **kw))
             for (u, _), enc in zip(batches, encoded)]
alone_s = time.perf_counter() - t0
print(f"generate alone: {alone_s:.1f} s ({seconds / alone_s:.1f}x)", flush=True)

report = {"clips": clips, "batches": len(batches), "audio_s": seconds, "alone_s": alone_s, "streams": []}
for max_batches, max_rows in configs:
    stream = ct2.open_stream(max_batches=max_batches, max_rows=max_rows, max_pending=2, **kw)
    t0 = time.perf_counter()

    def feed():
        for tag, ((u, _), enc) in enumerate(zip(batches, encoded)):
            stream.submit(tag, enc, [prompt] * len(u))
        stream.close()

    feeder = threading.Thread(target=feed)
    feeder.start()
    got = {}
    while (item := stream.next()) is not None:
        got[item[0]] = record(item[1])
    feeder.join()
    took = time.perf_counter() - t0
    same = sum(got.get(b) == reference[b] for b in range(len(batches)))
    clips_same = sum(a == b for i in range(len(batches)) for a, b in zip(got.get(i, []), reference[i]))
    print(f"stream max_batches={max_batches} max_rows={max_rows}: {took:.1f} s ({seconds / took:.1f}x), "
          f"{same} of {len(batches)} batches and {clips_same} of {clips} clips identical to generate alone",
          flush=True)
    report["streams"].append({"max_batches": max_batches, "max_rows": max_rows, "s": took,
                              "batches_same": same, "clips_same": clips_same})
json.dump(report, open(result, "w"))
print("IDENTICAL" if all(s["clips_same"] == clips for s in report["streams"]) else "DIFFERENT")
