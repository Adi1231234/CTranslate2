"""Every number the model outputs, bit for bit: greedy decoding (beam 1, temperature 0) of every clip of the first
units of a list, in the runner's batches of 8 (shortest first), returning each step's full-vocabulary logits
(return_logits_vocab: 51,866 values a step, after the timestamp rules and the suppressed tokens, before the
log-softmax). Writes per clip the sha256 of all its steps' logits bytes, the steps, the tokens and the bits of its
score and no-speech probability. Run with the stock wheel and with the fork (both the full context:
stock_context.py), then compare the two files byte for byte.
usage: logits_check.py <runner dir> <cache dir> <units list> <units> <result.jsonl>   (stock or fork: PYTHONPATH)"""
import hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import ctranslate2
from faster_whisper.transcribe import get_ctranslate2_storage, get_suppressed_tokens
import common
from stock_context import is_fork, start_step, stock_max_length

runner, cache, listing, count, result = sys.argv[1:6]
first = result + ".units"
open(first, "w").write("\n".join(open(listing).read().split()[:int(count)]))
model, tk, prompt = common.whisper()
max_length = 448 if is_fork() else stock_max_length(448, start_step(prompt, tk.sot, tk.no_timestamps))
kw = dict(beam_size=1, length_penalty=1, max_length=max_length, sampling_temperature=0, return_scores=True,
          return_no_speech_prob=True, return_logits_vocab=True, suppress_blank=True,
          suppress_tokens=list(get_suppressed_tokens(tk, [-1])), max_initial_timestamp_index=50)
cpu, clips_done, values = ctranslate2.Device.cpu, 0, 0
with open(result, "w", encoding="utf-8") as f:
    for unit in common.cached_units(cache, first):
        clips = sorted(unit, key=lambda c: len(c[1]))          # the runner's order (engine.length_order)
        for i in range(0, len(clips), 8):
            batch = clips[i:i + 8]
            x = get_ctranslate2_storage(common.features(model, [w for _, w in batch]))
            results = model.model.generate(model.model.encode(x, to_cpu=False), [prompt] * len(batch), **kw)
            for (uuid, _), r in zip(batch, results):
                digest = hashlib.sha256()
                for step in r.logits[0]:
                    a = np.asarray(step.to_device(cpu))
                    digest.update(a.tobytes())
                    values += a.size
                f.write(json.dumps({"uuid": uuid, "steps": len(r.logits[0]), "logits": digest.hexdigest(),
                                    "ids": r.sequences_ids[0], "score": float(r.scores[0]).hex(),
                                    "no_speech": float(r.no_speech_prob).hex()}, ensure_ascii=False) + "\n")
                clips_done += 1
print(f"{'fork' if is_fork() else 'stock'}: {clips_done} clips, {values:,} logits, max_length {max_length}")
