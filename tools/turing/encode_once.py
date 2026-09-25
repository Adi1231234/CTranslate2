"""One encoder call on 8 real sample clips (the longest, or from index FIRST), after one warm-up call: a small
target for a kernel profiler (e.g. `ncu --kernel-name regex:exact_attention --launch-skip 32 --launch-count 1`,
skipping the warm-up call's 32 layers).
usage: encode_once.py <sample_dir> [ctranslate2 package parent dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
common.init(sys.argv[2] if len(sys.argv) > 2 else None)
from faster_whisper.transcribe import get_ctranslate2_storage

model, _, _ = common.whisper()
waves = common.sample_waves(sys.argv[1])
first = int(os.environ.get("FIRST", len(waves) - 8))
features = get_ctranslate2_storage(common.features(model, waves[first:first + 8]))
for _ in range(2):
    model.model.encode(features, to_cpu=True)
print("encoded 8 clips twice", flush=True)
del model
