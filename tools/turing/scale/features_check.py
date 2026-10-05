"""Byte check of runner/chunked_features.py against faster-whisper's own FeatureExtractor on real recordings: each file
of a folder (or its first hours), the original's features against the chunked ones at several block sizes (odd
sizes, an hour), byte for byte.
usage: features_check.py <runner dir> <audio dir> [max hours a file, default 2]"""
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
from chunked_features import ChunkedFeatures

audio_dir = sys.argv[2]
hours = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
fe, total, bad = FeatureExtractor(feature_size=128), 0, 0
for name in sorted(os.listdir(audio_dir)):
    wav = decode_audio(os.path.join(audio_dir, name))[:int(hours * 3600 * 16000)]
    ref = fe(wav)
    for frames in (997, 12345, 360000):
        chunked = ChunkedFeatures(fe)
        chunked.frames = frames
        got = chunked(wav)
        same = got.shape == ref.shape and got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
        total += 1
        bad += not same
        if not same or frames == 360000:
            print(f"{name}: {len(wav) / 16000 / 3600:.2f} h, {ref.shape[1]} frames, blocks of {frames}: "
                  f"{'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
print(f"TOTAL {bad} of {total} differ (numpy {np.__version__})")
