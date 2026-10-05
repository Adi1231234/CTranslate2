"""The Whisper encoder alone on every clip of a units list, in the stream's batches of 8 (stream_engine.py encodes
features[i:i + 8] with WhisperModel.encode), one process: time, the GPU's energy (NVML's counter) and power, energy
a clip, and the encoder's own rate in audio seconds a second, the batched path's ceiling were its decoding free.
usage: encoder_bench.py <runner dir> <cache dir> <units list> [passes, default 2]"""
import ctypes, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import common
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim

runner, cache, listing = sys.argv[1:4]
passes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
clips = [w for unit in common.cached_units(cache, listing) for _, w in unit]
audio_s = sum(len(w) for w in clips) / 16000

nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvml.nvmlInit_v2()
handle = ctypes.c_void_p()
nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))


def energy_j():
    e = ctypes.c_ulonglong()
    nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(e))
    return e.value / 1000


model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default", num_workers=1,
                     cpu_threads=1)
with ThreadPoolExecutor(6) as pool:
    feats = list(pool.map(lambda w: pad_or_trim(model.feature_extractor(w)[..., :-1]).astype(np.float32), clips))
batches = [np.stack(feats[i:i + 8]) for i in range(0, len(feats), 8)]
model.encode(batches[0])                                    # load, allocator warm-up
e0, t0 = energy_j(), time.time()
time.sleep(3)
idle_w = (energy_j() - e0) / (time.time() - t0)
for p in range(passes):
    e0, t0 = energy_j(), time.time()
    for b in batches:
        out = model.encode(b)
    del out
    wall, joules = time.time() - t0, energy_j() - e0
    print(f"encoder pass {p}: {len(clips)} clips ({audio_s / 3600:.2f} h) in {wall:.1f} s = {audio_s / wall:.1f}x "
          f"realtime, {joules / 1000:.1f} kJ, {joules / wall:.0f} W, {joules / len(clips):.2f} J a clip "
          f"(idle {idle_w:.0f} W)", flush=True)
