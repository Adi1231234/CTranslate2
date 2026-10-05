"""The fallback ladders alone: every clip of a uuid list (a run's fallback clips) through the runner's own fallback
pool (fallback.make_pool, the RUN_FALLBACK* settings from the environment) and faster-whisper's sequential
transcribe, all submitted at once, as the full run gathers them. Prints the wall time, the GPU's energy over it
(NVML's counter) and per ladder, the idle power before, and a digest of the rows: two settings that must decode
alike (e.g. RUN_FALLBACK_SPECULATE=0 and 1) print the same digest.
usage: ladder_bench.py <runner dir> <cache dir> <units list> <uuid list>"""
import ctypes, hashlib, json, os, sys, time
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from faster_whisper import WhisperModel
import engine
from fallback import make_pool

runner, cache, listing, uuids = sys.argv[1:5]
wanted = {line.rstrip("\n") for line in open(uuids, encoding="utf-8") if line.strip()}   # uuids hold spaces
clips = [(u, w) for unit in common.cached_units(cache, listing) for u, w in unit if u in wanted]

nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvml.nvmlInit_v2()
handle = ctypes.c_void_p()
nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))


def energy_j():
    e = ctypes.c_ulonglong()
    nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(e))
    return e.value / 1000


model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default", num_workers=2,
                     cpu_threads=1)
pool, _ = make_pool(os.environ.get("RUN_FALLBACK", "batched"), lambda m: None)
# One ladder first, alone (load, JIT, allocator warm-up), then the idle power, then every ladder at once
# (RUN_FALLBACK_THREADS at least the clips, or the pool runs them in turns).
warm = pool.submit(engine._fallback, model, clips[0][0], clips[0][1])
if hasattr(pool, "flush"):
    pool.flush()
warm.result()
time.sleep(1)
e0, t0 = energy_j(), time.time()
time.sleep(3)
idle_w = (energy_j() - e0) / (time.time() - t0)
e0, t0 = energy_j(), time.time()
futures = [pool.submit(engine._fallback, model, u, w) for u, w in clips]
if hasattr(pool, "flush"):
    pool.flush()
rows = [f.result() for f in futures]
wall, joules = time.time() - t0, energy_j() - e0
digest = hashlib.sha256("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                                for r in rows).encode("utf-8")).hexdigest()[:16]
print(f"ladders: {len(clips)} clips, {wall:.1f} s, {joules / 1000:.2f} kJ ({joules / len(clips):.0f} J a ladder, "
      f"{joules / wall:.0f} W; idle {idle_w:.0f} W); rows digest {digest}", flush=True)
