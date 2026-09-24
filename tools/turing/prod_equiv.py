"""Production-path canary: run the production transcription engine (engine.transcribe_unit, the same
batching, encoder/decoder pipelining, segment splitting and temperature-ladder fallback as the real
run) on every sample clip, and print a hash of every output row. A build may replace the stock wheel
only if this hash equals the stock wheel's.
usage: prod_equiv.py <sample_dir> <engine_dir> <mode, e.g. pipe8> [ctranslate2 package parent dir]
N_CLIPS=<n>: only the first n sample clips. CPU_THREADS=<n>: CTranslate2 intra_threads.
PROFILE_RANGE=1: run once to warm up, then mark the measured run with cuProfilerStart/Stop, so
`nsys profile --capture-range=cudaProfilerApi` records only the steady state.
POOL_RETAIN=1: the CUDA memory pool keeps freed memory (release threshold UINT64_MAX) instead of
returning it to the OS at every synchronize. The output also reports the pool's high-water marks.
GPU_TIME=1: CUPTI kernel records of the measured run: gpu_busy_s (union of kernel intervals) and
gpu_kernel_s (sum of kernel durations), which other processes' CPU load does not inflate."""
import os, sys, json, time, hashlib
from concurrent.futures import Future, ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SAMPLE, ENGINE, MODE = sys.argv[1:4]
common.init(sys.argv[4] if len(sys.argv) > 4 else None)
sys.path.insert(0, ENGINE)
import ctranslate2
from faster_whisper import WhisperModel
from engine import transcribe_unit

n_clips = int(os.environ.get("N_CLIPS", "150"))       # fewer clips keep a sampled profile small
clips = common.sample_clips(SAMPLE)[:n_clips]
workers = 1 + int(MODE.startswith("pipe")) + 1                  # as transcribe_run.py
cpu_threads = int(os.environ.get("CPU_THREADS", "0"))    # CTranslate2 intra_threads (OpenMP team size)
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=workers, cpu_threads=cpu_threads)
mem = common.mempool()
if os.environ.get("POOL_RETAIN") == "1":
    mem(common.RELEASE_THRESHOLD, 2**64 - 1)
pool = ThreadPoolExecutor(max_workers=1)
run = lambda: [r.result() if isinstance(r, Future) else r for r in transcribe_unit(model, clips, MODE, pool)]
profile = os.environ.get("PROFILE_RANGE") == "1"   # nsys --capture-range=cudaProfilerApi: warm run only
if profile:
    import ctypes
    cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    run()
    cuda.cuProfilerStart()
tracer = None
if os.environ.get("GPU_TIME") == "1":
    from cupti import Tracer, busy
    tracer = Tracer(os.path.join(ENGINE, "cupti"), names=False)
    tracer.start()
t = time.time()
rows = run()
T = time.time() - t
if profile:
    cuda.cuProfilerStop()
gpu = {}
if tracer:
    tracer.stop()
    iv = tracer.records
    gpu = {"gpu_busy_s": round(busy(iv) / 1e9, 2), "gpu_kernel_s": round(sum(e - s for s, e in iv) / 1e9, 2),
           "kernels": len(iv)}
audio = sum(len(w) for _, w in clips) / 16000
print(json.dumps({"ctranslate2": ctranslate2.__file__, "mode": MODE, "cpu_threads": cpu_threads, "clips": len(rows),
                  "fallback": sum(r.get("path") == "fallback" for r in rows),
                  "seconds": round(T, 1), "realtime": round(audio / T, 1),
                  "release_threshold": mem(common.RELEASE_THRESHOLD), "pool_reserved_high_mb":
                  mem(common.RESERVED_HIGH) >> 20, "pool_used_high_mb": mem(common.USED_HIGH) >> 20,
                  **gpu,
                  "rows_sha": hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:16]}),
      flush=True)
pool.shutdown()
del model                                       # release the model's worker threads while Python is alive
import gc; gc.collect()
