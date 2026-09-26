"""Fetch units' row groups into a cache folder ahead of a run (runner/fetch.py and its RUN_CACHE layout), several at
a time, so a later run (scale_run.ps1 -Cache, seeded.py with RUN_CACHE) reads them from disk. Network only: it can
run while the GPU works on something else. The list holds unit ids one per line, or seeded.py's "unit uuid" lines.
usage: prefetch.py <runner_dir> <list file> <cache dir> [threads, default 4]    (the runner dir holds hf_token.txt)"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor

RUNNER, LISTING, CACHE = sys.argv[1:4]
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
sys.path.insert(0, RUNNER)
os.environ["HF_HOME"] = os.path.join(RUNNER, "hf")
from huggingface_hub import HfFileSystem
from units import unit_id
from fetch import read_unit

wanted = sorted({line.split(" ", 1)[0].strip() for line in open(LISTING, encoding="utf-8") if line.strip()})
units = {unit_id(u): u for u in json.load(open(os.path.join(RUNNER, "units.json")))}
os.makedirs(CACHE, exist_ok=True)
todo = [u for u in wanted if not os.path.exists(os.path.join(CACHE, u + ".parquet"))]
fs = HfFileSystem(token=open(os.path.join(RUNNER, "hf_token.txt")).read().strip())
t0, done = time.time(), [0]


def one(uid):
    table = read_unit(fs, {}, units[uid], uid, print, CACHE)
    done[0] += 1
    if done[0] % 100 == 0:
        print(f"{done[0]}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
    return table is not None


with ThreadPoolExecutor(THREADS) as pool:
    fetched = sum(pool.map(one, todo))
print(json.dumps({"wanted": len(wanted), "cached_before": len(wanted) - len(todo), "fetched": fetched,
                  "failed": len(todo) - fetched, "seconds": round(time.time() - t0)}), flush=True)
