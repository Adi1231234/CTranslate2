"""Work units = (shard, row_group) of crowd-transcribe-v5. Front/back split between machines."""
import os, json
DS = "ivrit-ai/crowd-transcribe-v5"

def list_units(fs, api):
    """Every (shard, rg) in a stable global order: train shards then test shards."""
    import pyarrow.parquet as pq
    files = sorted(f for f in api.list_repo_files(DS, repo_type="dataset") if f.endswith(".parquet"))
    files = [f for f in files if "/train-" in f] + [f for f in files if "/test-" in f]
    units = []
    for f in files:
        with fs.open(f"datasets/{DS}/{f}", "rb") as fh:
            n = pq.ParquetFile(fh).num_row_groups
        units += [[f, g] for g in range(n)]
    return units

def unit_id(u):
    return f"{os.path.basename(u[0]).replace('.parquet', '')}_rg{u[1]:03d}"

def my_order(units, direction):
    """'front' walks 0..N-1, 'back' walks N-1..0; a stop file bounds each side."""
    return list(units) if direction == "front" else list(reversed(units))

def load_stop(root):
    """Optional stop.json {"stop_before_unit": "<unit_id>", "deadline": "YYYY-MM-DD HH:MM"}.
    Re-read before every unit, so the coordinator can move either bound while running."""
    p = os.path.join(root, "stop.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}

def should_skip(root, uid):
    """stop.json {"skip_units": [...]} = units another machine already finished."""
    return uid in set(load_stop(root).get("skip_units", []))

def should_stop(root, uid):
    import time
    s = load_stop(root)
    if s.get("stop_before_unit") == uid:
        return "stop_before_unit"
    d = s.get("deadline")
    if d and time.time() >= time.mktime(time.strptime(d, "%Y-%m-%d %H:%M")):
        return f"deadline {d}"
    return None
