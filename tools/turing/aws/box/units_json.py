"""Writes <runner>/units.json from HF the way the runner does (units.list_units) and checks it against the
production file's sha256 prefix, so the box walks the same 2,268 units in the same order.
usage: units_json.py <runner dir with hf_token.txt> <sha256 prefix>"""
import hashlib, json, os, sys

runner, want = sys.argv[1], sys.argv[2].upper()
sys.path.insert(0, runner)
from huggingface_hub import HfApi, HfFileSystem
from units import list_units

token = open(os.path.join(runner, "hf_token.txt")).read().strip()
path = os.path.join(runner, "units.json")
json.dump(list_units(HfFileSystem(token=token), HfApi(token=token)), open(path, "w"))
got = hashlib.sha256(open(path, "rb").read()).hexdigest().upper()
print("units.json", got[:16], "units", len(json.load(open(path))))
sys.exit(0 if got.startswith(want) else f"units.json {got[:16]} is not the production file {want}")
