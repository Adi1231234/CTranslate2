"""The long-form speed sample from the transcribe work list (corpus-backend todo.mjs): the first 48 of its YODAS v3 iw
recordings found in espnet/yodas3's first audio archive (streamed until found, nothing else kept) and 2 of its
Knesset plenums of 1-2 hours (ivrit-ai/knesset-plenums/<id>/audio.m4a). Writes <out>/audio/<file> and <out>/list.txt
(source<TAB>id<TAB>file). usage: python fetch_long_sample.py <choice.json> <out dir>"""
import json, os, shutil, sys, tarfile
from huggingface_hub import HfFileSystem, hf_hub_download

choice, out = sys.argv[1:3]
doc = json.load(open(choice, encoding="utf-8"))
want = {i for s, i, h, left in doc["recordings"] if s == "YODAS v3 iw"}
plenums = sorted((h, i) for s, i, h, left in doc["recordings"] if s == "מליאות הכנסת" and 1 <= h <= 2)[:2]
audio = os.path.join(out, "audio")
os.makedirs(audio, exist_ok=True)
rows = []
with HfFileSystem().open("datasets/espnet/yodas3/data/iw/audio/0000.tar", "rb", block_size=16 << 20) as f:
    tar = tarfile.open(fileobj=f, mode="r|")
    for m in tar:
        key, ext = os.path.splitext(m.name)
        if ext == ".webm" and key in want:
            with open(os.path.join(audio, m.name), "wb") as dst:
                shutil.copyfileobj(tar.extractfile(m), dst)
            rows.append(("YODAS v3 iw", key, m.name))
            print(len(rows), m.name, m.size, flush=True)
            if len(rows) >= 48:
                break
for hours, pid in plenums:
    src = hf_hub_download("ivrit-ai/knesset-plenums", f"{pid}/audio.m4a", repo_type="dataset")
    shutil.copyfile(src, os.path.join(audio, f"knesset-{pid}.m4a"))
    rows.append(("מליאות הכנסת", pid, f"knesset-{pid}.m4a"))
    print("plenum", pid, hours, flush=True)
with open(os.path.join(out, "list.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("".join("\t".join(r) + "\n" for r in rows))
print(f"{len(rows)} recordings")
