"""Long recordings from a list (source<TAB>id<TAB>file in an audio folder), transcribed with the crowd-v5 parameters:
MODE=long through LongEngine (longform.py, the fork: many recordings at once, the audio decoded in worker processes,
long_decode.py), MODE=seq one after the other through faster-whisper's own transcribe (the stock wheel's reference
with RUN_STOCK_FULL_CONTEXT=1 and RUN_SEED, or the fork). Rows to <out>/rows.jsonl in the list's order, each with its
source and id (MODE=long: each as its recording ends, then all in the list's order at the end, so a run stopped on
time keeps the rows it finished). LONG_SECONDS=<n> (measurement only): each recording's first n seconds. LONG_SHARD=<i>/<n>: every n-th
recording from the i-th (several processes on one GPU). Prints each recording's audio and time, then the rate from
the model load to the end.
usage: python longform_run.py <list> <audio dir> <out dir>"""
import json, os, sys, time
from concurrent.futures import as_completed
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import cudaenv  # noqa: F401


def main():
    from faster_whisper import WhisperModel, decode_audio
    import engine
    listing, audio_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    mode, cut = os.environ.get("MODE", "long"), int(os.environ.get("LONG_SECONDS", "0"))
    items = [line.rstrip("\n").split("\t") for line in open(listing, encoding="utf-8") if line.strip()]
    shard, shards = map(int, os.environ.get("LONG_SHARD", "0/1").split("/"))
    items = items[shard::shards]
    if mode == "long":                          # the decoding processes start before CUDA and the threads
        from long_decode import Decoder
        decoder = Decoder()
    if os.environ.get("RUN_SEED"):              # before the model: its workers seed their sampler states from it
        import ctranslate2
        ctranslate2.set_random_seed(int(os.environ["RUN_SEED"]))
    if mode == "long":
        from longform import workers_needed
    model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                         num_workers=workers_needed() if mode == "long" else 1,   # long: longform.py's
                         cpu_threads=1)
    if os.environ.get("RUN_STOCK_FULL_CONTEXT") == "1":
        from stock_context import full_context
        full_context(model)
    t0, totals, rows = time.time(), {"audio_s": 0.0}, {}

    def done(i, row, started):
        source, rid, _ = items[i]
        rows[i] = {"source": source, "id": rid, **row}
        totals["audio_s"] += row["dur_s"]
        print(f"{source} {rid}: {row['dur_s']:.0f} s audio, {time.time() - started:.0f} s, "
              f"{len(row['segments'])} segments, total {totals['audio_s'] / (time.time() - t0):.1f}x", flush=True)

    if mode == "long":
        from longform import LongEngine
        long = LongEngine(model)
        started = time.time()
        futures = {long.submit(f"{s}|{i}", decoder.loader(os.path.join(audio_dir, name), cut)): k
                   for k, (s, i, name) in enumerate(items)}
        # Each row as its recording ends (a run stopped on time keeps the rows done), all in order at the end.
        with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
            for future in as_completed(futures):
                k = futures[future]
                done(k, future.result(), started)
                f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
                f.flush()
        print(long.stats.report(), flush=True)
    else:
        for k, (s, i, name) in enumerate(items):
            started, wav = time.time(), decode_audio(os.path.join(audio_dir, name))
            wav = wav[:cut * 16000] if cut else wav
            segments, _ = model.transcribe(wav, **engine.EXACT)
            done(k, engine._row(f"{s}|{i}", wav, list(segments), "seq"), started)
    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for k in range(len(items)):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    wall = time.time() - t0
    print(f"RESULT mode={mode} recordings={len(items)} audio_h={totals['audio_s'] / 3600:.3f} wall_s={wall:.1f} "
          f"x_realtime={totals['audio_s'] / wall:.2f}", flush=True)


if __name__ == "__main__":                      # the decoding processes (spawn) import this file as __mp_main__
    main()
