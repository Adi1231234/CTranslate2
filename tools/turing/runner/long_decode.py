"""Audio decoding off the transcribing process (LONG_DECODERS worker processes, default 3): faster-whisper's own
decode_audio, so the same samples, features and rows. In the transcribing process it is a Python loop over the file's
frames that holds the GIL (YODAS v3's webm: ~170x realtime on one core): long2 and long3's 48 recordings decoded
there, each whole before its cut, and their transcription mostly waited for the GIL (5 windows decoding at once of 48
recordings, 43x). A decoded recording is written as .npy to LONG_DECODE_DIR (default a temporary folder on the root
disk, not /dev/shm: a plenum of 24 h is 5.5 GB), read back and deleted (not memory-mapped: Windows keeps a mapped
file from being deleted, and the features copy the samples anyway)."""
import multiprocessing, os, tempfile, uuid
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _decode(path, out, cut):
    np.save(out, decode_first(path, cut) if cut else __import__("faster_whisper").decode_audio(path))
    return out


def decode_first(path, cut):
    """decode_audio(path)[:cut * 16000] without decoding the rest (measurement runs, LONG_SECONDS): its own frames,
    500,000-sample groups and resampler (faster_whisper.audio's helpers), in the same order, stopped a minute of
    output past the cut; a sample comes out of the resampler once and never changes, so the cut's are the same."""
    import io
    import av
    from faster_whisper import audio as fa
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    raw = io.BytesIO()
    with av.open(path, mode="r", metadata_errors="ignore") as container:
        groups = fa._group_frames(fa._ignore_invalid_frames(container.decode(audio=0)), 500000)
        for frame in fa._resample_frames(groups, resampler):
            raw.write(frame.to_ndarray())
            if raw.tell() >= (cut + 60) * 16000 * 2:
                break
    return (np.frombuffer(raw.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0)[:cut * 16000]


class Decoder:
    """loader(path, cut) gives load(): the recording's 16 kHz samples (its first `cut` seconds when cut > 0)."""

    def __init__(self):
        workers = int(os.environ.get("LONG_DECODERS", "3"))
        # spawn: the transcribing process holds CUDA and threads, which a forked child must not inherit
        self.pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
                     if workers else None)
        self.dir = os.environ.get("LONG_DECODE_DIR") or tempfile.mkdtemp(prefix="long_decode_")

    def loader(self, path, cut=0):
        def load():
            if self.pool is None:
                from faster_whisper import decode_audio
                wav = decode_audio(path)
                return wav[:cut * 16000] if cut else wav
            out = self.pool.submit(_decode, path, os.path.join(self.dir, uuid.uuid4().hex + ".npy"), cut).result()
            wav = np.load(out)
            os.unlink(out)
            return wav
        return load
