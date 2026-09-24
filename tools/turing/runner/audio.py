"""Audio bytes of a dataset row -> 16 kHz mono float32, as the production runner decodes them."""
import io, os
import numpy as np, av

def audio_format(path):
    """The container the dataset declares (the audio file name's extension). Left to probe the
    content, FFmpeg took 3 valid crowd-v5 MP3s for raw VVC video (same probe score), with no audio
    stream; forcing the declared format decodes them, and changes nothing for the others."""
    ext = os.path.splitext(path or "")[1].lstrip(".").lower()
    return ext if ext in ("mp3", "wav", "flac", "ogg") else None

def decode(buf, fmt=None):
    with av.open(io.BytesIO(buf), format=fmt) as c:
        rs = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
        ch = [r.to_ndarray().reshape(-1) for fr in c.decode(c.streams.audio[0]) for r in rs.resample(fr)]
        ch += [r.to_ndarray().reshape(-1) for r in rs.resample(None)]
    return np.concatenate(ch).astype("float32") if ch else np.zeros(0, "float32")
