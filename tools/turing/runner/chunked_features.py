"""faster-whisper's FeatureExtractor.__call__ (1.2.1) in less memory, for recordings of many hours (a 24 h Knesset
plenum: 8.6 million frames; the original holds the padded waveform, the complex STFT and the power at once, ~26 GB).
The FFT and the power of each frame do not depend on the other frames, so they run a block of frames at a time, into
one power array laid out as the original's (frames by 201 bins, read transposed). The mel product does depend on its
shape (OpenBLAS sums a column differently in a call of another width: ../scale/features_check.py found it), so it is
the original's one call on the whole recording, then the same elementwise steps, in place. Every feature is the
original's bytes on the same machine (checked by features_check.py on the box's numpy).
LONG_FEATURE_FRAMES (default 360000, an hour): frames a block."""
import os
import numpy as np


class ChunkedFeatures:
    def __init__(self, extractor):
        self.extractor = extractor
        self.frames = int(os.environ.get("LONG_FEATURE_FRAMES", "360000"))

    def __getattr__(self, name):                   # sampling_rate, chunk_length, ... of the real extractor
        return getattr(self.extractor, name)

    @staticmethod
    def padded(w, lo, hi, n):
        """[lo, hi) of the original's padded waveform: w, 160 zeros, then n samples reflected at each end."""
        ext = len(w) + 160
        out = np.empty(hi - lo, np.float32)
        a, b = max(lo, n), min(hi, len(w) + n)
        if a < b:
            out[a - lo:b - lo] = w[a - n:b - n]
        for p_lo, p_hi in ((lo, min(hi, n)), (max(lo, len(w) + n), hi)):     # the ends: reflection, zeros
            if p_lo < p_hi:
                p = np.arange(p_lo, p_hi)
                j = np.where(p < n, n - p, np.where(p < ext + n, p - n, 2 * ext + n - 2 - p))
                out[p_lo - lo:p_hi - lo] = np.where(j < len(w), w[np.minimum(j, len(w) - 1)], np.float32(0))
        return out

    def __call__(self, waveform, padding=160, chunk_length=None):
        fe = self.extractor
        if padding != 160:
            raise ValueError("ChunkedFeatures pads as faster-whisper's transcribe does: 160 samples")
        if chunk_length is not None:               # the original's side effects on the extractor
            fe.n_samples = chunk_length * fe.sampling_rate
            fe.nb_max_frames = fe.n_samples // fe.hop_length
        if waveform.dtype is not np.float32:
            waveform = waveform.astype(np.float32)
        n_fft, hop = fe.n_fft, fe.hop_length
        window = np.hanning(n_fft + 1)[:-1].astype("float32")
        frames = (len(waveform) + 160) // hop      # the STFT's frames but its last, which the original drops
        power = np.empty((frames, n_fft // 2 + 1), np.float32)
        for a in range(0, frames, self.frames):
            b = min(a + self.frames, frames)
            block = self.padded(waveform, a * hop, (b - 1) * hop + n_fft, n_fft // 2)
            stft = fe.stft(block, n_fft, hop, window=window, center=False, return_complex=True).astype("complex64")
            power[a:b] = np.abs(stft.T) ** 2
        log_spec = fe.mel_filters @ power.T
        del power
        np.clip(log_spec, a_min=1e-10, a_max=None, out=log_spec)
        np.log10(log_spec, out=log_spec)
        np.maximum(log_spec, log_spec.max() - 8.0, out=log_spec)
        log_spec += 4.0
        log_spec /= 4.0
        return log_spec
