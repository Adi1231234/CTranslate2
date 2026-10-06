"""RUN_FALLBACK=batched: the fallback clips' ladders (faster-whisper's own sequential transcribe, unchanged) run on
several threads whose CTranslate2 calls go through a broker. Calls of different clips that CTranslate2 can run
together (the same options and the same prompt) go to the GPU as one call with group_size=1: every clip keeps the
arithmetic of a call of its own (cuda/clip_groups.h), and the decoder reads its weights once for all of them.
The beam attempts (T=0) are then exactly what each clip gets alone. Sampled attempts (T>0) draw other random
numbers when joined (a row's random state is its place in the batch), as they do on any other run; they are joined
only with RUN_FALLBACK_SAMPLING=batched, otherwise each runs alone as before.
RUN_FALLBACK_SEEDS=1: every sampled attempt draws from streams seeded by its clip and its place in the clip's ladder
(CTranslate2's sampling_seeds, the ladder-probe fork): a clip then samples the same alone or joined, and on every
run, so joined sampled attempts are exactly what each clip gets alone too.
RUN_FALLBACK_THREADS (default 8): clips whose ladders run at once.
RUN_FALLBACK_SPECULATE=1 (with the seeds and RUN_FALLBACK_SAMPLING=batched): a window's first sampled attempt runs
all of the ladder's sampled temperatures from it on in one search (CTranslate2's sampling_temperatures, the
ladder-probe fork), each with the seed its own call would draw, and the ladder's next calls take theirs from it:
the window's attempts then read the decoder weights and the clip's memory keys and values together, not one
temperature after the other. Every attempt is what its call alone returns, so the ladder (faster-whisper's own
logic) picks the same; attempts past the one it keeps are spent for nothing. RUN_FALLBACK_SPEC_CLIPS (default 2):
clips in one such search (each holds 25 rows of up to 448 steps of self-attention cache, ~73 MB a row).
RUN_FALLBACK_SPEC_FIRST=<k> (default 0): the ladder's first k sampled temperatures each run alone, the rest together
from the one after them: on long recordings a fifth of the windows that sample keep their first attempt (T=0.2), and
the four later temperatures' rows, the longest (high temperatures run to the length limit), were spent for nothing.
Each attempt keeps its seed (its place in the ladder), so what each returns does not change."""
import contextlib, hashlib, threading, time
import numpy as np
import ctranslate2

_ladder = threading.local()                                 # the clip whose ladder runs on this thread


@contextlib.contextmanager
def ladder_of(uuid):
    """While a clip's ladder runs on this thread: its sampled calls take the clip's seeds, in call order."""
    _ladder.uuid, _ladder.calls, _ladder.spec = uuid, 0, None
    try:
        yield
    finally:
        _ladder.uuid = _ladder.spec = None


def _seed(n):
    """The seed of the running ladder's sampled call number n: its clip and n."""
    digest = hashlib.blake2b(f"{_ladder.uuid}|{n}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _next_seed():
    """The seed of the running ladder's next sampled call (and its number in the ladder)."""
    n = _ladder.calls
    _ladder.calls += 1
    return _seed(n)


def kw_rest(kw):
    """A sampled call's options but its temperature (what a window's attempts share)."""
    return {k: v for k, v in kw.items() if k != "sampling_temperature"}


class _Encoded:
    """An encoder output kept on the host for the broker (faster-whisper only passes it back to generate)."""
    def __init__(self, array):
        self.array = array


class _Call:
    def __init__(self, kind, key, data, prompt=None):
        self.kind, self.key, self.data, self.prompt = kind, key, data, prompt
        self.done = threading.Event()
        self.result = self.error = None


class Broker:
    """Stands in for the ctranslate2 Whisper model of the fallback threads."""

    def __init__(self, model, join_sampled, seeded=False, wait_s=0.2, speculate=None, spec_clips=2, spec_first=0):
        """speculate: the ladder's sampled temperatures in order (RUN_FALLBACK_SPECULATE), or None; spec_first: how
        many of them run alone first (RUN_FALLBACK_SPEC_FIRST)."""
        if speculate and not (join_sampled and seeded):
            raise ValueError("speculated attempts need seeded, joined sampling: each must be what its call draws")
        self._m, self._join_sampled, self._seeded, self._wait = model, join_sampled, seeded, wait_s
        self._speculate, self._spec_clips, self._spec_first = speculate, spec_clips, spec_first
        self._cv = threading.Condition()
        self._pending, self._busy = [], 0
        threading.Thread(target=self._dispatch, daemon=True).start()

    def __getattr__(self, name):                             # device, device_index, is_multilingual, ...
        return getattr(self._m, name)

    def ladder_started(self):
        with self._cv:
            self._busy += 1

    def ladder_finished(self):
        with self._cv:
            self._busy -= 1
            self._cv.notify_all()

    def encode(self, features, to_cpu=False):
        return _Encoded(self._call(_Call("encode", ("encode",), np.asarray(features))))

    def generate(self, encoder_output, prompts, **kw):
        if len(prompts) != 1 or not isinstance(encoder_output, _Encoded):
            raise ValueError("the broker joins single-clip calls of the fallback ladder")
        sampled = kw.get("sampling_temperature", 1) > 0 and kw.get("beam_size", 5) == 1
        if sampled and self._speculate:
            return [self._speculated(encoder_output, prompts[0], kw)]
        key = ("generate", tuple(prompts[0]), tuple(sorted((k, str(v)) for k, v in kw.items())))
        if sampled and not self._join_sampled:
            key += (object(),)                                 # alone, as faster-whisper calls it
        call = _Call("generate", key, encoder_output.array, prompts[0])
        call.seed = _next_seed() if sampled and self._seeded else None
        return self._call(call, kw)

    def _speculated(self, encoder_output, prompt, kw):
        """A sampled attempt of a window whose attempts from it on were sampled together: the window's first one
        samples them all (each with the seed its own call takes), the later ones take theirs."""
        t = kw["sampling_temperature"]
        n, seed = _ladder.calls, _next_seed()
        spec = _ladder.spec
        if not (spec and spec["encoded"] is encoder_output and spec["prompt"] == prompt and spec["kw"] == kw_rest(kw)
                and spec["seeds"].get(t) == seed):
            temps = [x for x in self._speculate if x >= t]      # this attempt and the ladder's later ones
            if self._speculate.index(t) < self._spec_first:     # one of the first: alone
                temps = temps[:1]
            rest = kw_rest(kw)
            call = _Call("spec", ("spec", tuple(prompt), tuple(sorted((k, str(v)) for k, v in rest.items())),
                                  tuple(temps)), encoder_output.array, prompt)
            call.seeds = [_seed(n + i) for i in range(len(temps))]
            results = self._call(call, rest)
            spec = _ladder.spec = {"encoded": encoder_output, "prompt": list(prompt), "kw": rest,
                                   "results": dict(zip(temps, results)), "seeds": dict(zip(temps, call.seeds))}
        return spec["results"][t]

    def _call(self, call, kw=None):
        call.kw = kw
        with self._cv:
            self._pending.append(call)
            self._cv.notify_all()
        call.done.wait()
        if call.error:
            raise call.error
        return call.result

    def _dispatch(self):
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                first = time.monotonic()
                # Every busy ladder waiting here, or a ladder slow to come (its host work): run what is there.
                while len(self._pending) < self._busy and time.monotonic() - first < self._wait:
                    self._cv.wait(self._wait)
                calls, self._pending = self._pending, []
            groups = {}
            for c in calls:
                groups.setdefault(c.key, []).append(c)
            for group in groups.values():
                try:
                    self._run(group)
                except Exception as e:                       # the callers raise it
                    for c in group:
                        c.error = e
                for c in group:
                    c.done.set()

    def _run(self, group, model=None):
        """The group's calls on the broker's model, or on another instance of the same model (the same bits)."""
        m = model or self._m
        if group[0].kind == "spec":
            # Each clip's attempts at the window's temperatures (a result per clip and temperature, clip-major).
            temps = list(group[0].key[3])
            for first in range(0, len(group), self._spec_clips):
                chunk = group[first:first + self._spec_clips]
                data = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.concatenate([c.data for c in chunk])))
                results = m.generate(data, [c.prompt for c in chunk], group_size=1, sampling_temperatures=temps,
                                     sampling_seeds=[s for c in chunk for s in c.seeds], **chunk[0].kw)
                for i, c in enumerate(chunk):
                    c.result = results[i * len(temps):(i + 1) * len(temps)]
            return
        data = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.concatenate([c.data for c in group])))
        if group[0].kind == "encode":
            out = np.asarray(m.encode(data, to_cpu=True, group_size=1))
            for i, c in enumerate(group):
                c.result = np.array(out[i:i + 1])
        else:
            seeds = {} if group[0].seed is None else {"sampling_seeds": [c.seed for c in group]}
            results = m.generate(data, [c.prompt for c in group], group_size=1, **group[0].kw, **seeds)
            for c, r in zip(group, results):
                c.result = [r]
