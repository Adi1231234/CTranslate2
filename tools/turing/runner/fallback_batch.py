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
RUN_FALLBACK_THREADS (default 8): clips whose ladders run at once."""
import contextlib, hashlib, threading, time
import numpy as np
import ctranslate2

_ladder = threading.local()                                 # the clip whose ladder runs on this thread


@contextlib.contextmanager
def ladder_of(uuid):
    """While a clip's ladder runs on this thread: its sampled calls take the clip's seeds, in call order."""
    _ladder.uuid, _ladder.calls = uuid, 0
    try:
        yield
    finally:
        _ladder.uuid = None


def _next_seed():
    """The seed of the running ladder's next sampled call: its clip and the call's number in the ladder."""
    n = _ladder.calls
    _ladder.calls += 1
    digest = hashlib.blake2b(f"{_ladder.uuid}|{n}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


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

    def __init__(self, model, join_sampled, seeded=False, wait_s=0.2):
        self._m, self._join_sampled, self._seeded, self._wait = model, join_sampled, seeded, wait_s
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
        key = ("generate", tuple(prompts[0]), tuple(sorted((k, str(v)) for k, v in kw.items())))
        if sampled and not self._join_sampled:
            key += (object(),)                                 # alone, as faster-whisper calls it
        call = _Call("generate", key, encoder_output.array, prompts[0])
        call.seed = _next_seed() if sampled and self._seeded else None
        return self._call(call, kw)

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

    def _run(self, group):
        data = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.concatenate([c.data for c in group])))
        if group[0].kind == "encode":
            out = np.asarray(self._m.encode(data, to_cpu=True, group_size=1))
            for i, c in enumerate(group):
                c.result = np.array(out[i:i + 1])
        else:
            seeds = {} if group[0].seed is None else {"sampling_seeds": [c.seed for c in group]}
            results = self._m.generate(data, [c.prompt for c in group], group_size=1, **group[0].kw, **seeds)
            for c, r in zip(group, results):
                c.result = [r]
