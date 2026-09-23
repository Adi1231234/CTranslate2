"""CUPTI activity records through ctypes (cupti-python is Linux-only) - the data Nsight Systems shows.
Kernel records: (name, stream, start_ns, end_ns, grid (x, y, z), block (x, y, z), dynamic shared memory).
API records (with api=True): (name, thread_id, start_ns, end_ns) for every CUDA driver/runtime call."""
import os, ctypes, zipfile, urllib.request
from ctypes import CFUNCTYPE, POINTER, byref, c_char_p, c_size_t, c_uint8, c_uint32, c_int32, c_uint64, c_void_p

WHL = ("https://files.pythonhosted.org/packages/1c/81/7796f096afaf726796b1b648f3bc80cafc61fe7f77f44a483c89e6c5ef34/"
       "nvidia_cuda_cupti_cu12-12.6.80-py3-none-win_amd64.whl")
KIND_DRIVER, KIND_RUNTIME, KIND_CONCURRENT_KERNEL, BUF = 4, 5, 10, 8 << 20
REQ = CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(c_size_t))
DONE = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(c_uint8), c_size_t, c_size_t)


def _i32(a, off):
    return c_int32.from_address(a + off).value


def _u32(a, off):
    return c_uint32.from_address(a + off).value


def _u64(a, off):
    return c_uint64.from_address(a + off).value


class Tracer:
    def __init__(self, cache_dir, api=False):
        dll = os.path.join(cache_dir, "cupti64_2024.3.2.dll")
        if not os.path.exists(dll):
            os.makedirs(cache_dir, exist_ok=True)
            whl = os.path.join(cache_dir, "cupti.whl")
            urllib.request.urlretrieve(WHL, whl)
            with zipfile.ZipFile(whl) as z:
                for n in z.namelist():
                    if n.endswith(".dll"):
                        open(os.path.join(cache_dir, os.path.basename(n)), "wb").write(z.read(n))
        os.add_dll_directory(cache_dir)
        self.lib = ctypes.CDLL(dll)
        self.api = api
        self.bufs, self.records, self.api_records, self._names = {}, [], [], {}
        self._req, self._done = REQ(self._requested), DONE(self._completed)   # keep the callbacks alive
        assert self.lib.cuptiActivityRegisterCallbacks(self._req, self._done) == 0

    def _requested(self, buf_pp, size_p, maxrec_p):
        b = (c_uint64 * (BUF // 8))()                           # 8-byte aligned, as CUPTI requires
        self.bufs[ctypes.addressof(b)] = b
        buf_pp[0] = ctypes.cast(b, POINTER(c_uint8)); size_p[0] = BUF; maxrec_p[0] = 0

    def _api_name(self, kind, cbid):                            # CUpti_CallbackDomain: 1 driver, 2 runtime
        key = (kind, cbid)
        if key not in self._names:
            name = c_char_p()
            ok = self.lib.cuptiGetCallbackName(1 if kind == KIND_DRIVER else 2, cbid, byref(name)) == 0
            self._names[key] = name.value.decode() if ok and name.value else f"cbid{cbid}"
        return self._names[key]

    def _completed(self, ctx, stream, buf, size, valid):
        rec = c_void_p()
        while self.lib.cuptiActivityGetNextRecord(buf, c_size_t(valid), byref(rec)) == 0:
            a = rec.value
            kind = _u32(a, 0)
            if kind == KIND_CONCURRENT_KERNEL:                  # CUpti_ActivityKernel9 offsets
                name = ctypes.string_at(c_void_p.from_address(a + 104).value).decode(errors="replace")
                self.records.append((name, _u32(a, 48), _u64(a, 16), _u64(a, 24),
                                     (_i32(a, 52), _i32(a, 56), _i32(a, 60)),
                                     (_i32(a, 64), _i32(a, 68), _i32(a, 72)), _i32(a, 80)))
            elif kind in (KIND_DRIVER, KIND_RUNTIME):           # CUpti_ActivityAPI offsets
                self.api_records.append((("drv " if kind == KIND_DRIVER else "rt ") + self._api_name(kind, _u32(a, 4)),
                                         _u32(a, 28), _u64(a, 8), _u64(a, 16)))
        self.bufs.pop(ctypes.addressof(buf.contents), None)

    def start(self):
        assert self.lib.cuptiActivityEnable(KIND_CONCURRENT_KERNEL) == 0
        if self.api:
            assert self.lib.cuptiActivityEnable(KIND_DRIVER) == 0
            assert self.lib.cuptiActivityEnable(KIND_RUNTIME) == 0

    def stop(self):
        self.lib.cuptiActivityFlushAll(1)

    def now(self):
        t = c_uint64(); self.lib.cuptiGetTimestamp(byref(t)); return t.value


def short(n, width=90):
    n = n.replace("void ", "")
    return (n.split("(")[0].split("<")[0] + ("<" + n.split("<")[1][:40] if "<" in n else ""))[:width]
