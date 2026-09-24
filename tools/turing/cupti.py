"""CUPTI kernel activity records through ctypes (cupti-python is Linux-only).
Records: (name, stream, start_ns, end_ns, grid (x, y, z), block (x, y, z), dynamic shared memory),
or with names=False only (start_ns, end_ns), cheap enough for a whole production run."""
import os, ctypes, zipfile, urllib.request
from ctypes import CFUNCTYPE, POINTER, byref, c_size_t, c_uint8, c_uint32, c_int32, c_uint64, c_void_p

PYPI = "https://files.pythonhosted.org/packages/"
# CUPTI 12.6 answers CUPTI_ERROR_INVALID_DEVICE on Blackwell once a context exists; 12.8 is the first with it.
WHEELS = {"12.6.80": ("1c/81/7796f096afaf726796b1b648f3bc80cafc61fe7f77f44a483c89e6c5ef34/"
                      "nvidia_cuda_cupti_cu12-12.6.80-py3-none-win_amd64.whl", "cupti64_2024.3.2.dll"),
          "12.8.90": ("41/bc/83f5426095d93694ae39fe1311431b5d5a9bb82e48bf0dd8e19be2765942/"
                      "nvidia_cuda_cupti_cu12-12.8.90-py3-none-win_amd64.whl", "cupti64_2025.1.1.dll")}
KIND_CONCURRENT_KERNEL, BUF = 10, 8 << 20
REQ = CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(c_size_t))
DONE = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(c_uint8), c_size_t, c_size_t)


def _i32(a, off):
    return c_int32.from_address(a + off).value


def _version(device=0):
    """12.8 on compute capability 10+ (Blackwell); 12.6, which every Turing measurement used, below."""
    cu, dev, major = ctypes.WinDLL("nvcuda.dll"), c_int32(), c_int32()
    assert cu.cuInit(0) == 0 and cu.cuDeviceGet(byref(dev), device) == 0
    assert cu.cuDeviceGetAttribute(byref(major), 75, dev) == 0      # COMPUTE_CAPABILITY_MAJOR
    return "12.8.90" if major.value >= 10 else "12.6.80"


class Tracer:
    def __init__(self, cache_dir, names=True):
        ver = _version()
        path, name = WHEELS[ver]
        cache_dir = os.path.join(cache_dir, ver)                       # versions share DLL names
        dll = os.path.join(cache_dir, name)
        if not os.path.exists(dll):
            os.makedirs(cache_dir, exist_ok=True)
            whl = os.path.join(cache_dir, "cupti.whl")
            urllib.request.urlretrieve(PYPI + path, whl)
            with zipfile.ZipFile(whl) as z:
                for n in z.namelist():
                    if n.endswith(".dll"):
                        open(os.path.join(cache_dir, os.path.basename(n)), "wb").write(z.read(n))
        os.add_dll_directory(cache_dir)
        self.lib = ctypes.CDLL(dll)
        self.bufs, self.records, self.names = {}, [], names
        self._req, self._done = REQ(self._requested), DONE(self._completed)   # keep the callbacks alive
        self._check(self.lib.cuptiActivityRegisterCallbacks(self._req, self._done), "cuptiActivityRegisterCallbacks")

    def _check(self, rc, what):
        if rc != 0:
            s = ctypes.c_char_p()
            self.lib.cuptiGetResultString(rc, byref(s))
            raise RuntimeError(f"{what}: {rc} {s.value.decode() if s.value else ''}")

    def _requested(self, buf_pp, size_p, maxrec_p):
        b = (c_uint64 * (BUF // 8))()                           # 8-byte aligned, as CUPTI requires
        self.bufs[ctypes.addressof(b)] = b
        buf_pp[0] = ctypes.cast(b, POINTER(c_uint8)); size_p[0] = BUF; maxrec_p[0] = 0

    def _completed(self, ctx, stream, buf, size, valid):
        rec = c_void_p()
        while self.lib.cuptiActivityGetNextRecord(buf, c_size_t(valid), byref(rec)) == 0:
            a = rec.value                                       # CUpti_ActivityKernel9 offsets
            if c_uint32.from_address(a).value != KIND_CONCURRENT_KERNEL:
                continue
            if not self.names:
                self.records.append((c_uint64.from_address(a + 16).value, c_uint64.from_address(a + 24).value))
            else:
                name = ctypes.string_at(c_void_p.from_address(a + 104).value).decode(errors="replace")
                self.records.append((name, c_uint32.from_address(a + 48).value,
                                     c_uint64.from_address(a + 16).value, c_uint64.from_address(a + 24).value,
                                     (_i32(a, 52), _i32(a, 56), _i32(a, 60)),
                                     (_i32(a, 64), _i32(a, 68), _i32(a, 72)), _i32(a, 80)))
        self.bufs.pop(ctypes.addressof(buf.contents), None)

    def start(self):
        self._check(self.lib.cuptiActivityEnable(KIND_CONCURRENT_KERNEL), "cuptiActivityEnable")

    def stop(self):
        self.lib.cuptiActivityFlushAll(1)

    def now(self):
        t = c_uint64(); self.lib.cuptiGetTimestamp(byref(t)); return t.value


def short(n, width=90):
    n = n.replace("void ", "")
    return (n.split("(")[0].split("<")[0] + ("<" + n.split("<")[1][:40] if "<" in n else ""))[:width]
