import os, glob, sys
_base = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "nvidia"))
for d in glob.glob(os.path.join(_base, "*", "bin")):
    if os.path.isdir(d):
        os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
