# Build one probe with nvcc for sm_75 and run it against the cuBLAS that production loads (the venv's
# nvidia-cublas wheel, first on PATH as in bench_whisper.py).
# usage: run_probe.ps1 <name> [-Include <dir>] [args...]
# -Include puts a directory before the library sources, so a probe can be built against a modified
# copy of a header (e.g. to show that it catches a known bug); the exe is then named <name>_alt.
param([string]$Name, [string]$Root = 'D:\ct2build', [string]$Venv = 'D:\wsbench-tmp\venv',
      [string]$Include = '', [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
. "$PSScriptRoot\..\devenv.ps1"
$Out = "$Root\probes"
$Src = (Resolve-Path "$PSScriptRoot\..\..\..").Path                 # probes include library headers
$Exe = if ($Include) { "$Out\${Name}_alt.exe" } else { "$Out\$Name.exe" }
$Inc = @(if ($Include) { '-I', $Include }) + @('-I', "$Src\src", '-I', "$Src\include")
New-Item -ItemType Directory -Force $Out | Out-Null
nvcc -O3 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -diag-suppress 2219 @Inc `
  -o $Exe "$PSScriptRoot\$Name.cu" -lcublas 2>&1 | Out-String -Stream
Check "nvcc $Name"
$env:PATH = "$Venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"
& $Exe @Rest
