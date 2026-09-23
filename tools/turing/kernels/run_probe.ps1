# Build one probe with nvcc for sm_75 and run it against the cuBLAS that production loads (the venv's
# nvidia-cublas wheel, first on PATH as in bench_whisper.py). usage: run_probe.ps1 <name> [args...]
param([string]$Name, [string]$Root = 'D:\ct2build', [string]$Venv = 'D:\wsbench-tmp\venv',
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
. "$PSScriptRoot\..\devenv.ps1"
$Out = "$Root\probes"
New-Item -ItemType Directory -Force $Out | Out-Null
nvcc -O3 -std=c++17 -arch=sm_75 -o "$Out\$Name.exe" "$PSScriptRoot\$Name.cu" -lcublas 2>&1 | Out-String -Stream
Check "nvcc $Name"
$env:PATH = "$Venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"
& "$Out\$Name.exe" @Rest
