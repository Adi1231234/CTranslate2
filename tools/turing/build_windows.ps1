# Build this fork for one GPU arch on Windows, with the same CUDA (12.8) and flags as the official
# wheel (python/tools/prepare_build_environment_windows.sh), minus the CPU backends (GPU-only use).
# Output: $Root\pyct2\ctranslate2 (drop-in package: put $Root\pyct2 first on sys.path).
param([string]$Root = 'D:\ct2build', [string]$Arch = '7.5', [string]$Python = 'D:\wsbench-tmp\venv\Scripts\python.exe')
# Not 'Stop': Windows PowerShell 5.1 turns any stderr line of a native tool (a cmake warning) into a
# terminating error. Native steps are checked through their exit codes instead.
function Check($what) {
  if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $what (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$Src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$T = "$Root\tools"
$env:PATH = "$T\git\cmd;$T\cmake-3.31.12-windows-x86_64\bin;$T\ninja;$Cuda\bin;$env:PATH"
$vcvars = (Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' | Select-Object -First 1).FullName
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] } }
$Build = "$Root\build-sm$($Arch -replace '\.','')"; $Inst = "$Root\install"
if (-not (Test-Path "$Build\build.ninja")) {
  cmake -S $Src -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$Inst" `
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_CLI=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE `
    -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_TOOLKIT_ROOT_DIR="$Cuda" -DCUDA_DYNAMIC_LOADING=ON `
    -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="$Arch" 2>&1 | Out-String -Stream
  Check 'cmake configure'
}
cmake --build $Build --target install --parallel 2>&1 | Out-String -Stream
Check 'cmake build'
# Python extension against the fresh library, packaged next to its DLL.
$env:CTRANSLATE2_ROOT = $Inst
$env:PYTHONPATH = "$Root\pydeps"
Push-Location "$Src\python"
& $Python setup.py build_ext --inplace 2>&1 | Out-String -Stream
Check 'python extension'
Pop-Location
$Pkg = "$Root\pyct2\ctranslate2"
if (Test-Path $Pkg) { Remove-Item $Pkg -Recurse -Force }
Copy-Item "$Src\python\ctranslate2" $Pkg -Recurse
Copy-Item "$Inst\bin\ctranslate2.dll" $Pkg
Write-Output "built $Pkg from $(git -C $Src rev-parse --short HEAD)"
