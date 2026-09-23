# Build this fork for one GPU arch on Windows, with the same CUDA (12.8) and flags as the official
# wheel (python/tools/prepare_build_environment_windows.sh), minus the CPU backends (GPU-only use).
# Output: $Out\ctranslate2 (drop-in package: put $Out first on sys.path). The default is a staging
# directory, so a build never touches a package that a running process has loaded.
# OpenMP is required on Windows: without it every worker thread owns a thread_local BS::thread_pool
# whose destructor joins threads during thread exit, under the loader lock, and deadlocks teardown.
param([string]$Root = 'D:\ct2build', [string]$Arch = '7.5', [string]$Python = 'D:\wsbench-tmp\venv\Scripts\python.exe',
      [string]$Out = 'D:\ct2build\pyct2-next')
# Not 'Stop': Windows PowerShell 5.1 turns any stderr line of a native tool (a cmake warning) into a
# terminating error. Native steps are checked through their exit codes instead.
function Check($what) {
  if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $what (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$Src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$T = "$Root\tools"
$env:PATH = "$T\git\cmd;$T\cmake-3.31.12-windows-x86_64\bin;$T\ninja;$Cuda\bin;$env:PATH"
# Newest Visual Studio: the official wheels use VS 2022, and MSVC 19.27 miscompiles pybind11 2.11.
$vcvars = (Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
           Sort-Object { [int][regex]::Match($_.FullName, 'Visual Studio\\(\d{4})').Groups[1].Value } -Descending |
           Select-Object -First 1).FullName
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] } }
Write-Output "MSVC $env:VCToolsVersion from $vcvars"
$Build = "$Root\build-sm$($Arch.Replace('.', ''))-msvc$env:VCToolsVersion-omp"; $Inst = "$Root\install"
if (-not (Test-Path "$Build\build.ninja")) {
  $fwd = { param($p) $p.Replace('\', '/') }         # CMake reads backslashes in paths as escapes
  cmake -S $Src -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(& $fwd $Inst)" `
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_CLI=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP `
    -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_TOOLKIT_ROOT_DIR="$(& $fwd $Cuda)" -DCUDA_DYNAMIC_LOADING=ON `
    -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="$Arch" 2>&1 | Out-String -Stream
  Check 'cmake configure'
}
cmake --build $Build --target install --parallel 2>&1 | Out-String -Stream
Check 'cmake build'
# Python extension against the fresh library, packaged next to its DLL.
$env:CTRANSLATE2_ROOT = $Inst
$env:PYTHONPATH = "$Root\pydeps"
Push-Location "$Src\python"
& $Python setup.py build_ext --inplace --force 2>&1 | Out-String -Stream
Check 'python extension'
Pop-Location
$Pkg = "$Out\ctranslate2"
if (Test-Path $Pkg) { Remove-Item $Pkg -Recurse -Force }
Copy-Item "$Src\python\ctranslate2" $Pkg -Recurse
Copy-Item "$Inst\bin\ctranslate2.dll" $Pkg
# The OpenMP runtime ships next to the DLL, as the official wheel does with libiomp5md.dll.
$omp = Get-ChildItem $env:VCToolsRedistDir -Recurse -Filter vcomp140.dll |
       Where-Object { $_.FullName -like '*\x64\*OpenMP*' } | Select-Object -First 1
if (-not $omp) { Write-Output "FAILED: vcomp140.dll not found under $env:VCToolsRedistDir"; exit 1 }
Copy-Item $omp.FullName $Pkg
git -C $Src log --oneline -1 | Set-Content "$Pkg\BUILD.txt"
Write-Output "built $Pkg from $(git -C $Src rev-parse --short HEAD)"
