# Build this fork for one GPU arch on Windows, with the same CUDA (12.8) and flags as the official
# wheel (python/tools/prepare_build_environment_windows.sh), minus the CPU backends (GPU-only use).
# Output: $Out\ctranslate2 (drop-in package: put $Out first on sys.path). The default is a staging
# directory, so a build never touches a package that a running process has loaded.
# OpenMP is required on Windows: without it every worker thread owns a thread_local BS::thread_pool
# whose destructor joins threads during thread exit, under the loader lock, and deadlocks teardown.
param([string]$Root = 'D:\ct2build', [string]$Arch = '7.5', [string]$Python = 'D:\wsbench-tmp\venv\Scripts\python.exe',
      [string]$Out = 'D:\ct2build\pyct2-next')
$Src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\devenv.ps1"
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
# Python extension against the fresh library, packaged next to its DLL. Not forced: setup.py lists the
# public headers in `depends`, so it is rebuilt only when they (or its sources) are newer than it. A change
# to the library internals leaves the bindings' ABI alone; forcing cost 64 s per build (Yarin, 24.9).
$env:CTRANSLATE2_ROOT = $Inst
$env:PYTHONPATH = "$Root\pydeps"
Push-Location "$Src\python"
& $Python setup.py build_ext --inplace 2>&1 | Out-String -Stream
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
