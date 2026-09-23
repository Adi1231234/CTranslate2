# Dot-source: the build environment of this host (CUDA 12.8, newest Visual Studio, CMake, Ninja, git
# from $Root\tools) plus Check, the exit-code test for native steps. Needs $Root set by the caller.
# Not 'Stop': Windows PowerShell 5.1 turns any stderr line of a native tool (a cmake warning) into a
# terminating error. Native steps are checked through their exit codes instead.
function Check($what) {
  if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $what (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$T = "$Root\tools"
$env:PATH = "$T\git\cmd;$T\cmake-3.31.12-windows-x86_64\bin;$T\ninja;$Cuda\bin;$env:PATH"
# Newest Visual Studio: the official wheels use VS 2022, and MSVC 19.27 miscompiles pybind11 2.11.
$vcvars = (Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
           Sort-Object { [int][regex]::Match($_.FullName, 'Visual Studio\\(\d{4})').Groups[1].Value } -Descending |
           Select-Object -First 1).FullName
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] } }
Write-Output "MSVC $env:VCToolsVersion from $vcvars"
