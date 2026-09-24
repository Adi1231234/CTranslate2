# Build one probe (build_probe.ps1, sm_75) and run it against the cuBLAS that production loads (the venv's
# nvidia-cublas wheel, first on PATH as in bench_whisper.py).
# usage: run_probe.ps1 <name> [-Include <dir>] [args...]
[CmdletBinding(PositionalBinding = $false)]   # probe arguments after the name go to $Rest, not to -Root
param([Parameter(Position = 0)][string]$Name, [string]$Root = 'D:\ct2build', [string]$Venv = 'D:\wsbench-tmp\venv',
      [string]$Include = '', [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$Exe = & "$PSScriptRoot\build_probe.ps1" $Name -Root $Root -Include $Include | Select-Object -Last 1
if ($LASTEXITCODE -ne 0) { Write-Output $Exe; exit $LASTEXITCODE }
$env:PATH = "$Venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"
& $Exe @Rest
