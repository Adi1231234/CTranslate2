# Shared helpers (dot-source): pause/resume the crowd-v5 production run around GPU experiments, and
# run one python tool with a hard time limit so production is always resumed.
# Host layout: this clone lives in $R\CTranslate2 next to $R\tools (git, cmake, ninja) and the builds;
# $W holds the production run (supervise.ps1, transcribe_run.py, the venv with the stock wheel, sample):
# D:\wsbench-tmp on Yarin, or the folder named in $R\work_dir.txt (the store PC: C:\Windows\Temp\wsbench).
$Src = (Resolve-Path "$PSScriptRoot\..\..\..").Path; $R = Split-Path -Parent $Src
$W = if (Test-Path "$R\work_dir.txt") { (Get-Content "$R\work_dir.txt" -TotalCount 1).Trim() } else { 'D:\wsbench-tmp' }
$Py = "$W\venv\Scripts\python.exe"
$Git = if (Test-Path "$R\tools\git\cmd\git.exe") { "$R\tools\git\cmd\git.exe" } else { 'git' }
$env:HF_HOME = "$W\hf"
function Log($m) { Add-Content -Path "$R\ab.log" -Value "$(Get-Date -Format HH:mm:ss) $m" }
# Pulls this clone. PowerShell has already parsed the running script (and this file), so when the pull moves
# HEAD the fresh copy runs instead: usage `Sync-Checkout $PSCommandPath $PSBoundParameters`.
function Sync-Checkout($script, $params) {
  $before = & $Git -C $Src rev-parse HEAD
  & $Git -C $Src pull -q --ff-only 2>&1 | Out-Null
  if ((& $Git -C $Src rev-parse HEAD) -ne $before) { & $script @params; exit $LASTEXITCODE }
}
# A ctranslate2 package for the python tools: a build's parent dir, or 'stock' for the venv's own wheel.
function PkgArg($p) { if ($p -ne 'stock') { $p } }                 # wrap in @(): 'stock' adds no argument
function PkgLabel($p) { if ($p -eq 'stock') { 'stock' } else { Split-Path $p -Leaf } }
function Suspend-Production {
  $ids = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like '*supervise.ps1*' }) +
         @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*transcribe_run*' }) |
         ForEach-Object { $_.ProcessId }
  foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
  foreach ($id in $ids) { Wait-Process -Id $id -Timeout 60 -ErrorAction SilentlyContinue }   # files and VRAM released
  Remove-Item "$W\out\*.tmp" -ErrorAction SilentlyContinue
  Log 'paused production'
}
function Resume-Production {
  Start-Process -FilePath 'powershell.exe' -WorkingDirectory $W -WindowStyle Hidden -ArgumentList '-NoProfile', '-NonInteractive',
    '-ExecutionPolicy', 'Bypass', '-File', "$W\supervise.ps1", '-root', $W, '-dir', 'back', '-mode', 'pipe8', '-pythonpath', "$R\pyct2"
  Log 'resumed production'
}
# Runs python with $argv, killing it after $limit s; logs and returns its last JSON line (or the tail of stdout).
function Invoke-Timed($label, $argv, $limit) {
  $p = Start-Process -FilePath $Py -ArgumentList $argv -PassThru -WindowStyle Hidden -RedirectStandardOutput "$R\run.out" -RedirectStandardError "$R\run.err"
  $null = $p.Handle
  $done = $p.WaitForExit($limit * 1000)
  if (-not $done) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
  $line = Get-Content "$R\run.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like '{*' } | Select-Object -Last 1
  if (-not $line) { $line = (Get-Content "$R\run.out" -Tail 2 -ErrorAction SilentlyContinue) -join ' | ' }
  $err = if ($done -and $p.ExitCode -ne 0) { ' ERR: ' + ((Get-Content "$R\run.err" -Tail 3 -ErrorAction SilentlyContinue) -join ' | ') } else { '' }
  Log ("{0,-10} {1} exit={2}{3}" -f $label, $line, $(if ($done) { $p.ExitCode } else { 'KILLED' }), $err)
  $line
}
