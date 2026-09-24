# Restarts transcribe_run.py on crash (exit != 0) or hang (no progress.log write for 20 min).
param([string]$root, [string]$dir, [string]$mode, [string]$pythonpath = '')
if ($pythonpath) { $env:PYTHONPATH = $pythonpath }   # e.g. a ctranslate2 build to import first
$py = "$root\venv\Scripts\python.exe"
function Log($m) { Add-Content -Path "$root\progress.log" -Value "$(Get-Date -Format HH:mm:ss) [supervisor] $m" -Encoding UTF8 }
for ($i = 0; $i -lt 30; $i++) {
  $p = Start-Process -FilePath $py -ArgumentList "$root\transcribe_run.py",$dir,$mode -WorkingDirectory $root `
       -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root\run.out.$i" -RedirectStandardError "$root\run.err.$i"
  $null = $p.Handle   # open the handle now, or ExitCode reads back empty and a clean exit restarts
  while (-not $p.WaitForExit(60000)) {
    $age = ((Get-Date) - (Get-Item "$root\progress.log").LastWriteTime).TotalMinutes
    if ($age -gt 20) { Log "watchdog: no progress for $([int]$age) min, killing pid $($p.Id)"; Stop-Process -Id $p.Id -Force; break }
  }
  $p.WaitForExit()
  if ($p.ExitCode -eq 0) { Log "clean exit"; break }
  Log "exit code $($p.ExitCode), restart #$($i + 1) in 60s"
  Start-Sleep -Seconds 60
}
