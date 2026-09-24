# The inner development loop for one change: pull, build into pyct2-next, the fast digest check
# against the stock golden file, and only if it passes a speed A/B of the production engine
# (prod_equiv.py, CPU_THREADS=1, every run's rows_sha and GPU kernel time logged) between -Base and the new build:
# a discarded warm-up run, then base/next pairs and a verdict line (verdict.ps1). One pair judged on GPU busy
# time sees a 1% kernel change; -WallDelta <percent> judges on wall time instead (host-side and overlap changes)
# with as many pairs as that change needs. For an idle GPU only. Log: D:\ct2build\ab.log.
# Start it detached so it survives the remote session.
# usage: iterate.ps1 [-Base D:\ct2build\pyct2-base | stock] [-Rounds 1] [-WallDelta 2] [-NoBuild] [-Mode pipe8]
param([string]$Base = 'D:\ct2build\pyct2-base', [int]$Rounds = 1, [double]$WallDelta = 0, [switch]$NoBuild,
      [string]$Mode = 'pipe8')
. "$PSScriptRoot\prod.ps1"
. "$PSScriptRoot\verdict.ps1"
$T = "$Src\tools\turing"; $Next = "$R\pyct2-next"
& $Git -C $Src pull -q --ff-only 2>&1 | Out-Null
Log ("---- iterate at " + (& $Git -C $Src log --oneline -1))
if (-not $NoBuild) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$T\build_windows.ps1" *> "$R\build.log"
  $rc = $LASTEXITCODE
  Log ("build exit ${rc}: " + (Get-Content "$R\build.log" -Tail 1))
  if ($rc -ne 0) { exit 1 }
}
$line = Invoke-Timed 'digest' @("$T\digest.py", "$W\sample", $Next, '--golden', "$R\golden_digest.json") 300
if ($line -notlike '*"digest": "PASS"*') { Log 'iterate stop: digest did not pass'; exit 1 }
$metric = 'gpu'
if ($WallDelta -gt 0) { $metric = 'wall'; $Rounds = Get-RoundsFor 'wall' $WallDelta }
$env:CPU_THREADS = '1'; $env:GPU_TIME = '1'
$null = Invoke-Timed 'warmup' @("$T\prod_equiv.py", "$W\sample", $W, $Mode, $Next) 300
$base = @(); $next = @()
for ($i = 0; $i -lt $Rounds; $i++) {
  $base += Invoke-Timed 'base' (@("$T\prod_equiv.py", "$W\sample", $W, $Mode) + @(PkgArg $Base)) 300
  $next += Invoke-Timed 'next' @("$T\prod_equiv.py", "$W\sample", $W, $Mode, $Next) 300
}
Remove-Item env:CPU_THREADS, env:GPU_TIME
Log ('verdict    ' + (Get-AbVerdict $base $next $metric))
Log '---- iterate done'
