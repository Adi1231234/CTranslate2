# Pull the fork and build it into pyct2-next while production keeps running, then with production
# paused A/B it against the deployed build (and the stock wheel with -Stock). Log: D:\ct2build\ab.log
param([switch]$NoBuild, [switch]$Stock, [int]$Rounds = 2)
. "$PSScriptRoot\prod.ps1"
Log '---- ab start'
if (-not $NoBuild) {
  Sync-Checkout $PSCommandPath $PSBoundParameters
  Log ('commit ' + (& $Git -C $Src log --oneline -1))
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$Src\tools\turing\build_windows.ps1" *> "$R\build.log"
  $rc = $LASTEXITCODE
  Log ("build exit ${rc}: " + (Get-Content "$R\build.log" -Tail 1))
  if ($rc -ne 0) { exit 1 }
}
Suspend-Production
try {
  $b = "$Src\tools\turing\bench_whisper.py"
  for ($i = 0; $i -lt $Rounds; $i++) {
    if ($Stock) { Invoke-Timed 'stock' @($b, "$W\sample") 180 }
    Invoke-Timed 'deployed' @($b, "$W\sample", "$R\pyct2") 180
    Invoke-Timed 'next' @($b, "$W\sample", "$R\pyct2-next") 180
  }
} finally { Resume-Production }
