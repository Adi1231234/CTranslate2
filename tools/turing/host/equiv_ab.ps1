# A/B of prod_equiv.py (the production engine on the 150 sample clips) between environment settings:
# rounds alternate the configurations, each run is its own process with a time limit. For an idle GPU
# only: it does not pause production. Log: D:\ct2build\ab.log
# usage: equiv_ab.ps1 -Configs 'base=CPU_THREADS=1', 'retain=CPU_THREADS=1;POOL_RETAIN=1' [-Pkg dir] [-Rounds n]
param([string[]]$Configs = @('base=CPU_THREADS=1'), [string]$Pkg = 'D:\ct2build\pyct2-next',
      [string]$Mode = 'pipe8', [int]$Rounds = 2, [int]$Limit = 300)
. "$PSScriptRoot\prod.ps1"
& $Git -C $Src pull -q --ff-only 2>&1 | Out-Null
Log ("---- equiv_ab at " + (& $Git -C $Src log --oneline -1) + " pkg $Pkg mode $Mode")
for ($i = 0; $i -lt $Rounds; $i++) {
  foreach ($c in $Configs) {
    $label, $vars = $c -split '=', 2
    $set = @($vars -split ';' | Where-Object { $_ })
    foreach ($v in $set) { $k, $val = $v -split '=', 2; Set-Item "env:$k" $val }
    try { Invoke-Timed $label @("$Src\tools\turing\prod_equiv.py", "$W\sample", $W, $Mode, $Pkg) $Limit }
    finally { foreach ($v in $set) { Remove-Item ("env:" + ($v -split '=', 2)[0]) -ErrorAction SilentlyContinue } }
  }
}
