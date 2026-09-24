# A/B of prod_equiv.py (the production engine on the 150 sample clips) between environment settings:
# rounds alternate the configurations, each run is its own process with a time limit, and nvidia-smi
# samples the GPU (memory, utilization, power, SM clock) during each run. For an idle GPU only: it
# does not pause production. Log: D:\ct2build\ab.log. Start it detached (Start-Process) so it survives
# the remote session, e.g.
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "& '<this file>' -Configs 'base=CPU_THREADS=1','retain=CPU_THREADS=1;POOL_RETAIN=1'"
# A configuration's MODE (e.g. 'b8=MODE=batch8') replaces -Mode for its runs. -Pkg 'stock' = the venv's wheel.
param([string[]]$Configs = @('base=CPU_THREADS=1'), [string]$Pkg = 'D:\ct2build\pyct2-next',
      [string]$Mode = 'pipe8', [int]$Rounds = 2, [int]$Limit = 300)
. "$PSScriptRoot\prod.ps1"
Sync-Checkout $PSCommandPath $PSBoundParameters
Log ("---- equiv_ab at " + (& $Git -C $Src log --oneline -1) + " pkg $Pkg mode $Mode")
$csv = "$R\smi_run.csv"
for ($i = 0; $i -lt $Rounds; $i++) {
  foreach ($c in $Configs) {
    $label, $vars = $c -split '=', 2
    $set = @($vars -split ';' | Where-Object { $_ })
    foreach ($v in $set) { $k, $val = $v -split '=', 2; Set-Item "env:$k" $val }
    Remove-Item $csv -ErrorAction SilentlyContinue
    $smi = Start-Process nvidia-smi -PassThru -WindowStyle Hidden -ArgumentList ('--query-gpu=memory.used,utilization.gpu,' +
      "power.draw,clocks.sm --format=csv,noheader,nounits -lms 500 -f $csv")
    $m = if ($env:MODE) { $env:MODE } else { $Mode }
    try { Invoke-Timed $label (@("$Src\tools\turing\prod_equiv.py", "$W\sample", $W, $m) + @(PkgArg $Pkg)) $Limit }
    finally {
      Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue
      foreach ($v in $set) { Remove-Item ("env:" + ($v -split '=', 2)[0]) -ErrorAction SilentlyContinue }
    }
    $s = @(Get-Content $csv -ErrorAction SilentlyContinue | ForEach-Object { , [double[]]($_ -split ',\s*') } |
           Where-Object { $_[1] -gt 0 })                      # samples while the GPU was busy
    if ($s.Count) {
      Log ("{0,-10} smi busy samples {1}: max memory {2} MiB, mean util {3:N0}%, power {4:N0} W, SM {5:N0} MHz" -f $label,
        $s.Count, ($s | ForEach-Object { $_[0] } | Measure-Object -Maximum).Maximum,
        ($s | ForEach-Object { $_[1] } | Measure-Object -Average).Average,
        ($s | ForEach-Object { $_[2] } | Measure-Object -Average).Average,
        ($s | ForEach-Object { $_[3] } | Measure-Object -Average).Average)
    }
  }
}
Log '---- equiv_ab done'
