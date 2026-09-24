# Speed verdict of a base/next A/B from prod_equiv.py result lines (dot-source).
# Run-to-run noise (CV) on Yarin 24.9, pooled within iterate runs of the same two builds: CUPTI GPU busy
# time 0.18%, wall time 1.74%. The NIST sample size (e-Handbook 7.2.2.2), N = (z(1-a/2) + z(1-b))^2
# (CV/delta)^2, doubled for the difference of two means, then asks for 1 run per build to see a 1% change
# on GPU time and 64 on wall time (alpha 0.05, power 90%). GPU time does not see host-side or overlap gains
# (allocator, syncs, stream priorities): judge those on wall time (iterate.ps1 -WallDelta).
$NoiseCv = @{ gpu = 0.18; wall = 1.74 }          # percent
$MetricKey = @{ gpu = 'gpu_busy_s'; wall = 'seconds' }

function Get-RoundsFor($metric, $deltaPct) {
  $z = 1.959964 + 1.281552
  [int][math]::Max(1, [math]::Ceiling(2 * $z * $z * [math]::Pow($NoiseCv[$metric] / $deltaPct, 2)))
}

function ConvertFrom-RunLines($lines) {
  @($lines | Where-Object { $_ -like '{*' } | ForEach-Object { $_ | ConvertFrom-Json })
}

function Get-AbVerdict($baseLines, $nextLines, $metric) {
  $b = @(ConvertFrom-RunLines $baseLines); $n = @(ConvertFrom-RunLines $nextLines)   # @(): PS 5.1 unrolls 1 run
  $bad = @($baseLines).Count - $b.Count + @($nextLines).Count - $n.Count
  if ($b.Count -eq 0 -or $n.Count -eq 0) { return "$metric INCOMPLETE: $bad run(s) without a result line" }
  $key = $MetricKey[$metric]
  $ratio = ($n | Measure-Object $key -Average).Average / ($b | Measure-Object $key -Average).Average
  $noise = 3 * $NoiseCv[$metric] * [math]::Sqrt(1 / $b.Count + 1 / $n.Count)   # 3 sigma of the ratio, percent
  $call = if ([math]::Abs($ratio - 1) * 100 -le $noise) { 'within noise' } elseif ($ratio -lt 1) { 'FASTER' } else { 'SLOWER' }
  $shas = @($b + $n | ForEach-Object { $_.rows_sha } | Select-Object -Unique)
  $rows = if ($shas.Count -eq 1) { "rows_sha $($shas[0])" } else { "ROWS DIFFER: $($shas -join ',')" }
  $failed = if ($bad) { "; $bad run(s) without a result line" } else { '' }
  '{0} next/base {1:N4} {2} (3-sigma noise {3:N2}%, runs {4}+{5}); {6}{7}' -f $metric, $ratio, $call, $noise, $b.Count, $n.Count, $rows, $failed
}
