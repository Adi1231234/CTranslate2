# The release gate of tools/turing/README.md for one build: the kernel probes, bench_whisper.py on the
# short, middle and long clips, and prod_equiv.py in both production modes. Every hash must equal the
# stock wheel's. For an idle GPU only (it does not pause production). Log: D:\ct2build\ab.log, ending
# with "GATE PASS" or "GATE FAIL". Start it detached so it survives the remote session.
param([string]$Pkg = 'D:\ct2build\pyct2-next')
. "$PSScriptRoot\prod.ps1"
$T = "$Src\tools\turing"
$fail = 0
function Test-Hashes($label, $line, $keys, $want) {
  $got = try { $j = $line | ConvertFrom-Json; ($keys | ForEach-Object { $j.$_ }) -join ' ' } catch { "unparsed: $line" }
  $ok = $got -eq $want
  if (-not $ok) { $script:fail++ }
  Log ("{0,-13} {1} {2}" -f $label, $(if ($ok) { 'PASS' } else { "FAIL, stock $want" }), $got)
}
Log ("---- gate at " + (& $Git -C $Src log --oneline -1) + " pkg $Pkg built from " + (Get-Content "$Pkg\ctranslate2\BUILD.txt"))
foreach ($p in 'softmax_check', 'qk_check', 'ts_check') {
  $o = powershell -NoProfile -ExecutionPolicy Bypass -File "$T\kernels\run_probe.ps1" $p 2>&1 | Out-String
  $m = [regex]::Match($o, 'TOTAL (\d+) mismatch')
  $ok = $m.Success -and $m.Groups[1].Value -eq '0'
  if (-not $ok) { $fail++ }
  Log ("{0,-13} {1} {2}" -f $p, $(if ($ok) { 'PASS' } else { 'FAIL' }),
       $(if ($m.Success) { $m.Value } else { ($o -split "`n" | Select-Object -Last 3) -join ' | ' }))
}
$bench = @{ '60' = '6a0ac32080d7a470 9e22e90b78daeac6 106e321f983cf1c5'; '90' = '25b0f78ad0a29e20 3607009d6b13b9e5 a5578d8055111e6c'
            '118' = '67bd5eebe0d5f92b bd7bde6d311f4801 caa6012de4a3b407' }
foreach ($first in '60', '90', '118') {
  $env:BENCH_FIRST = $first
  $line = Invoke-Timed "bench$first" @("$T\bench_whisper.py", "$W\sample", $Pkg) 300
  Test-Hashes "bench$first" $line @('tokens_sha', 'full_sha', 'enc_sha') $bench[$first]
}
Remove-Item env:BENCH_FIRST
$env:CPU_THREADS = '1'                                   # as production
foreach ($m in @(@('pipe8', '262ababd557e7c80'), @('exact2', 'a83ba8808ab352df'))) {
  $line = Invoke-Timed $m[0] @("$T\prod_equiv.py", "$W\sample", $W, $m[0], $Pkg) 600
  Test-Hashes $m[0] $line @('rows_sha') $m[1]
}
Remove-Item env:CPU_THREADS
Log $(if ($fail) { "GATE FAIL ($fail)" } else { 'GATE PASS' })
