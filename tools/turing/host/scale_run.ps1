# One pass of the production runner (tools/turing/runner, transcribe_run.py) over a unit list for the scale
# check and the real-data benchmark (tools/turing/scale/README.md): a build or the stock wheel, outputs in
# $R\verify\<Label> (a new label per run: the runner skips units whose output exists), a deadline for a clean
# stop between units. The runner runs from a fresh copy in $R\verify\runner, with the production run's
# units.json and its hf cache (a junction: the model is already there); that folder needs hf_token.txt unless
# -Cache holds every unit. -Env 'NAME=value;...' sets variables for the run (e.g. a CT2_* setting).
# $R\verify\<Label>.gpu gets the GPU's shared (system) memory in use every 5 s: above ~0 WDDM is paging.
# usage: scale_run.ps1 -Label <name> -Units <list file> [-Mode pipe8] [-Pkg <build parent dir> | stock]
#                      [-Cache <dir>] [-Env 'A=1;B=2'] [-Deadline 'yyyy-MM-dd HH:mm']
param([string]$Label, [string]$Units, [string]$Mode = 'pipe8', [string]$Pkg = 'stock', [string]$Cache = '',
      [string]$Env = '', [string]$Deadline = '')
. "$PSScriptRoot\prod.ps1"
Sync-Checkout $PSCommandPath $PSBoundParameters
$Run = "$R\verify\runner"
New-Item -ItemType Directory -Force $Run | Out-Null
Copy-Item "$Src\tools\turing\runner\*.py", "$W\units.json" $Run -Force
if (-not (Test-Path "$Run\hf")) { New-Item -ItemType Junction -Path "$Run\hf" -Target "$W\hf" | Out-Null }
# ReadAllLines: plain strings (Get-Content's carry properties that ConvertTo-Json writes out as objects)
$stop = @{ only_units = @([IO.File]::ReadAllLines($Units) | Where-Object { $_ }) }
if ($Deadline) { $stop.deadline = $Deadline }
[IO.File]::WriteAllText("$Run\stop.json", ($stop | ConvertTo-Json -Compress))   # no BOM: json.load fails on one
$env:RUN_OUT = "$R\verify\$Label"
if ($Cache) { New-Item -ItemType Directory -Force $Cache | Out-Null; $env:RUN_CACHE = $Cache }
if ($Pkg -ne 'stock') { $env:PYTHONPATH = $Pkg }
foreach ($v in @($Env -split ';' | Where-Object { $_ })) { $k, $val = $v -split '=', 2; Set-Item "env:$k" $val }
Log ("---- scale_run {0}: {1} units, {2}, {3}, env '{4}', at {5}" -f $Label, $stop.only_units.Count, $Mode,
     (PkgLabel $Pkg), $Env, (& $Git -C $Src log --oneline -1))
$probe = "while (`$true) { `$m = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | " +
         "Measure-Object CookedValue -Maximum; Add-Content '$R\verify\$Label.gpu' ([int](`$m.Maximum / 1MB)); Start-Sleep 5 }"
[IO.File]::WriteAllText("$Run\progress.log", '')           # this run's log only (kept as <Label>.log)
$sampler = Start-Process powershell.exe -PassThru -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $probe
$p = Start-Process -FilePath $Py -ArgumentList "$Run\transcribe_run.py", 'front', $Mode -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput "$R\verify\$Label.out" -RedirectStandardError "$R\verify\$Label.err"
$null = $p.Handle                                          # keeps ExitCode readable after the exit
$p.WaitForExit()
Stop-Process -Id $sampler.Id -Force -ErrorAction SilentlyContinue
Copy-Item "$Run\progress.log" "$R\verify\$Label.log" -Force
$shared = Get-Content "$R\verify\$Label.gpu" -ErrorAction SilentlyContinue | ForEach-Object { [int]$_ } | Measure-Object -Maximum
Log "---- scale_run $Label exit $($p.ExitCode), GPU shared memory peak $($shared.Maximum) MB"
