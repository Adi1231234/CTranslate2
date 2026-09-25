# One pass of the production runner (tools/turing/runner, transcribe_run.py) over a unit list for the scale
# check (tools/turing/scale/README.md): a build or the stock wheel, outputs in $R\verify\<Label>, a deadline
# for a clean stop between units. The runner runs from a fresh copy in $R\verify\runner, with the production
# run's units.json and its hf cache (a junction: the model is already there); that folder must hold
# hf_token.txt. Start it detached so it survives the remote session.
# usage: scale_run.ps1 -Label <name> -Units <list file> [-Mode pipe8] [-Pkg <build parent dir> | stock]
#                      [-Deadline 'yyyy-MM-dd HH:mm']
param([string]$Label, [string]$Units, [string]$Mode = 'pipe8', [string]$Pkg = 'stock', [string]$Deadline = '')
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
New-Item -ItemType Directory -Force "$R\verify" | Out-Null
$env:RUN_OUT = "$R\verify\$Label"
if ($Pkg -ne 'stock') { $env:PYTHONPATH = $Pkg }
Log ("---- scale_run {0}: {1} units, {2}, {3}, at {4}" -f $Label, $stop.only_units.Count, $Mode, (PkgLabel $Pkg),
     (& $Git -C $Src log --oneline -1))
$p = Start-Process -FilePath $Py -ArgumentList "$Run\transcribe_run.py", 'front', $Mode -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput "$R\verify\$Label.out" -RedirectStandardError "$R\verify\$Label.err"
$null = $p.Handle                                          # keeps ExitCode readable after the exit
$p.WaitForExit()
Log "---- scale_run $Label exit $($p.ExitCode)"
