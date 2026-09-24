# Run one tools/turing python script with production paused and a hard time limit.
# usage: run_paused.ps1 <script.py> <limit_s> [script args...]
param([string]$Script, [int]$Limit = 180, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
. "$PSScriptRoot\prod.ps1"
Sync-Checkout $PSCommandPath $PSBoundParameters
Log ("---- $Script at " + (& $Git -C $Src log --oneline -1))
Suspend-Production
try { Invoke-Timed $Script (@("$Src\tools\turing\$Script") + $Rest) $Limit } finally { Resume-Production }
