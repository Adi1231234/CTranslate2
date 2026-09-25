# The real-data benchmark in its two parts (scale/parts.py: the batched path alone, then the fallback clips
# alone) for each package ('stock' = the venv's wheel). For an idle GPU only. Log: $R\ab.log. Start it
# detached so it survives the remote session.
# usage: parts.ps1 [-Pkgs 'D:\ct2build\pyct2-g', 'stock'] [-Mode pipe8] [-Cache <dir>] [-Units <list>]
param([string[]]$Pkgs = @('D:\ct2build\pyct2-g'), [string]$Mode = 'pipe8', [string]$Cache = 'D:\ct2build\verify\cache_real',
      [string]$Units = '', [int]$Limit = 3600)
. "$PSScriptRoot\prod.ps1"
Sync-Checkout $PSCommandPath $PSBoundParameters
if (-not $Units) { $Units = "$Src\tools\turing\scale\units_real.txt" }
Log ("---- parts at " + (& $Git -C $Src log --oneline -1) + " mode $Mode")
foreach ($p in $Pkgs) {
  $null = Invoke-Timed "parts-$(PkgLabel $p)" (@("$Src\tools\turing\scale\parts.py", $Cache, $Units,
    "$Src\tools\turing\runner", $Mode) + @(PkgArg $p)) $Limit
}
Log '---- parts done'
