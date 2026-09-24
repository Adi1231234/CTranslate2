# The fast byte-exactness check (digest.py) of builds against the golden file of the stock wheel;
# -Record first writes that file from the stock wheel. 'stock' as a package means the stock wheel.
# For an idle GPU only. Log: D:\ct2build\ab.log. Start it detached so it survives the remote session.
# usage: digest.ps1 [-Pkgs 'D:\ct2build\pyct2-next', 'stock', ...] [-Record]
param([string[]]$Pkgs = @('D:\ct2build\pyct2-next'), [switch]$Record)
. "$PSScriptRoot\prod.ps1"
& $Git -C $Src pull -q --ff-only 2>&1 | Out-Null
$gold = "$R\golden_digest.json"
$d = "$Src\tools\turing\digest.py"
Log ("---- digest at " + (& $Git -C $Src log --oneline -1))
if ($Record) { $null = Invoke-Timed 'record' @($d, "$W\sample", '--save', $gold) 300 }
foreach ($p in $Pkgs) {
  $null = Invoke-Timed (PkgLabel $p) (@($d, "$W\sample") + @(PkgArg $p) + @('--golden', $gold)) 300
}
Log '---- digest done'
