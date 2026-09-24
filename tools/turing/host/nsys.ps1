# Nsight Systems profile of the production engine (prod_equiv.py, warm run only via PROFILE_RANGE) on
# one build, exported to SQLite and summarized by nsys_gaps.py and nsys_streams.py into
# D:\ct2build\nsys\<name>.txt. For an idle GPU only. Start it detached so it survives the remote session.
# usage: nsys.ps1 [-Name v5] [-Pkg D:\ct2build\pyct2-next] [-Mode pipe8] [-Sample]
param([string]$Name = 'v5', [string]$Pkg = 'D:\ct2build\pyct2-next', [string]$Mode = 'pipe8', [switch]$Sample)
. "$PSScriptRoot\prod.ps1"
Sync-Checkout $PSCommandPath $PSBoundParameters
$T = "$Src\tools\turing"; $N = "$R\nsys\$Name"
$Nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe'
Log ("---- nsys $Name at " + (& $Git -C $Src log --oneline -1) + " pkg $Pkg mode $Mode")
$env:PROFILE_RANGE = '1'; $env:CPU_THREADS = '1'
$samp = if ($Sample) { @('--sample=process-tree', '--python-sampling=true') } else { @('--sample=none', '--cpuctxsw=none') }
& $Nsys profile --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true `
  -o $N @samp $Py "$T\prod_equiv.py" "$W\sample" $W $Mode $Pkg *> "$N.log"
& $Nsys export --type=sqlite --force-overwrite=true -o "$N.sqlite" "$N.nsys-rep" *> $null
& $Py "$T\nsys_gaps.py" "$N.sqlite" *> "$N.txt"
& $Py "$T\nsys_streams.py" "$N.sqlite" *>> "$N.txt"
Log ("nsys $Name " + ((Get-Content "$N.log" | Where-Object { $_ -like '{*' } | Select-Object -Last 1)) + " -> $N.txt")
