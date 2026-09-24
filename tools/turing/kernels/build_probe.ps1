# Build one probe with nvcc against the library's headers and thrust/cub; prints the exe path last.
# -Gencode picks the code the probe carries: sm_75 on Yarin; '-gencode=arch=compute_86,code=compute_86'
# (the PTX the official wheel runs on newer GPUs) for the store PC's RTX 5060 Ti. -Include puts a directory
# before the library sources, so a probe can be built against a modified copy of a header (e.g. to show
# that it catches a known bug); the exe is then named <name>_alt.
# usage: build_probe.ps1 <name> [-Root D:\ct2build] [-Gencode <nvcc arch flag>] [-Include <dir>]
[CmdletBinding(PositionalBinding = $false)]
param([Parameter(Position = 0)][string]$Name, [string]$Root = 'D:\ct2build', [string]$Gencode = '-arch=sm_75',
      [string]$Include = '')
. "$PSScriptRoot\..\devenv.ps1"
$Out = "$Root\probes"
$Src = (Resolve-Path "$PSScriptRoot\..\..\..").Path                 # probes include library headers
$Exe = if ($Include) { "$Out\${Name}_alt.exe" } else { "$Out\$Name.exe" }
$Cccl = "$Src\third_party\thrust"                                   # the library's thrust/cub, as CMake orders them
$Inc = @(if ($Include) { '-I', $Include }) + @('-I', "$Cccl\cub", '-I', "$Cccl\thrust", '-I', "$Cccl\libcudacxx\include",
                                                '-I', "$Src\src", '-I', "$Src\include")
New-Item -ItemType Directory -Force $Out | Out-Null
nvcc -O3 -std=c++17 $Gencode --expt-relaxed-constexpr -diag-suppress 2219 @Inc `
  -o $Exe "$PSScriptRoot\$Name.cu" -lcublas -lcublasLt 2>&1 | Out-String -Stream
Check "nvcc $Name"
$Exe
