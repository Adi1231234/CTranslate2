# CUDA 12.8.1 without the installer: the components the official wheel build installs (nvcc, cudart,
# cublas_dev, curand_dev; python/tools/prepare_build_environment_windows.sh) plus cuobjdump, from NVIDIA's
# redistributable archives, each checked against the sha256 of the release manifest and unpacked into one
# tree ($Dest\bin, include, lib, nvvm). Library archives contribute headers and import libraries only (the
# *_dev components); the runtime DLLs come from the nvidia-* wheels at run time. Then build with
# $env:CUDA_PATH_V12_8 = $Dest (devenv.ps1).
param([Parameter(Mandatory)][string]$Dest, [string]$Release = '12.8.1')
$base = 'https://developer.download.nvidia.com/compute/cuda/redist'
$manifest = Invoke-RestMethod "$base/redistrib_$Release.json"
$full = @('cuda_nvcc', 'cuda_cudart', 'cuda_cuobjdump'); $dev = @('libcublas', 'libcurand')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$null = New-Item -ItemType Directory -Force "$Dest\_zips"
foreach ($c in $full + $dev) {
  $w = $manifest.$c.'windows-x86_64'
  $zip = "$Dest\_zips\" + (Split-Path $w.relative_path -Leaf)
  if (-not (Test-Path $zip) -or (Get-FileHash $zip -Algorithm SHA256).Hash -ne $w.sha256) {
    curl.exe -sSfL -o $zip "$base/$($w.relative_path)"
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: download $c"; exit 1 }
  }
  if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $w.sha256) { Write-Output "FAILED: sha256 of $c"; exit 1 }
  $z = [IO.Compression.ZipFile]::OpenRead($zip)
  try {
    foreach ($e in $z.Entries) {
      $rel = ($e.FullName -split '/', 2)[1]                 # drop the archive's top folder
      if (-not $rel -or $rel.EndsWith('/') -or ($c -in $dev -and $rel -notmatch '^(include|lib)/')) { continue }
      $out = Join-Path $Dest $rel
      $null = New-Item -ItemType Directory -Force (Split-Path $out)
      [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $out, $true)
    }
  } finally { $z.Dispose() }
  Write-Output "$c $($manifest.$c.version) ok"
}
