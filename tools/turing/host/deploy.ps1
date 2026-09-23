# Make pyct2-next the production build: pause, swap the package (previous one kept as pyct2-prev), resume.
. "$PSScriptRoot\prod.ps1"
Suspend-Production
try {
  if (Test-Path "$R\pyct2-prev") { Remove-Item "$R\pyct2-prev" -Recurse -Force }
  Rename-Item "$R\pyct2" 'pyct2-prev'
  Copy-Item "$R\pyct2-next" "$R\pyct2" -Recurse
  Log ('deployed ' + (Get-Content "$R\pyct2\ctranslate2\BUILD.txt" -ErrorAction SilentlyContinue))
} finally { Resume-Production }
