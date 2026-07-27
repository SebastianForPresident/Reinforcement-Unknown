$ErrorActionPreference = "Stop"

Write-Host "Building PPO_Harness..."
dotnet build -c Debug

$dllSource = Join-Path $PSScriptRoot "bin\Debug\netstandard2.1\PPO_Harness.dll"
$dllTarget = Join-Path $PSScriptRoot "..\BepInEx\plugins\PPO_Harness.dll"

Write-Host "Copying DLL..."
Copy-Item -Path $dllSource -Destination $dllTarget -Force

Write-Host "Updating clipboard..."
Set-Clipboard -Value "H4sIAAAAAAAACoWUTW4bMQyF7zLroEiaoovcIGcIsqAkaoaw/kJRddyidw/HWXg8huSFN+MP5CPfE9/+Ta+C8Wl6mVqyC5WCbno4f/s5vXgIFf8/XBgDFUPO4jBVktOFfPrxuOVWJrYgVAIhd7G1nDCUe+VWhpJlVP4CPV5DEA1hkkDzIhvoqg5FDBRp8//vXZUqwEJprq2s2utG0xb7LDNQ6gqOKGByoBoZBLvYghC015DxDFYaYxn1MwHRDctQ8miFcqoFtxb3MLtAsgNVZJhCgBUd9o05u5Q56qB/79Mmp1bVIQMcM3etVpMiJWCcsb8TEAF7cBBhHsijRDNE1GldXeDQJ6uuuNiTDTjeICbbouF1exZKv5xwPmzjJdyunxqr4d/q/Wcf85ktnrX1mZCPqw34R5/HoGOgj0autLpoJvuYo+pp1kRGLdfHUj6rYtRJ6Tt5g8OiuVMnBANUUXfT6AjpNXDI+4A+P9/cDKW0ZISY21bp7XFRUB9Y9l7PUTd0aymnnp6GCQ7ZHgrpj9FS1an7JJxQU+7I0+00v3bN18tlUI6ICfVALR8NrtKznwljQdWpLt2ZKutRHe/HoWnzMXNwO7PfvwCoHYQSPQYAAA=="

$gameExe = Join-Path $PSScriptRoot "..\CasualtiesUnknown.exe"

Write-Host "Launching Casualties Unknown..."
Start-Process $gameExe

Write-Host ""
Write-Host "Done!"
Write-Host "  - Built PPO_Harness"
Write-Host "  - Copied PPO_Harness.dll"
Write-Host "  - Updated clipboard"
Write-Host "  - Launched Casualties Unknown"

