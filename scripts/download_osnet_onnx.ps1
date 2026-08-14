param(
    [string]$Output = "",
    [string]$Url = "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Output) {
    $Output = Join-Path $repo "models\source\osnet_x0_25_msmt17.onnx"
}
$outPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

Write-Host "Downloading OSNet ONNX:"
Write-Host "  $Url"
Write-Host "  -> $outPath"
Invoke-WebRequest -Uri $Url -OutFile $outPath
Get-Item -LiteralPath $outPath | Select-Object FullName,Length,LastWriteTime
