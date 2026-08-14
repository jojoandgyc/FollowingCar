param(
    [string]$Output = "",
    [string]$Url = "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11n.onnx"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Output) {
    $Output = Join-Path $repo "models\source\yolo11n.onnx"
}
$outPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

Write-Host "Downloading YOLO11 ONNX:"
Write-Host "  $Url"
Write-Host "  -> $outPath"
Invoke-WebRequest -Uri $Url -OutFile $outPath
Get-Item -LiteralPath $outPath | Select-Object FullName,Length,LastWriteTime
