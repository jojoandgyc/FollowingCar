param(
    [ValidateSet("n", "s", "m")]
    [string]$Variant = "s",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Output) {
    $Output = Join-Path $repo "models\source\yolo11$Variant.pt"
}
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11$Variant.pt"
Write-Host "Downloading YOLO11$Variant weights:"
Write-Host "  URL    : $url"
Write-Host "  Output : $outputPath"
Invoke-WebRequest -Uri $url -OutFile $outputPath
Get-Item -LiteralPath $outputPath | Select-Object FullName,Length,LastWriteTime
