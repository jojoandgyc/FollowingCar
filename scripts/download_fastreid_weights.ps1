param(
    [ValidateSet("market_bot_R50")]
    [string]$Model = "market_bot_R50",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$urls = @{
    "market_bot_R50" = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth"
}

if (-not $Output) {
    $Output = Join-Path $repo "models\source\fastreid_$Model.pth"
}

$outPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

Write-Host "Downloading FastReID weights:"
Write-Host "  model: $Model"
Write-Host "  $($urls[$Model])"
Write-Host "  -> $outPath"
Invoke-WebRequest -Uri $urls[$Model] -OutFile $outPath
Get-Item -LiteralPath $outPath | Select-Object FullName,Length,LastWriteTime
