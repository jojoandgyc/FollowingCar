param(
    [ValidateSet("x0_25", "x0_5", "x0_50", "x0_75", "x1_0")]
    [string]$Variant = "x0_5",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$variantKey = $Variant
if ($variantKey -eq "x0_50") {
    $variantKey = "x0_5"
}

$files = @{
    "x0_25" = "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    "x0_5" = "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    "x0_75" = "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    "x1_0" = "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
}

$fileName = $files[$variantKey]
if (-not $Output) {
    $Output = Join-Path $repo "models\source\osnet_$variantKey`_msmt17_combineall.pth"
}

$url = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/$fileName"
$outPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

Write-Host "Downloading Torchreid OSNet weights:"
Write-Host "  variant: $variantKey"
Write-Host "  $url"
Write-Host "  -> $outPath"
Invoke-WebRequest -Uri $url -OutFile $outPath
Get-Item -LiteralPath $outPath | Select-Object FullName,Length,LastWriteTime
