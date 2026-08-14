param(
    [ValidateSet("x0_25", "x0_5", "x0_50", "x0_75", "x1_0")]
    [string]$Variant = "x0_5",
    [string]$Weights = "",
    [string]$OsnetSource = "",
    [string]$Output = "",
    [int]$Batch = 1,
    [int]$Height = 256,
    [int]$Width = 128,
    [string]$ImageName = "rk-car-rknn-toolkit2:2.3.2"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$variantKey = $Variant
if ($variantKey -eq "x0_50") {
    $variantKey = "x0_5"
}
if (-not $Weights) {
    $Weights = Join-Path $repo "models\source\osnet_$variantKey`_msmt17_combineall.pth"
}
if (-not $OsnetSource) {
    $OsnetSource = Join-Path $repo ".cache\torchreid\osnet.py"
}
if (-not $Output) {
    $Output = Join-Path $repo "models\source\osnet_$variantKey`_msmt17_combineall_b$Batch.onnx"
}

$weightsPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Weights)
$sourcePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OsnetSource)
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (-not (Test-Path -LiteralPath $weightsPath)) {
    throw "OSNet weights not found. Run scripts\download_osnet_torchreid_weights.ps1 -Variant $variantKey first: $weightsPath"
}
if (-not (Test-Path -LiteralPath $sourcePath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $sourcePath) | Out-Null
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/models/osnet.py" `
        -OutFile $sourcePath
}

$repoDocker = $repo -replace "\\", "/"
function Convert-ToRepoRel([string]$Path) {
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path) -replace "\\", "/"
    if (-not $full.StartsWith($repoDocker)) {
        throw "Path must be inside the repository so Docker can access it: $Path"
    }
    return $full.Substring($repoDocker.Length + 1)
}

$weightsRel = Convert-ToRepoRel $weightsPath
$sourceRel = Convert-ToRepoRel $sourcePath
$outputRel = Convert-ToRepoRel $outputPath

$cmd = "python3 tools/export_torchreid_osnet_to_onnx.py --arch osnet_$variantKey --weights $weightsRel --osnet-source $sourceRel --output $outputRel --batch $Batch --height $Height --width $Width"
docker run --rm `
    -v "${repoDocker}:/workspace" `
    -w /workspace `
    $ImageName `
    sh -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "OSNet ONNX export failed with exit code $LASTEXITCODE"
}
