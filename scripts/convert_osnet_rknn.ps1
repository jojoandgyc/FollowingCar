param(
    [string]$SourceOnnx = "",
    [string]$PreparedOnnx = "",
    [string]$Output = "",
    [ValidateSet("fp", "i8", "u8")]
    [string]$DType = "fp",
    [string]$Dataset = "",
    [int]$Batch = 1,
    [int]$FromBatch = 16,
    [string]$ImageName = "rk-car-rknn-toolkit2:2.3.2"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $SourceOnnx) {
    $SourceOnnx = Join-Path $repo "models\source\osnet_x0_25_msmt17.onnx"
}
if (-not $PreparedOnnx) {
    $PreparedOnnx = Join-Path $repo "models\source\osnet_x0_25_msmt17_b$Batch.onnx"
}
if (-not $Output) {
    $Output = Join-Path $repo "models\osnet_x0_25_msmt17_b$Batch.rknn"
}

$sourcePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SourceOnnx)
$preparedPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PreparedOnnx)
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "OSNet ONNX not found. Run scripts\download_osnet_onnx.ps1 first: $sourcePath"
}

$repoDocker = $repo -replace "\\", "/"
function Convert-ToRepoRel([string]$Path) {
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path) -replace "\\", "/"
    if (-not $full.StartsWith($repoDocker)) {
        throw "Path must be inside the repository so Docker can access it: $Path"
    }
    return $full.Substring($repoDocker.Length + 1)
}

$sourceRel = Convert-ToRepoRel $sourcePath
$preparedRel = Convert-ToRepoRel $preparedPath
$outputRel = Convert-ToRepoRel $outputPath

$cmd = "python3 tools/patch_onnx_batch.py $sourceRel $preparedRel --batch $Batch --from-batch $FromBatch && python3 tools/convert_onnx_to_rknn.py $preparedRel $outputRel --target rk3588 --dtype $DType --mean-values 0,0,0 --std-values 1,1,1"
if ($Dataset) {
    $datasetRel = Convert-ToRepoRel $Dataset
    $cmd = "$cmd --dataset $datasetRel"
}

docker run --rm `
    -v "${repoDocker}:/workspace" `
    -w /workspace `
    $ImageName `
    sh -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "OSNet RKNN conversion failed with exit code $LASTEXITCODE"
}
