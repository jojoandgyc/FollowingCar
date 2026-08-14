param(
    [ValidateSet("n", "s", "m")]
    [string]$Variant = "s",
    [ValidateSet("fp", "i8", "u8")]
    [string]$DType = "fp",
    [string]$Dataset = "",
    [string]$ImageName = "rk-car-rknn-toolkit2:2.3.2"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$onnxPath = Join-Path $repo "models\source\yolo11$Variant.onnx"
if (-not (Test-Path -LiteralPath $onnxPath)) {
    throw "YOLO11$Variant ONNX not found. Run scripts\export_yolo11_onnx.ps1 -Variant $Variant first."
}

$repoDocker = $repo -replace "\\", "/"
$cmd = "python3 tools/convert_onnx_to_rknn.py models/source/yolo11$Variant.onnx models/yolo11$Variant.rknn --target rk3588 --dtype $DType"
if ($Dataset) {
    $datasetDocker = ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dataset) -replace "\\", "/")
    if (-not $datasetDocker.StartsWith($repoDocker)) {
        throw "Dataset must be inside the repository so Docker can access it: $Dataset"
    }
    $datasetRel = $datasetDocker.Substring($repoDocker.Length + 1)
    $cmd = "$cmd --dataset $datasetRel"
}

docker run --rm `
    -v "${repoDocker}:/workspace" `
    -w /workspace `
    $ImageName `
    sh -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "YOLO11 RKNN conversion failed with exit code $LASTEXITCODE"
}
