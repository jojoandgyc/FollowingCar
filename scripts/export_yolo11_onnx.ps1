param(
    [ValidateSet("n", "s", "m")]
    [string]$Variant = "s",
    [string]$ImageName = "python:3.10-slim",
    [string]$UltralyticsVersion = "8.3.0"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ptPath = Join-Path $repo "models\source\yolo11$Variant.pt"
if (-not (Test-Path -LiteralPath $ptPath)) {
    throw "YOLO11$Variant weights not found. Run scripts\download_yolo11_pt.ps1 -Variant $Variant first."
}

$repoDocker = $repo -replace "\\", "/"
$modelPath = "models/source/yolo11$Variant.pt"

docker run --rm `
    -v "${repoDocker}:/workspace" `
    -w /workspace `
    $ImageName `
    sh -lc "python3 -m pip install --no-cache-dir ultralytics==$UltralyticsVersion onnx onnxslim && yolo export model=$modelPath format=onnx imgsz=640 opset=12 simplify=True nms=False"
if ($LASTEXITCODE -ne 0) {
    throw "YOLO11 ONNX export failed with exit code $LASTEXITCODE"
}
