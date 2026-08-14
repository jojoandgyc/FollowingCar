param(
    [ValidateSet("market_bot_R50")]
    [string]$Model = "market_bot_R50",
    [string]$Weights = "",
    [string]$FastReIDRoot = "",
    [string]$Output = "",
    [int]$Batch = 1,
    [int]$Height = 256,
    [int]$Width = 128,
    [string]$ImageName = "rk-car-rknn-toolkit2:2.3.2"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Weights) {
    $Weights = Join-Path $repo "models\source\fastreid_$Model.pth"
}
if (-not $FastReIDRoot) {
    $FastReIDRoot = Join-Path $repo ".cache\fast-reid-v0.1.1"
}
if (-not $Output) {
    $Output = Join-Path $repo "models\source\fastreid_$Model`_b$Batch.onnx"
}

$weightsPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Weights)
$fastreidPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($FastReIDRoot)
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (-not (Test-Path -LiteralPath $weightsPath)) {
    throw "FastReID weights not found. Run scripts\download_fastreid_weights.ps1 -Model $Model first: $weightsPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $fastreidPath "fastreid"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $fastreidPath) | Out-Null
    git clone --branch v0.1.1 --depth 1 https://github.com/JDAI-CV/fast-reid.git $fastreidPath
}

$configPath = Join-Path $fastreidPath "configs\Market1501\bagtricks_R50.yml"
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "FastReID config not found: $configPath"
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
$fastreidRel = Convert-ToRepoRel $fastreidPath
$configRel = Convert-ToRepoRel $configPath
$outputRel = Convert-ToRepoRel $outputPath

$cmd = "python3 -m pip install -q yacs fvcore iopath termcolor tabulate && python3 tools/export_fastreid_to_onnx.py --fastreid-root $fastreidRel --config-file $configRel --weights $weightsRel --output $outputRel --batch $Batch --height $Height --width $Width"
docker run --rm `
    -v "${repoDocker}:/workspace" `
    -w /workspace `
    $ImageName `
    sh -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "FastReID ONNX export failed with exit code $LASTEXITCODE"
}
