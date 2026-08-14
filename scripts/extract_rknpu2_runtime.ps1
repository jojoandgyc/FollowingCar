param(
    [string]$Package = "C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$packagePath = (Resolve-Path $Package).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repo "third_party\rknpu2\runtime"
}
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$cache = Join-Path $repo ".cache\rknpu2-runtime"
$extract = Join-Path $cache "extract"
$runtimeExtract = Join-Path $cache "runtime-extract"

New-Item -ItemType Directory -Force -Path $cache | Out-Null
foreach ($path in @($extract, $runtimeExtract)) {
    if (Test-Path -LiteralPath $path) {
        $resolvedPath = (Resolve-Path -LiteralPath $path).Path
        if (-not $resolvedPath.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove outside repo: $resolvedPath"
        }
        Remove-Item -LiteralPath $resolvedPath -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $extract | Out-Null

tar -xf $packagePath -C $extract
$runtimeTar = Get-ChildItem -Path $extract -Recurse -Filter "rknpu2_runtime_*.tar.gz" |
    Sort-Object FullName |
    Select-Object -First 1
$runtimeSource = $null
$sourceDescription = ""
if ($null -ne $runtimeTar) {
    New-Item -ItemType Directory -Force -Path $runtimeExtract | Out-Null
    tar -xzf $runtimeTar.FullName -C $runtimeExtract
    $runtimeSource = Get-ChildItem -Path $runtimeExtract -Directory -Recurse |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName "Linux\librknn_api")) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "Linux\rknn_server"))
        } |
        Sort-Object FullName |
        Select-Object -First 1
    $sourceDescription = $runtimeTar.FullName
}
else {
    $runtimeSource = Get-ChildItem -Path $extract -Directory -Recurse |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName "Linux\librknn_api")) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "Linux\rknn_server"))
        } |
        Sort-Object FullName |
        Select-Object -First 1
    $sourceDescription = $packagePath
}
if ($null -eq $runtimeSource) {
    throw "No RKNPU2 runtime directory found in $packagePath"
}

if (Test-Path -LiteralPath $outputPath) {
    $resolvedOutput = (Resolve-Path -LiteralPath $outputPath).Path
    if (-not $resolvedOutput.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside repo: $resolvedOutput"
    }
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
Copy-Item -Path (Join-Path $runtimeSource.FullName "*") -Destination $outputPath -Recurse -Force

$requiredRuntimeFiles = @(
    "Linux\librknn_api\include\rknn_api.h",
    "Linux\librknn_api\aarch64\librknnrt.so",
    "Linux\rknn_server\aarch64\usr\bin\rknn_server"
)
foreach ($relativePath in $requiredRuntimeFiles) {
    $candidate = Join-Path $outputPath $relativePath
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Runtime extraction missing expected file: $candidate"
    }
}

$versions = Get-ChildItem -Path $extract -Recurse -Filter "versions.txt" |
    Sort-Object FullName |
    Select-Object -First 1
if ($null -ne $versions) {
    Copy-Item -LiteralPath $versions.FullName -Destination (Join-Path $outputPath "versions.txt") -Force
}

Write-Host "Extracted RKNPU2 runtime:"
Write-Host "  Source : $sourceDescription"
Write-Host "  Output : $outputPath"
Write-Host ""
Write-Host "Key paths:"
Write-Host "  include: third_party/rknpu2/runtime/Linux/librknn_api/include"
Write-Host "  lib    : third_party/rknpu2/runtime/Linux/librknn_api/aarch64"
Write-Host "  server : third_party/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin"
