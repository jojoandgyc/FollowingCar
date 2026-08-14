param(
    [string]$Package = "C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz",
    [string]$PythonTag = "cp310",
    [string]$WheelArch = "aarch64",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$packagePath = (Resolve-Path $Package).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repo "third_party\rknpu2\packages"
}
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$cache = Join-Path $repo ".cache\rknn-lite2-wheel"
$extract = Join-Path $cache "extract"

New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (Test-Path -LiteralPath $extract) {
    $resolvedExtract = (Resolve-Path -LiteralPath $extract).Path
    if (-not $resolvedExtract.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside repo: $resolvedExtract"
    }
    Remove-Item -LiteralPath $resolvedExtract -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $extract | Out-Null
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$archiveEntries = & tar -tf $packagePath
if ($LASTEXITCODE -ne 0) {
    throw "tar list failed with exit code $LASTEXITCODE"
}
$wheelEntry = $archiveEntries |
    Where-Object {
        $_ -like "*rknn_toolkit_lite2-*-$PythonTag-*.whl" -and
        (-not $WheelArch -or $_ -like "*$WheelArch*")
    } |
    Sort-Object |
    Select-Object -First 1
if ($null -eq $wheelEntry) {
    throw "No RKNN Toolkit Lite2 wheel found for Python tag '$PythonTag' and arch '$WheelArch' in $packagePath"
}
tar -xf $packagePath -C $extract $wheelEntry
if ($LASTEXITCODE -ne 0) {
    throw "tar extract failed with exit code $LASTEXITCODE"
}
$wheelName = Split-Path $wheelEntry -Leaf
$wheel = Get-ChildItem -Path $extract -Recurse -Filter $wheelName -File |
    Select-Object -First 1
if ($null -eq $wheel) {
    throw "Extracted RKNN Toolkit Lite2 wheel not found: $wheelName"
}

$wheelOut = Join-Path $outputPath $wheel.Name
Copy-Item -LiteralPath $wheel.FullName -Destination $wheelOut -Force

Write-Host "Staged RKNN Toolkit Lite2 wheel:"
Write-Host "  Source : $($wheel.FullName)"
Write-Host "  Output : $wheelOut"
