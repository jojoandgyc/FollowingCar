param(
    [string]$Package = "C:\Users\Dylon\Downloads\rk\2.3.2\release\rknn-toolkit2-v2.3.2-2025-04-09.tgz",
    [string]$PythonTag = "cp310",
    [string]$WheelArch = "x86_64",
    [string]$ImageName = "rk-car-rknn-toolkit2:2.3.2",
    [string]$BaseImage = "ubuntu:22.04",
    [string]$Proxy = "",
    [string]$NoProxy = "localhost,127.0.0.1,host.docker.internal,192.168.0.64"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$packagePath = (Resolve-Path $Package).Path
$cache = Join-Path $repo ".cache\rknn-toolkit2"
$extract = Join-Path $cache "extract"
$wheelStage = Join-Path $cache "packages"

New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (Test-Path -LiteralPath $extract) {
    $resolvedExtract = (Resolve-Path -LiteralPath $extract).Path
    if (-not $resolvedExtract.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside repo: $resolvedExtract"
    }
    Remove-Item -LiteralPath $resolvedExtract -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $extract | Out-Null

$archiveEntries = & tar -tf $packagePath
if ($LASTEXITCODE -ne 0) {
    throw "tar list failed with exit code $LASTEXITCODE"
}
$wheelEntry = $archiveEntries |
    Where-Object {
        $_ -like "*rknn_toolkit2-*-$PythonTag-*.whl" -and
        (-not $WheelArch -or $_ -like "*$WheelArch*")
    } |
    Sort-Object |
    Select-Object -First 1
if ($null -eq $wheelEntry) {
    throw "No RKNN Toolkit2 wheel found for Python tag '$PythonTag' and arch '$WheelArch' in $packagePath"
}
tar -xf $packagePath -C $extract $wheelEntry
if ($LASTEXITCODE -ne 0) {
    throw "tar extract failed with exit code $LASTEXITCODE"
}
$wheelName = Split-Path $wheelEntry -Leaf
$wheel = Get-ChildItem -Path $extract -Recurse -Filter $wheelName -File |
    Select-Object -First 1
if ($null -eq $wheel) {
    throw "Extracted RKNN Toolkit2 wheel not found: $wheelName"
}

New-Item -ItemType Directory -Force -Path $wheelStage | Out-Null
$wheelOut = Join-Path $wheelStage $wheel.Name
Copy-Item -LiteralPath $wheel.FullName -Destination $wheelOut -Force
$repoUri = New-Object System.Uri (($repo.TrimEnd("\") + "\"))
$wheelUri = New-Object System.Uri $wheelOut
$wheelRel = [System.Uri]::UnescapeDataString($repoUri.MakeRelativeUri($wheelUri).ToString())
Write-Host "Using RKNN wheel: $($wheel.FullName)"
Write-Host "Building Docker image: $ImageName"

$dockerArgs = @(
    "build",
    "-f", (Join-Path $repo "docker\rknn-toolkit2\Dockerfile"),
    "--build-arg", "BASE_IMAGE=$BaseImage",
    "--build-arg", "RKNN_WHL=$wheelRel",
    "-t", $ImageName
)
if ($Proxy) {
    $dockerArgs += @(
        "--build-arg", "HTTP_PROXY=$Proxy",
        "--build-arg", "HTTPS_PROXY=$Proxy",
        "--build-arg", "http_proxy=$Proxy",
        "--build-arg", "https_proxy=$Proxy",
        "--build-arg", "NO_PROXY=$NoProxy",
        "--build-arg", "no_proxy=$NoProxy"
    )
}
$dockerArgs += $repo

& docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed with exit code $LASTEXITCODE"
}
