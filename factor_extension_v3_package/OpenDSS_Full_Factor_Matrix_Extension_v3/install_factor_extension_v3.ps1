[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BundleRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$packageRoot = $PSScriptRoot
$bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
$backup = Join-Path $bundle "frozen\pre_factor_v3_source_backup"
$packageManifest = Join-Path $packageRoot "PACKAGE_SHA256SUMS.csv"

if (-not (Test-Path -LiteralPath $packageManifest)) {
    throw "Package hash manifest is missing: $packageManifest"
}

foreach ($row in @(Import-Csv -LiteralPath $packageManifest)) {
    $relative = $row.Path.Replace("/", [IO.Path]::DirectorySeparatorChar)
    $path = Join-Path $packageRoot $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Package file is missing: $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $row.SHA256) {
        throw "Package hash mismatch for $($row.Path): $actual"
    }
}

$baseHashes = @{
    "net_fed.cc"         = "C95ACEF3128B992B26F11EFF8ACCB3536AE84F40A576D9F70CFC9284F458F59E"
    "twin_fed.py"        = "F2A2060F7F9449D6453BB8E9B06325B7101ADE855361832BE48D102B486E1D7B"
    "docker-compose.yml" = "17E9B8E471F076CCA4A3247F8F7CC87527CB3A64A1B7C9A113C1D47E76614CE5"
    "run_experiment.ps1" = "F01EB9C5418D3188FFEDFD4B4AE8A3B3DCED0A5E0B7B181E12BB60480A2C6912"
}

foreach ($required in @(
    "Dockerfile",
    "net_fed.cc",
    "twin_fed.py",
    "docker-compose.yml",
    "run_experiment.ps1",
    "calibration.v2.json",
    "gamma.frozen.v2.txt",
    "W.frozen.v2.npy",
    "physical_design.production.v1.json",
    "run_opendss_exporter.ps1",
    "validate_truth.ps1"
)) {
    $path = Join-Path $bundle $required
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Bundle input is missing: $path"
    }
}

foreach ($name in $baseHashes.Keys) {
    $path = Join-Path $bundle $name
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $baseHashes[$name]) {
        throw (
            "Base-source hash mismatch for $name. " +
            "Expected $($baseHashes[$name]); actual $actual. " +
            "Do not force installation over an unknown source revision."
        )
    }
}

if (Test-Path -LiteralPath $backup) {
    throw "Source backup already exists: $backup"
}

New-Item -ItemType Directory -Path $backup -Force | Out-Null

$sourceTargets = @(
    "net_fed.cc",
    "twin_fed.py",
    "docker-compose.yml",
    "run_experiment.ps1"
)

foreach ($name in $sourceTargets) {
    Copy-Item `
        -LiteralPath (Join-Path $bundle $name) `
        -Destination (Join-Path $backup $name)
}

$backupHashes = foreach ($name in $sourceTargets) {
    Get-FileHash `
        -LiteralPath (Join-Path $backup $name) `
        -Algorithm SHA256
}

$backupHashes |
    Export-Csv `
        -LiteralPath (Join-Path $backup "SOURCE_SHA256SUMS.csv") `
        -NoTypeInformation

foreach ($name in $sourceTargets) {
    Copy-Item `
        -LiteralPath (Join-Path $packageRoot $name) `
        -Destination (Join-Path $bundle $name) `
        -Force
}

foreach ($name in @(
    "factor_design.production.v3.json",
    "run_factor_matrix_v3.ps1",
    "verify_factor_extension_v3.py",
    "FULL_FACTOR_MATRIX_README.md"
)) {
    Copy-Item `
        -LiteralPath (Join-Path $packageRoot $name) `
        -Destination (Join-Path $bundle $name) `
        -Force
}

$bundleTests = Join-Path $bundle "tests"
New-Item -ItemType Directory -Path $bundleTests -Force | Out-Null
Copy-Item `
    -LiteralPath (Join-Path $packageRoot "tests\test_factor_extension.py") `
    -Destination (Join-Path $bundleTests "test_factor_extension.py") `
    -Force

$installedFiles = @(
    "net_fed.cc",
    "twin_fed.py",
    "docker-compose.yml",
    "run_experiment.ps1",
    "factor_design.production.v3.json",
    "run_factor_matrix_v3.ps1",
    "verify_factor_extension_v3.py",
    "FULL_FACTOR_MATRIX_README.md",
    "tests\test_factor_extension.py"
)

$installedHashes = foreach ($name in $installedFiles) {
    Get-FileHash `
        -LiteralPath (Join-Path $bundle $name) `
        -Algorithm SHA256
}

$installedHashes |
    Export-Csv `
        -LiteralPath (Join-Path $backup "FACTOR_V3_INSTALLED_SHA256SUMS.csv") `
        -NoTypeInformation

"FACTOR_V3_INSTALL_OK"
"BUNDLE_ROOT=$bundle"
"BACKUP_DIRECTORY=$backup"
$installedHashes | Format-Table -AutoSize
