[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Truth,
    [Parameter(Mandatory = $true)][string]$Role,
    [Parameter(Mandatory = $true)][int]$Seed,
    [int]$Events = 1100,
    [string]$DisjointWith,
    [string]$Image = "dt-opendss-exporter:0.1"
)

$ErrorActionPreference = "Stop"
$bundleRoot = (Get-Location).Path.TrimEnd("\", "/")

function Convert-ToContainerPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $full = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $bundleRoot $Path))
    }
    if (-not (Test-Path -LiteralPath $full)) {
        throw "File does not exist: $full"
    }
    $prefix = $bundleRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must be inside the bundle directory: $full"
    }
    return "/workspace/" + $full.Substring($prefix.Length).Replace("\", "/")
}

$arguments = @(
    "/workspace/validate_opendss_truth.py",
    (Convert-ToContainerPath $Truth),
    "--feeder", "/workspace/feeder.npz",
    "--expected-role", $Role,
    "--expected-seed", "$Seed",
    "--expected-events", "$Events"
)
if (-not [string]::IsNullOrWhiteSpace($DisjointWith)) {
    $arguments += @("--disjoint-with", (Convert-ToContainerPath $DisjointWith))
}

& docker run --rm `
    --mount "type=bind,source=$bundleRoot,target=/workspace" `
    --workdir /workspace `
    $Image @arguments

if ($LASTEXITCODE -ne 0) {
    throw "Truth validation failed with exit code $LASTEXITCODE"
}
