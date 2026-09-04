[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Validate", "Export")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$WeightSource,

    [Parameter(Mandatory = $true)]
    [string]$Role,

    [Parameter(Mandatory = $true)]
    [int]$Seed,

    [string]$Output,
    [int]$Events = 1100,
    [int]$StepsPerEvent = 12,
    [double]$Dt = 1.0,
    [string]$Image = "dt-opendss-exporter:0.1",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$bundleRoot = (Get-Location).Path.TrimEnd("\", "/")

function Convert-ToContainerPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [bool]$MustExist = $true
    )

    $candidate = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $bundleRoot $Path))
    }

    if ($MustExist -and -not (Test-Path -LiteralPath $candidate)) {
        throw "Required file does not exist: $candidate"
    }

    $prefix = $bundleRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith(
        $prefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Path must be inside the bundle directory: $candidate"
    }

    $relative = $candidate.Substring($prefix.Length).Replace("\", "/")
    return "/workspace/$relative"
}

foreach ($required in @(
    ".\export_opendss_truth.py",
    ".\validate_opendss_truth.py",
    ".\physical_design.production.v1.json",
    ".\feeder.npz",
    ".\opendss\123Bus\IEEE123Master.dss",
    ".\opendss\IEEELineCodes.DSS"
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Exporter prerequisite is missing: $required"
    }
}

$weightContainer = Convert-ToContainerPath -Path $WeightSource
$arguments = @(
    "/workspace/export_opendss_truth.py",
    "--master", "/workspace/opendss/123Bus/IEEE123Master.dss",
    "--model-root", "/workspace/opendss",
    "--feeder", "/workspace/feeder.npz",
    "--design", "/workspace/physical_design.production.v1.json",
    "--weight-source", $weightContainer,
    "--role", $Role,
    "--seed", $Seed.ToString(
        [Globalization.CultureInfo]::InvariantCulture
    ),
    "--events", $Events.ToString(
        [Globalization.CultureInfo]::InvariantCulture
    ),
    "--steps-per-event", $StepsPerEvent.ToString(
        [Globalization.CultureInfo]::InvariantCulture
    ),
    "--dt", $Dt.ToString(
        "G17",
        [Globalization.CultureInfo]::InvariantCulture
    )
)

if ($Mode -eq "Validate") {
    $arguments += "--validate-only"
}
else {
    if ([string]::IsNullOrWhiteSpace($Output)) {
        throw "-Output is required in Export mode."
    }
    $outputContainer = Convert-ToContainerPath `
        -Path $Output `
        -MustExist $false
    $arguments += @("--output", $outputContainer)
    if ($Overwrite) {
        $arguments += "--overwrite"
    }
}

& docker run --rm `
    --mount "type=bind,source=$bundleRoot,target=/workspace" `
    --workdir /workspace `
    $Image @arguments

if ($LASTEXITCODE -ne 0) {
    throw "OpenDSS exporter failed with exit code $LASTEXITCODE"
}

