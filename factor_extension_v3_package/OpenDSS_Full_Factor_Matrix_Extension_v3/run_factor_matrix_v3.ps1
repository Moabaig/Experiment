[CmdletBinding()]
param(
    [ValidateSet("Validate", "Truth", "Run", "All")]
    [string]$Mode = "Validate",

    [ValidateRange(1, 30)]
    [int]$SeedFrom = 1,

    [ValidateRange(1, 30)]
    [int]$SeedTo = 30,

    [ValidateRange(0, 4)]
    [int]$BandwidthFrom = 0,

    [ValidateRange(0, 4)]
    [int]$BandwidthTo = 4
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($SeedFrom -gt $SeedTo) {
    throw "SeedFrom must not exceed SeedTo."
}
if ($BandwidthFrom -gt $BandwidthTo) {
    throw "BandwidthFrom must not exceed BandwidthTo."
}

function Get-HashLower {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}

function Assert-Hash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required frozen input is missing: $Path"
    }
    $actual = Get-HashLower -Path $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "Frozen-input hash mismatch for $Path`: $actual"
    }
}

function Test-CompletedCell {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$BandwidthLevel,
        [Parameter(Mandatory = $true)][double]$BandwidthCap
    )

    $root = ".\runs\$RunId"
    if (-not (Test-Path -LiteralPath $root)) {
        return $false
    }

    foreach ($service in @("power", "net", "twin", "oracle")) {
        $metaPath = Join-Path $root "$service\meta.json"
        if (-not (Test-Path -LiteralPath $metaPath)) {
            throw "Existing cell is incomplete: $metaPath"
        }
        $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
        if ($meta.status -ne "complete") {
            throw "Existing cell is not complete: $metaPath"
        }
    }

    $net = Get-Content -LiteralPath "$root\net\meta.json" -Raw |
        ConvertFrom-Json
    $twin = Get-Content -LiteralPath "$root\twin\meta.json" -Raw |
        ConvertFrom-Json

    if (
        $net.bandwidth_level -ne $BandwidthLevel -or
        [double]($net.bandwidth_cap_bps) -ne $BandwidthCap -or
        $twin.factor_design.bandwidth_level -ne $BandwidthLevel -or
        [double]($twin.factor_design.bandwidth_cap_bps) -ne $BandwidthCap
    ) {
        throw "Existing cell has conflicting bandwidth metadata: $RunId"
    }

    return $true
}

$designPath = ".\factor_design.production.v3.json"
if (-not (Test-Path -LiteralPath $designPath)) {
    throw "Missing factor design: $designPath"
}
$design = Get-Content -LiteralPath $designPath -Raw | ConvertFrom-Json

if (
    $design.schema -ne "twin.factor.design.v3" -or
    [int]$design.seed_policy.count -ne 30 -or
    [int]$design.campaign_cells -ne 150 -or
    @($design.bandwidth_levels).Count -ne 5
) {
    throw "factor_design.production.v3.json has an invalid contract."
}

Assert-Hash `
    -Path ".\calibration.v2.json" `
    -Expected $design.frozen_inputs.calibration_sha256
Assert-Hash `
    -Path ".\gamma.frozen.v2.txt" `
    -Expected $design.frozen_inputs.gamma_sha256
Assert-Hash `
    -Path ".\W.frozen.v2.npy" `
    -Expected $design.frozen_inputs.weight_sha256
Assert-Hash `
    -Path ".\physical_design.production.v1.json" `
    -Expected $design.frozen_inputs.physical_design_sha256

$gamma = (Get-Content -LiteralPath ".\gamma.frozen.v2.txt" -Raw).Trim()
$gammaNumber = 0.0
if (
    -not [double]::TryParse($gamma, [ref]$gammaNumber) -or
    [double]::IsNaN($gammaNumber) -or
    [double]::IsInfinity($gammaNumber) -or
    $gammaNumber -le 0.0
) {
    throw "Frozen gamma is invalid."
}

$campaignRoot = ".\frozen\factor_v3_campaign"
New-Item -ItemType Directory -Path $campaignRoot -Force | Out-Null
$originalEnvPath = Join-Path $campaignRoot "original.env"
if (-not (Test-Path -LiteralPath $originalEnvPath)) {
    Copy-Item -LiteralPath ".\.env" -Destination $originalEnvPath
}
$originalEnv = Get-Content -LiteralPath $originalEnvPath -Raw

if ($Mode -eq "Validate") {
    docker compose config --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose configuration validation failed."
    }

    docker compose build dev
    if ($LASTEXITCODE -ne 0) {
        throw "Factor-v3 image rebuild failed."
    }

    docker compose --profile cosim run --rm --no-deps dev `
        python -m unittest discover `
        -s /workspace/tests `
        -p "test_*.py" `
        -v
    if ($LASTEXITCODE -ne 0) {
        throw "Factor-v3 regression tests failed."
    }

    docker compose --profile cosim run --rm --no-deps dev `
        python /workspace/verify_factor_extension_v3.py
    if ($LASTEXITCODE -ne 0) {
        throw "Factor-v3 static verification failed."
    }

    foreach ($level in @($design.bandwidth_levels)) {
        docker compose --profile cosim run --rm --no-deps dev `
            python /workspace/twin_fed.py `
            --feeder /workspace/feeder.npz `
            --patterns /workspace/patterns.npz `
            --scenarios /workspace/scenarios.csv `
            --calibration /workspace/calibration.v2.json `
            --bandwidth-level $level.id `
            --bandwidth-cap-bps $level.bandwidth_cap_bps `
            --compute-delta-check `
            --validate-only
        if ($LASTEXITCODE -ne 0) {
            throw "Twin rejected bandwidth level $($level.id)."
        }
    }

    "FACTOR_V3_FULL_PREFLIGHT_OK"
    "DESIGN_SHA256=$(Get-HashLower -Path $designPath)"
    "SEEDS=30"
    "BANDWIDTH_LEVELS=5"
    "CAMPAIGN_CELLS=150"
    exit 0
}

try {
    for ($seedIndex = $SeedFrom; $seedIndex -le $SeedTo; $seedIndex++) {
        $seedText = $seedIndex.ToString("000")
        $physicalSeed = 81000 + $seedIndex
        $powerSeed = 12000 + $seedIndex
        $networkSeed = 22000 + $seedIndex
        $truthName = "truth.eval.factor.v3.seed$seedText.npz"
        $truthPath = ".\$truthName"
        $truthRole = "eval.factor.v3.seed$seedText"

        if ($Mode -eq "Truth" -or $Mode -eq "All") {
            if (-not (Test-Path -LiteralPath $truthPath)) {
                powershell.exe `
                    -NoProfile `
                    -ExecutionPolicy Bypass `
                    -File .\run_opendss_exporter.ps1 `
                    -Mode Export `
                    -WeightSource .\W.frozen.v2.npy `
                    -Role $truthRole `
                    -Seed $physicalSeed `
                    -Output $truthPath
                if ($LASTEXITCODE -ne 0) {
                    throw "Truth export failed for seed $seedIndex."
                }
            }

            powershell.exe `
                -NoProfile `
                -ExecutionPolicy Bypass `
                -File .\validate_truth.ps1 `
                -Truth $truthPath `
                -Role $truthRole `
                -Seed $physicalSeed
            if ($LASTEXITCODE -ne 0) {
                throw "Truth validation failed for seed $seedIndex."
            }
        }

        if ($Mode -eq "Run" -or $Mode -eq "All") {
            if (-not (Test-Path -LiteralPath $truthPath)) {
                throw "Evaluation truth is missing: $truthPath"
            }

            for (
                $bandwidthIndex = $BandwidthFrom;
                $bandwidthIndex -le $BandwidthTo;
                $bandwidthIndex++
            ) {
                $level = @($design.bandwidth_levels)[$bandwidthIndex]
                $runId = "factor_v3_s$seedText`_$($level.id)"
                $capText = ([double]$level.bandwidth_cap_bps).ToString(
                    "R",
                    [System.Globalization.CultureInfo]::InvariantCulture
                )

                if (Test-CompletedCell `
                    -RunId $runId `
                    -BandwidthLevel $level.id `
                    -BandwidthCap ([double]$level.bandwidth_cap_bps)) {
                    "FACTOR_CELL_ALREADY_COMPLETE run=$runId"
                    continue
                }

                $envLines = @(
                    "RUN_ID=$runId",
                    "TRUTH_FILE=$truthName",
                    "CALIBRATION_FILE=calibration.v2.json",
                    "CALIBRATION_MODE=0",
                    "POWER_SEED=$powerSeed",
                    "NETWORK_SEED=$networkSeed",
                    "NETWORK_RUN=1",
                    "BANDWIDTH_LEVEL=$($level.id)",
                    "BANDWIDTH_CAP_BPS=$capText",
                    "DRIFT_GAMMA=$gamma",
                    "STOP_TIME=0",
                    "ALLOW_LINEARIZED_TELEMETRY=0"
                )
                $envLines | Set-Content -LiteralPath ".\.env" -Encoding ASCII

                docker compose --profile cosim down --remove-orphans
                if ($LASTEXITCODE -ne 0) {
                    throw "Compose cleanup failed before $runId."
                }

                powershell.exe `
                    -NoProfile `
                    -ExecutionPolicy Bypass `
                    -File .\run_experiment.ps1
                if ($LASTEXITCODE -ne 0) {
                    throw "Factor cell failed: $runId"
                }

                if (-not (Test-CompletedCell `
                    -RunId $runId `
                    -BandwidthLevel $level.id `
                    -BandwidthCap ([double]$level.bandwidth_cap_bps))) {
                    throw "Completed-cell validation unexpectedly failed: $runId"
                }

                $cellRoot = ".\runs\$runId"
                $cellFiles = @(
                    Get-ChildItem -LiteralPath $cellRoot -Recurse -File
                )
                $cellHashes = foreach ($file in $cellFiles) {
                    Get-FileHash `
                        -LiteralPath $file.FullName `
                        -Algorithm SHA256
                }
                $cellHashes |
                    Export-Csv `
                        -LiteralPath "$cellRoot\CELL_OUTPUT_SHA256SUMS.csv" `
                        -NoTypeInformation

                $cellRecord = [PSCustomObject]@{
                    schema = "twin.factor.cell.record.v3"
                    run_id = $runId
                    seed_index = $seedIndex
                    physical_seed = $physicalSeed
                    power_seed = $powerSeed
                    network_seed = $networkSeed
                    network_run = 1
                    bandwidth_level = $level.id
                    bandwidth_cap_bps = [double]$level.bandwidth_cap_bps
                    truth_file = $truthName
                    truth_sha256 = Get-HashLower -Path $truthPath
                    factor_design_sha256 = Get-HashLower -Path $designPath
                    calibration_sha256 = Get-HashLower -Path ".\calibration.v2.json"
                    gamma_sha256 = Get-HashLower -Path ".\gamma.frozen.v2.txt"
                    status = "complete"
                }
                $cellRecord |
                    ConvertTo-Json -Depth 8 |
                    Set-Content `
                        -LiteralPath "$cellRoot\cell_record.v3.json" `
                        -Encoding UTF8

                "FACTOR_CELL_COMPLETE run=$runId"
            }
        }
    }
}
finally {
    $originalEnv | Set-Content -LiteralPath ".\.env" -Encoding ASCII
    docker compose --profile cosim down --remove-orphans | Out-Host
}

"FACTOR_V3_BATCH_COMPLETE"
"MODE=$Mode"
"SEED_RANGE=$SeedFrom-$SeedTo"
"BANDWIDTH_RANGE=$BandwidthFrom-$BandwidthTo"
