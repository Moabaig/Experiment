# PAPER 1 V5.1 MECHANICAL-VALIDATION OVERLAY
#
# Purpose: estimator/schema/mechanical validation only.
# Output prefix: paper1_v5_1mv_
# This runner does not authorize confirmatory inference.
# The v4 calibration and matched-FAR thresholds are plumbing fixtures only.
# Performance outcomes must not be inspected or used for tuning.
[CmdletBinding()]
param(
    [ValidateSet("Validate", "Truth", "Run", "All")]
    [string]$Mode = "Validate",

    [ValidateRange(2, 31)]
    [int]$SeedFrom = 2,

    [ValidateRange(2, 31)]
    [int]$SeedTo = 31,

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
    $cellRecordPath = Join-Path $root "cell_record.paper1.v5_1.mechanical.json"
    if (-not (Test-Path -LiteralPath $cellRecordPath)) {
        throw "Existing cell lacks its V5.1 provenance record: $cellRecordPath"
    }
    $cellRecord = Get-Content -LiteralPath $cellRecordPath -Raw |
        ConvertFrom-Json

    if (
        $net.bandwidth_level -ne $BandwidthLevel -or
        [double]($net.bandwidth_cap_bps) -ne $BandwidthCap -or
        $twin.factor_design.bandwidth_level -ne $BandwidthLevel -or
        [double]($twin.factor_design.bandwidth_cap_bps) -ne $BandwidthCap -or
        $cellRecord.schema -ne "twin.factor.cell.record.paper1.v5_1.mechanical" -or
        $cellRecord.factor_design_sha256 -ne $script:DesignHash -or
        $cellRecord.matched_far_threshold_sha256 -ne $script:ThresholdHash -or
        $cellRecord.trust_metric_sha256 -ne $script:TrustMetricHash -or
        $cellRecord.twin_fed_sha256 -ne $script:TwinFederateHash
    ) {
        throw "Existing cell has conflicting V5.1 metadata: $RunId"
    }

    return $true
}

$designPath = ".\factor_design.paper1.v4.json"
if (-not (Test-Path -LiteralPath $designPath)) {
    throw "Missing factor design: $designPath"
}
$design = Get-Content -LiteralPath $designPath -Raw | ConvertFrom-Json
$script:DesignHash = Get-HashLower -Path $designPath

if (
    $design.schema -ne "twin.factor.design.paper1.v4" -or
    [int]$design.seed_policy.count -ne 30 -or
    [int]$design.campaign_cells -ne 150 -or
    @($design.bandwidth_levels).Count -ne 5
) {
    throw "factor_design.paper1.v4.json has an invalid contract."
}

$confirmatorySeeds = @(
    $design.seed_policy.confirmatory_seed_indices |
        ForEach-Object { [int]$_ }
)

if (
    $confirmatorySeeds.Count -ne 30 -or
    $confirmatorySeeds[0] -ne 2 -or
    $confirmatorySeeds[-1] -ne 31 -or
    $confirmatorySeeds -contains 1
) {
    throw "The v4 confirmatory seed contract is invalid."
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

$thresholdPath = ".\paper1_matched_far_thresholds.v1.json"
if (-not (Test-Path -LiteralPath $thresholdPath)) {
    throw (
        "Matched-FAR thresholds are not frozen. Run " +
        "paper1_diagnostics_v1\freeze_paper1_thresholds.py first."
    )
}

$thresholdContract = Get-Content -LiteralPath $thresholdPath -Raw |
    ConvertFrom-Json

if (
    $thresholdContract.schema -ne "paper1.matched_far.thresholds.v1" -or
    $thresholdContract.source_split -ne "calibration_only" -or
    [double]$thresholdContract.target_far -ne 0.01 -or
    $thresholdContract.quantile_method -ne "higher" -or
    $null -eq $thresholdContract.thresholds.s -or
    $null -eq $thresholdContract.thresholds.chi2
) {
    throw "paper1_matched_far_thresholds.v1.json has an invalid contract."
}

$thresholdHash = Get-HashLower -Path $thresholdPath
$script:ThresholdHash = $thresholdHash

$campaignRoot = ".\frozen\paper1_factor_campaign_v5_1_mechanical"
New-Item -ItemType Directory -Path $campaignRoot -Force | Out-Null

$expectedTrustMetricHash = "0a2627bdaacad03e582bb039eeb2fb3ac73d33d20b77e96881ebceec64aae437"
$expectedTwinFederateHash = "39e6729af233032ab9c58851c9682252f02d36eed739eb2ec769e165659da34c"

Assert-Hash `
    -Path ".\trust_metric.py" `
    -Expected $expectedTrustMetricHash

Assert-Hash `
    -Path ".\twin_fed.py" `
    -Expected $expectedTwinFederateHash

$script:TrustMetricHash = Get-HashLower -Path ".\trust_metric.py"
$script:TwinFederateHash = Get-HashLower -Path ".\twin_fed.py"

$implementationLockPath = Join-Path `
    $campaignRoot `
    "V5_1_IMPLEMENTATION_SHA256.json"

$implementationLock = [ordered]@{
    schema = "paper1.v5_1.mechanical.implementation.v1"
    trust_metric_sha256 = $script:TrustMetricHash
    twin_fed_sha256 = $script:TwinFederateHash
    solver = "weighted_lstsq_svd"
    normal_equations_used = $false
    jump_guard_policy = "fixed_model_increment"
    full_campaign_authorized = $false
    performance_outcomes_inspected = $false
}

if (Test-Path -LiteralPath $implementationLockPath) {
    $existingImplementationLock = Get-Content `
        -LiteralPath $implementationLockPath `
        -Raw |
        ConvertFrom-Json

    if (
        $existingImplementationLock.schema -ne
            "paper1.v5_1.mechanical.implementation.v1" -or
        $existingImplementationLock.trust_metric_sha256 -ne
            $script:TrustMetricHash -or
        $existingImplementationLock.twin_fed_sha256 -ne
            $script:TwinFederateHash
    ) {
        throw "V5.1 implementation lock conflicts with installed files."
    }
}
else {
    $implementationLock |
        ConvertTo-Json -Depth 6 |
        Set-Content `
            -LiteralPath $implementationLockPath `
            -Encoding ASCII
}

$thresholdLockPath = Join-Path $campaignRoot "MATCHED_FAR_THRESHOLD_SHA256.txt"
if (Test-Path -LiteralPath $thresholdLockPath) {
    $lockedThresholdHash = (
        Get-Content -LiteralPath $thresholdLockPath -Raw
    ).Trim().ToLowerInvariant()
    if ($lockedThresholdHash -ne $thresholdHash) {
        throw (
            "Matched-FAR threshold hash changed after the campaign lock: " +
            "$lockedThresholdHash -> $thresholdHash"
        )
    }
}
else {
    $thresholdHash |
        Set-Content -LiteralPath $thresholdLockPath -Encoding ASCII
}
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

    foreach ($testPattern in @(
        "test_regressions.py",
        "test_factor_extension.py"
    )) {
        docker compose --profile cosim run --rm --no-deps dev `
            python -m unittest discover `
            -s /workspace/tests `
            -p $testPattern `
            -v

        if ($LASTEXITCODE -ne 0) {
            throw "Paper-1 v4 test failed: $testPattern"
        }
    }

    docker compose --profile cosim run --rm --no-deps dev `
        python /workspace/verify_factor_extension_v3.py
    if ($LASTEXITCODE -ne 0) {
        throw "Factor-v3 static verification failed."
    }

    docker compose --profile cosim run --rm --no-deps dev `
        python /workspace/verify_paper1_factor_design_v4.py
    if ($LASTEXITCODE -ne 0) {
        throw "Paper-1 v4 design verification failed."
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

    "PAPER1_V5_1_MECHANICAL_PREFLIGHT_OK"
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
        $truthName = "truth.eval.paper1.v4.seed$seedText.npz"
        $truthPath = ".\$truthName"
        $truthRole = "eval.paper1.v4.seed$seedText"

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
                $runId = "paper1_v5_1mv_s$seedText`_$($level.id)"
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

                $cellRoot = ".\runs\$runId"
                $cellRecord = [PSCustomObject]@{
                    schema = "twin.factor.cell.record.paper1.v5_1.mechanical"
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
                    matched_far_threshold_sha256 = $thresholdHash
                    estimator_version = "paper1_v5_1"
                    trust_metric_sha256 = $script:TrustMetricHash
                    twin_fed_sha256 = $script:TwinFederateHash
                    estimator_solver = "weighted_lstsq_svd"
                    normal_equations_used = $false
                    jump_guard_policy = "fixed_model_increment"
                    mechanical_validation_only = $true
                    performance_outcomes_inspected = $false
                    qualification_seed_excluded = $true
                    status = "complete"
                }
                $cellRecord |
                    ConvertTo-Json -Depth 8 |
                    Set-Content `
                        -LiteralPath "$cellRoot\cell_record.paper1.v5_1.mechanical.json" `
                        -Encoding UTF8

                if (-not (Test-CompletedCell `
                    -RunId $runId `
                    -BandwidthLevel $level.id `
                    -BandwidthCap ([double]$level.bandwidth_cap_bps))) {
                    throw "Completed-cell validation unexpectedly failed: $runId"
                }

                $cellFiles = @(
                    Get-ChildItem -LiteralPath $cellRoot -Recurse -File |
                        Where-Object {
                            $_.Name -ne "CELL_OUTPUT_SHA256SUMS.csv"
                        }
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

                "FACTOR_CELL_COMPLETE run=$runId"
            }
        }
    }
}
finally {
    $originalEnv | Set-Content -LiteralPath ".\.env" -Encoding ASCII
    docker compose --profile cosim down --remove-orphans | Out-Host
}

"PAPER1_V5_1_MECHANICAL_BATCH_COMPLETE"
"MODE=$Mode"
"SEED_RANGE=$SeedFrom-$SeedTo"
"BANDWIDTH_RANGE=$BandwidthFrom-$BandwidthTo"
