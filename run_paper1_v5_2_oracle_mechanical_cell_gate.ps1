$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$runId = "paper1_v5_2mv_s002_bw04_oracle"
$runner = Join-Path $projectRoot "run_paper1_factor_campaign_v5_2_mechanical.ps1"
$contractPath = Join-Path $projectRoot "paper1_v5_2_repair\paper1_v5_2_mechanical_validation_contract.json"
$buildReportPath = Join-Path $projectRoot "paper1_v5_2_repair\v5_2_mechanical_runner_build.json"
$envPath = Join-Path $projectRoot ".env"
$trustPath = Join-Path $projectRoot "trust_metric.py"
$twinPath = Join-Path $projectRoot "twin_fed.py"
$runRoot = Join-Path $projectRoot "runs\$runId"

$expectedRunnerHash = "5753DE4E708206D0A1F8669ADC52FA74BA5E4395318A24CCFBFAD6C1FCEB6629"
$expectedContractHash = "11B20715BD970988A25429BA645373382671C8D5713CBA5A836705F83B09256C"
$expectedBuildReportHash = "7F7ED1A089D3592F0B68733E46BC76D93F751A02A875705D8DC3DFD825C0FAEB"
$expectedTrustHash = "936DD373A2D8A2F0B905604CA4C3DE61EC2CC889BA233AA150A24F44F2926FE6"
$expectedTwinHash = "9CD9FFAA32DCFE2F12ED161A8D62D2D97B2AB0B4D462FDA0E97E7F46572043A4"
$expectedEnvHash = "55A4FCB1ACB19D86CBE2DA4BCC4FE814170A14A5A637EC6CEC97D9C94195D694"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Assert-Sha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )

    $observed = Get-Sha256 -Path $Path
    if ($observed -ne $Expected) {
        throw "Hash mismatch for $Path : $observed"
    }
}

Assert-Sha256 -Path $runner -Expected $expectedRunnerHash
Assert-Sha256 -Path $contractPath -Expected $expectedContractHash
Assert-Sha256 -Path $buildReportPath -Expected $expectedBuildReportHash
Assert-Sha256 -Path $trustPath -Expected $expectedTrustHash
Assert-Sha256 -Path $twinPath -Expected $expectedTwinHash
Assert-Sha256 -Path $envPath -Expected $expectedEnvHash

$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
if ($contract.authorized_cells.Count -ne 1) {
    throw "The V5.2 contract must authorize exactly one mechanical cell."
}
$authorizedCell = $contract.authorized_cells[0]
if (
    $authorizedCell.run_id -ne $runId -or
    [int]$authorizedCell.seed_from -ne 2 -or
    [int]$authorizedCell.seed_to -ne 2 -or
    [int]$authorizedCell.bandwidth_from -ne 4 -or
    [int]$authorizedCell.bandwidth_to -ne 4
) {
    throw "The V5.2 contract does not authorize the requested single cell."
}
if (
    [bool]$contract.authorization.full_campaign_authorized -or
    [bool]$contract.authorization.calibration_authorized -or
    [bool]$contract.authorization.performance_outcome_columns_may_be_read -or
    [bool]$contract.authorization.performance_outcomes_may_be_inspected
) {
    throw "The V5.2 contract exceeds the mechanical-validation boundary."
}

if (Test-Path -LiteralPath $runRoot) {
    throw "The V5.2 mechanical run directory already exists. Stop for inspection: $runRoot"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evidenceRoot = Join-Path $projectRoot "frozen\v52cell\C$timestamp"
if (Test-Path -LiteralPath $evidenceRoot) {
    throw "Evidence destination already exists: $evidenceRoot"
}
New-Item -ItemType Directory -Path $evidenceRoot | Out-Null

$preRunEnvPath = Join-Path $evidenceRoot "pre_run.env"
$preRunEnvBytes = [System.IO.File]::ReadAllBytes($envPath)
[System.IO.File]::WriteAllBytes($preRunEnvPath, $preRunEnvBytes)
$preRunEnvHash = Get-Sha256 -Path $preRunEnvPath
if ($preRunEnvHash -ne $expectedEnvHash) {
    throw "The archived pre-run environment hash is incorrect."
}

$runLog = Join-Path $evidenceRoot "paper1_v5_2_oracle_mechanical_cell_$timestamp.log"
$runnerExitCode = -1
$envRestoredByWrapper = $false
$postRunTemporaryEnvHash = $null

"STARTING_PAPER1_V5_2_ORACLE_MECHANICAL_CELL"
"RUN_ID=$runId"
"RUNNER_SHA256=$expectedRunnerHash"
"CONTRACT_SHA256=$expectedContractHash"
"INSTALLED_TRUST_SHA256=$expectedTrustHash"
"INSTALLED_TWIN_SHA256=$expectedTwinHash"
"LOG=$runLog"
"AUTHORIZED_CELL_COUNT=1"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"Expected runtime on this host is approximately three hours."
"Monitor from a second PowerShell window with:"
"Get-Content -LiteralPath `"$runLog`" -Wait -Tail 30"

$runCommand = (
    'powershell.exe -NoProfile -ExecutionPolicy Bypass ' +
    '-File "{0}" -Mode Run ' +
    '-SeedFrom 2 -SeedTo 2 -BandwidthFrom 4 -BandwidthTo 4 ' +
    '> "{1}" 2>&1'
) -f $runner, $runLog

try {
    & cmd.exe /d /c $runCommand
    $runnerExitCode = $LASTEXITCODE
}
finally {
    $currentEnvHash = $null
    if (Test-Path -LiteralPath $envPath -PathType Leaf) {
        $currentEnvHash = Get-Sha256 -Path $envPath
    }

    if ($currentEnvHash -ne $preRunEnvHash) {
        if (Test-Path -LiteralPath $envPath -PathType Leaf) {
            $temporaryEnvPath = Join-Path $evidenceRoot "postrun_before_wrapper_restore.env"
            $temporaryEnvBytes = [System.IO.File]::ReadAllBytes($envPath)
            [System.IO.File]::WriteAllBytes(
                $temporaryEnvPath,
                $temporaryEnvBytes
            )
            $postRunTemporaryEnvHash = Get-Sha256 -Path $temporaryEnvPath
        }

        [System.IO.File]::WriteAllBytes($envPath, $preRunEnvBytes)
        $envRestoredByWrapper = $true
    }
}

Assert-Sha256 -Path $envPath -Expected $preRunEnvHash
Assert-Sha256 -Path $trustPath -Expected $expectedTrustHash
Assert-Sha256 -Path $twinPath -Expected $expectedTwinHash

if (Test-Path -LiteralPath $runLog -PathType Leaf) {
    Get-Content -LiteralPath $runLog -Tail 120 | Out-Host
}
if ($runnerExitCode -ne 0) {
    throw "V5.2 oracle mechanical cell failed. Inspect $runLog"
}

$runOutput = Get-Content -LiteralPath $runLog -Raw
$requiredMarkers = @(
    "FACTOR_CELL_COMPLETE run=$runId",
    "PAPER1_V5_2_MECHANICAL_BATCH_COMPLETE",
    "MODE=Run",
    "SEED_RANGE=2-2",
    "BANDWIDTH_RANGE=4-4"
)
foreach ($marker in $requiredMarkers) {
    if (-not $runOutput.Contains($marker)) {
        throw "Required V5.2 cell-completion marker is missing: $marker"
    }
}

$requiredOutputs = [ordered]@{
    power_meta = Join-Path $runRoot "power\meta.json"
    net_meta = Join-Path $runRoot "net\meta.json"
    twin_meta = Join-Path $runRoot "twin\meta.json"
    oracle_meta = Join-Path $runRoot "oracle\meta.json"
    twin_scores = Join-Path $runRoot "twin\scores.parquet"
    twin_events = Join-Path $runRoot "twin\scores_events.parquet"
    oracle_scores = Join-Path $runRoot "oracle\oracle_scores.parquet"
    oracle_events = Join-Path $runRoot "oracle\oracle_events.parquet"
}

$missingOutputs = @(
    foreach ($entry in $requiredOutputs.GetEnumerator()) {
        if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
            $entry.Value
        }
    }
)
if ($missingOutputs.Count -ne 0) {
    $missingOutputs | Out-Host
    throw "The V5.2 mechanical cell is missing required outputs."
}

foreach ($metaKey in @("power_meta", "net_meta", "twin_meta", "oracle_meta")) {
    $null = Get-Content -LiteralPath $requiredOutputs[$metaKey] -Raw |
        ConvertFrom-Json
}

$cellManifestPath = Join-Path $runRoot "CELL_OUTPUT_SHA256SUMS.csv"
if (-not (Test-Path -LiteralPath $cellManifestPath -PathType Leaf)) {
    throw "The cell-output SHA256 manifest is missing."
}
$cellManifestRows = @(Import-Csv -LiteralPath $cellManifestPath)
if ($cellManifestRows.Count -ne 11) {
    throw "Unexpected cell-output manifest row count: $($cellManifestRows.Count)"
}

$normalizedRunRoot = [System.IO.Path]::GetFullPath($runRoot).TrimEnd('\') + '\'
foreach ($row in $cellManifestRows) {
    if ([string]::IsNullOrWhiteSpace($row.Path)) {
        throw "Cell-output manifest contains an empty path."
    }
    if ([string]::IsNullOrWhiteSpace($row.Hash)) {
        throw "Cell-output manifest contains an empty hash."
    }

    $manifestFilePath = [System.IO.Path]::GetFullPath($row.Path)
    if (-not $manifestFilePath.StartsWith(
        $normalizedRunRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Cell-output manifest path escapes the run directory: $manifestFilePath"
    }

    $observedHash = Get-Sha256 -Path $manifestFilePath
    if ($observedHash -ne $row.Hash.ToUpperInvariant()) {
        throw "Cell-output hash mismatch for $manifestFilePath : $observedHash"
    }
}

$cellRecordCandidates = @(
    Get-ChildItem -LiteralPath $runRoot -Filter "cell_record*.json" -File
)
if ($cellRecordCandidates.Count -ne 1) {
    throw "Expected exactly one V5.2 cell record."
}
$cellRecord = Get-Content -LiteralPath $cellRecordCandidates[0].FullName -Raw |
    ConvertFrom-Json
if ($cellRecord.run_id -ne $runId) {
    throw "V5.2 cell record contains the wrong run id."
}

Copy-Item -LiteralPath $runner -Destination $evidenceRoot
Copy-Item -LiteralPath $contractPath -Destination $evidenceRoot
Copy-Item -LiteralPath $buildReportPath -Destination $evidenceRoot
Copy-Item -LiteralPath $cellManifestPath -Destination $evidenceRoot
Copy-Item -LiteralPath $cellRecordCandidates[0].FullName -Destination $evidenceRoot
Copy-Item -LiteralPath $requiredOutputs.power_meta -Destination (Join-Path $evidenceRoot "power_meta.json")
Copy-Item -LiteralPath $requiredOutputs.net_meta -Destination (Join-Path $evidenceRoot "net_meta.json")
Copy-Item -LiteralPath $requiredOutputs.twin_meta -Destination (Join-Path $evidenceRoot "twin_meta.json")
Copy-Item -LiteralPath $requiredOutputs.oracle_meta -Destination (Join-Path $evidenceRoot "oracle_meta.json")

$outputHashes = [ordered]@{}
foreach ($entry in $requiredOutputs.GetEnumerator()) {
    $outputHashes[$entry.Key] = (Get-Sha256 -Path $entry.Value).ToLowerInvariant()
}

$recordPath = Join-Path $evidenceRoot "paper1_v5_2_oracle_mechanical_cell_record.json"
$record = [ordered]@{
    schema = "paper1.v5_2.oracle_mechanical_cell_gate.v1"
    run_id = $runId
    runner_sha256 = $expectedRunnerHash.ToLowerInvariant()
    contract_sha256 = $expectedContractHash.ToLowerInvariant()
    installed_trust_sha256 = $expectedTrustHash.ToLowerInvariant()
    installed_twin_sha256 = $expectedTwinHash.ToLowerInvariant()
    pre_run_env_sha256 = $preRunEnvHash.ToLowerInvariant()
    postrun_temporary_env_sha256 = $postRunTemporaryEnvHash
    env_restored_by_wrapper = $envRestoredByWrapper
    env_restored_exactly = $true
    required_output_count = $requiredOutputs.Count
    missing_output_count = 0
    cell_output_manifest_rows = $cellManifestRows.Count
    cell_output_manifest_sha256 = (
        Get-Sha256 -Path $cellManifestPath
    ).ToLowerInvariant()
    output_sha256 = $outputHashes
    simulation_completed = $true
    simulation_rerun = $false
    full_campaign_authorized = $false
    calibration_authorized = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}
$record |
    ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $recordPath -Encoding UTF8

$evidenceManifestPath = Join-Path $evidenceRoot "PAPER1_V5_2_ORACLE_CELL_GATE_SHA256.csv"
$evidenceRows = @(
    Get-ChildItem -LiteralPath $evidenceRoot -File |
        Where-Object { $_.FullName -ne $evidenceManifestPath } |
        Sort-Object Name |
        ForEach-Object {
            [PSCustomObject]@{
                Name = $_.Name
                Length = $_.Length
                SHA256 = (
                    Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
                ).Hash
            }
        }
)
$evidenceRows |
    Export-Csv -LiteralPath $evidenceManifestPath -NoTypeInformation

"PAPER1_V5_2_ORACLE_MECHANICAL_CELL_COMPLETE"
"RUN_ID=$runId"
"RUNNER_SHA256=$expectedRunnerHash"
"CONTRACT_SHA256=$expectedContractHash"
"INSTALLED_TRUST_SHA256=$expectedTrustHash"
"INSTALLED_TWIN_SHA256=$expectedTwinHash"
"REQUIRED_OUTPUT_COUNT=$($requiredOutputs.Count)"
"MISSING_OUTPUT_COUNT=0"
"CELL_OUTPUT_MANIFEST_ROWS=$($cellManifestRows.Count)"
"CELL_OUTPUT_MANIFEST_INTEGRITY_OK=True"
"ENV_RESTORED_BY_WRAPPER=$envRestoredByWrapper"
"ENV_RESTORED_EXACTLY=True"
"GATE_RECORD_SHA256=$((Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash)"
"EVIDENCE_MANIFEST_SHA256=$((Get-FileHash -LiteralPath $evidenceManifestPath -Algorithm SHA256).Hash)"
"EVIDENCE_ROOT=$evidenceRoot"
"SIMULATION_COMPLETED=True"
"SIMULATION_RERUN=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_ORACLE_CELL_READY_FOR_MECHANICAL_AUDIT"
