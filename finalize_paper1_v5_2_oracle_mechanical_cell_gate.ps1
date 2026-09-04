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
$evidenceRoot = Join-Path $projectRoot "frozen\v52cell\C20260903_174122"
$runLog = Join-Path $evidenceRoot "paper1_v5_2_oracle_mechanical_cell_20260903_174122.log"
$preRunEnvPath = Join-Path $evidenceRoot "pre_run.env"
$temporaryEnvPath = Join-Path $evidenceRoot "postrun_before_wrapper_restore.env"
$finalizerPath = $MyInvocation.MyCommand.Path

$expectedRunnerHash = "5753DE4E708206D0A1F8669ADC52FA74BA5E4395318A24CCFBFAD6C1FCEB6629"
$expectedContractHash = "11B20715BD970988A25429BA645373382671C8D5713CBA5A836705F83B09256C"
$expectedBuildReportHash = "7F7ED1A089D3592F0B68733E46BC76D93F751A02A875705D8DC3DFD825C0FAEB"
$expectedTrustHash = "936DD373A2D8A2F0B905604CA4C3DE61EC2CC889BA233AA150A24F44F2926FE6"
$expectedTwinHash = "9CD9FFAA32DCFE2F12ED161A8D62D2D97B2AB0B4D462FDA0E97E7F46572043A4"
$expectedEnvHash = "55A4FCB1ACB19D86CBE2DA4BCC4FE814170A14A5A637EC6CEC97D9C94195D694"

$expectedParquetHashes = [ordered]@{
    "oracle\oracle_events.parquet" = "B754CC1CAFF4E2D67604D89D53F8D76AA54681387B4188D7A070B835C2FD74D8"
    "oracle\oracle_scores.parquet" = "80C2FBB446DCBF0F2FDE6131EBD35FE1CD69AECD3E4C8B1BF79C1A7736DE33C8"
    "twin\scores.parquet" = "ADAD69F15D4478EFC5698C01C34FB7AD7B2DB4D078F87DBFEC55519BCF50EEE1"
    "twin\scores.partial.parquet" = "ADAD69F15D4478EFC5698C01C34FB7AD7B2DB4D078F87DBFEC55519BCF50EEE1"
    "twin\scores_events.parquet" = "8BD205AA1F59AF3CEB19C83DA260EF43366C77ECF30919337B92A398CCB14447"
    "twin\scores_events.partial.parquet" = "8BD205AA1F59AF3CEB19C83DA260EF43366C77ECF30919337B92A398CCB14447"
}

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

function Copy-EvidenceFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $sourceHash = Get-Sha256 -Path $Source
        $destinationHash = Get-Sha256 -Path $Destination
        if ($sourceHash -ne $destinationHash) {
            throw "Existing evidence conflicts with source: $Destination"
        }
        return
    }

    Copy-Item -LiteralPath $Source -Destination $Destination
}

Assert-Sha256 -Path $runner -Expected $expectedRunnerHash
Assert-Sha256 -Path $contractPath -Expected $expectedContractHash
Assert-Sha256 -Path $buildReportPath -Expected $expectedBuildReportHash
Assert-Sha256 -Path $trustPath -Expected $expectedTrustHash
Assert-Sha256 -Path $twinPath -Expected $expectedTwinHash
Assert-Sha256 -Path $envPath -Expected $expectedEnvHash
Assert-Sha256 -Path $preRunEnvPath -Expected $expectedEnvHash

if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) {
    throw "Completed V5.2 run directory is missing: $runRoot"
}
if (-not (Test-Path -LiteralPath $evidenceRoot -PathType Container)) {
    throw "Original V5.2 evidence directory is missing: $evidenceRoot"
}

$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
if (@($contract.authorized_cells).Count -ne 1) {
    throw "The V5.2 contract must authorize exactly one mechanical cell."
}
$authorizedCell = @($contract.authorized_cells)[0]
if (
    $authorizedCell.run_id -ne $runId -or
    [int]$authorizedCell.seed_from -ne 2 -or
    [int]$authorizedCell.seed_to -ne 2 -or
    [int]$authorizedCell.bandwidth_from -ne 4 -or
    [int]$authorizedCell.bandwidth_to -ne 4
) {
    throw "The V5.2 contract does not authorize this exact single cell."
}
if (
    [bool]$contract.authorization.full_campaign_authorized -or
    [bool]$contract.authorization.calibration_authorized -or
    [bool]$contract.authorization.performance_outcome_columns_may_be_read -or
    [bool]$contract.authorization.performance_outcomes_may_be_inspected
) {
    throw "The V5.2 contract exceeds the mechanical-validation boundary."
}

$runOutput = Get-Content -LiteralPath $runLog -Raw
$requiredMarkers = @(
    "power_fed complete:",
    "net_fed complete:",
    "oracle_fed complete:",
    "twin_fed complete:",
    "CO_SIMULATION_COMPLETE run=$runId",
    "FACTOR_CELL_COMPLETE run=$runId",
    "PAPER1_V5_2_MECHANICAL_BATCH_COMPLETE",
    "MODE=Run",
    "SEED_RANGE=2-2",
    "BANDWIDTH_RANGE=4-4"
)
foreach ($marker in $requiredMarkers) {
    if (-not $runOutput.Contains($marker)) {
        throw "Required V5.2 completion marker is missing: $marker"
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
foreach ($entry in $requiredOutputs.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
        throw "Required V5.2 output is missing: $($entry.Value)"
    }
}

foreach ($metaKey in @("power_meta", "net_meta", "twin_meta", "oracle_meta")) {
    $null = Get-Content -LiteralPath $requiredOutputs[$metaKey] -Raw |
        ConvertFrom-Json
}

$cellRecordPath = Join-Path $runRoot "cell_record.paper1.v5_2.mechanical.json"
$cellRecord = Get-Content -LiteralPath $cellRecordPath -Raw | ConvertFrom-Json
if (
    $cellRecord.schema -ne "twin.factor.cell.record.paper1.v5_2.mechanical" -or
    $cellRecord.run_id -ne $runId -or
    $cellRecord.status -ne "complete"
) {
    throw "The V5.2 cell record is not a completed record for the authorized run."
}

$cellManifestPath = Join-Path $runRoot "CELL_OUTPUT_SHA256SUMS.csv"
$cellManifestRows = @(Import-Csv -LiteralPath $cellManifestPath)
if ($cellManifestRows.Count -ne 11) {
    throw "Unexpected cell-output manifest row count: $($cellManifestRows.Count)"
}

$runFiles = @(Get-ChildItem -LiteralPath $runRoot -Recurse -File)
if ($runFiles.Count -ne 12) {
    throw "Unexpected cell-output file count: $($runFiles.Count)"
}

$normalizedRunRoot = [System.IO.Path]::GetFullPath($runRoot).TrimEnd('\') + '\'
$manifestPaths = New-Object `
    -TypeName "System.Collections.Generic.HashSet[string]" `
    -ArgumentList ([System.StringComparer]::OrdinalIgnoreCase)

foreach ($row in $cellManifestRows) {
    if ([string]::IsNullOrWhiteSpace($row.Path)) {
        throw "Cell-output manifest contains an empty path."
    }
    if ([string]::IsNullOrWhiteSpace($row.Hash)) {
        throw "Cell-output manifest contains an empty hash."
    }

    if ([System.IO.Path]::IsPathRooted($row.Path)) {
        $manifestFilePath = [System.IO.Path]::GetFullPath($row.Path)
    }
    else {
        $manifestFilePath = [System.IO.Path]::GetFullPath(
            (Join-Path $runRoot $row.Path)
        )
    }

    if (-not $manifestFilePath.StartsWith(
        $normalizedRunRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Cell-output manifest path escapes the run directory: $manifestFilePath"
    }
    if (-not $manifestPaths.Add($manifestFilePath)) {
        throw "Cell-output manifest contains a duplicate path: $manifestFilePath"
    }

    $observedHash = Get-Sha256 -Path $manifestFilePath
    if ($observedHash -ne $row.Hash.ToUpperInvariant()) {
        throw "Cell-output hash mismatch for $manifestFilePath : $observedHash"
    }
}

foreach ($file in $runFiles) {
    if ($file.FullName -eq $cellManifestPath) {
        continue
    }
    if (-not $manifestPaths.Contains($file.FullName)) {
        throw "Cell-output file is absent from the manifest: $($file.FullName)"
    }
}

foreach ($entry in $expectedParquetHashes.GetEnumerator()) {
    $path = Join-Path $runRoot $entry.Key
    Assert-Sha256 -Path $path -Expected $entry.Value
}

$finalTwinScoresHash = Get-Sha256 -Path (
    Join-Path $runRoot "twin\scores.parquet"
)
$partialTwinScoresHash = Get-Sha256 -Path (
    Join-Path $runRoot "twin\scores.partial.parquet"
)
$finalTwinEventsHash = Get-Sha256 -Path (
    Join-Path $runRoot "twin\scores_events.parquet"
)
$partialTwinEventsHash = Get-Sha256 -Path (
    Join-Path $runRoot "twin\scores_events.partial.parquet"
)
if ($finalTwinScoresHash -ne $partialTwinScoresHash) {
    throw "Twin score final and partial files are not byte-identical."
}
if ($finalTwinEventsHash -ne $partialTwinEventsHash) {
    throw "Twin event final and partial files are not byte-identical."
}

$recordPath = Join-Path $evidenceRoot "paper1_v5_2_oracle_mechanical_cell_record.json"
$evidenceManifestPath = Join-Path $evidenceRoot "PAPER1_V5_2_ORACLE_CELL_GATE_SHA256.csv"
if (
    (Test-Path -LiteralPath $recordPath) -or
    (Test-Path -LiteralPath $evidenceManifestPath)
) {
    throw "The V5.2 oracle cell evidence has already been finalized."
}

Copy-EvidenceFile -Source $runner -Destination (
    Join-Path $evidenceRoot (Split-Path $runner -Leaf)
)
Copy-EvidenceFile -Source $contractPath -Destination (
    Join-Path $evidenceRoot (Split-Path $contractPath -Leaf)
)
Copy-EvidenceFile -Source $buildReportPath -Destination (
    Join-Path $evidenceRoot (Split-Path $buildReportPath -Leaf)
)
Copy-EvidenceFile -Source $cellManifestPath -Destination (
    Join-Path $evidenceRoot (Split-Path $cellManifestPath -Leaf)
)
Copy-EvidenceFile -Source $cellRecordPath -Destination (
    Join-Path $evidenceRoot (Split-Path $cellRecordPath -Leaf)
)
Copy-EvidenceFile -Source $requiredOutputs.power_meta -Destination (
    Join-Path $evidenceRoot "power_meta.json"
)
Copy-EvidenceFile -Source $requiredOutputs.net_meta -Destination (
    Join-Path $evidenceRoot "net_meta.json"
)
Copy-EvidenceFile -Source $requiredOutputs.twin_meta -Destination (
    Join-Path $evidenceRoot "twin_meta.json"
)
Copy-EvidenceFile -Source $requiredOutputs.oracle_meta -Destination (
    Join-Path $evidenceRoot "oracle_meta.json"
)
Copy-EvidenceFile -Source $finalizerPath -Destination (
    Join-Path $evidenceRoot (Split-Path $finalizerPath -Leaf)
)

$outputHashes = [ordered]@{}
foreach ($entry in $requiredOutputs.GetEnumerator()) {
    $outputHashes[$entry.Key] = (
        Get-Sha256 -Path $entry.Value
    ).ToLowerInvariant()
}

$temporaryEnvHash = $null
$envRestoredByWrapper = $false
if (Test-Path -LiteralPath $temporaryEnvPath -PathType Leaf) {
    $temporaryEnvHash = (Get-Sha256 -Path $temporaryEnvPath).ToLowerInvariant()
    $envRestoredByWrapper = $true
}

$record = [ordered]@{
    schema = "paper1.v5_2.oracle_mechanical_cell_gate.v1"
    run_id = $runId
    gate_correction = "oracle_file_names"
    finalizer_sha256 = (Get-Sha256 -Path $finalizerPath).ToLowerInvariant()
    runner_sha256 = $expectedRunnerHash.ToLowerInvariant()
    contract_sha256 = $expectedContractHash.ToLowerInvariant()
    build_report_sha256 = $expectedBuildReportHash.ToLowerInvariant()
    installed_trust_sha256 = $expectedTrustHash.ToLowerInvariant()
    installed_twin_sha256 = $expectedTwinHash.ToLowerInvariant()
    pre_run_env_sha256 = $expectedEnvHash.ToLowerInvariant()
    postrun_temporary_env_sha256 = $temporaryEnvHash
    env_restored_by_wrapper = $envRestoredByWrapper
    env_restored_exactly = $true
    output_file_count = $runFiles.Count
    required_output_count = $requiredOutputs.Count
    missing_output_count = 0
    cell_output_manifest_rows = $cellManifestRows.Count
    cell_output_manifest_sha256 = (
        Get-Sha256 -Path $cellManifestPath
    ).ToLowerInvariant()
    output_sha256 = $outputHashes
    twin_final_partial_scores_identical = $true
    twin_final_partial_events_identical = $true
    simulation_completed = $true
    simulation_rerun = $false
    implementation_modified = $false
    full_campaign_authorized = $false
    calibration_authorized = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}
$record |
    ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $recordPath -Encoding UTF8

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

"PAPER1_V5_2_ORACLE_MECHANICAL_CELL_FINALIZATION_OK"
"RUN_ID=$runId"
"GATE_CORRECTION=oracle_file_names"
"FINALIZER_SHA256=$((Get-FileHash -LiteralPath $finalizerPath -Algorithm SHA256).Hash)"
"RUNNER_SHA256=$expectedRunnerHash"
"CONTRACT_SHA256=$expectedContractHash"
"INSTALLED_TRUST_SHA256=$expectedTrustHash"
"INSTALLED_TWIN_SHA256=$expectedTwinHash"
"OUTPUT_FILE_COUNT=$($runFiles.Count)"
"REQUIRED_OUTPUT_COUNT=$($requiredOutputs.Count)"
"MISSING_OUTPUT_COUNT=0"
"CELL_OUTPUT_MANIFEST_ROWS=$($cellManifestRows.Count)"
"CELL_OUTPUT_MANIFEST_INTEGRITY_OK=True"
"PARQUET_HASHES_VERIFIED=True"
"TWIN_FINAL_PARTIAL_SCORES_IDENTICAL=True"
"TWIN_FINAL_PARTIAL_EVENTS_IDENTICAL=True"
"ENV_RESTORED_BY_WRAPPER=$envRestoredByWrapper"
"ENV_RESTORED_EXACTLY=True"
"GATE_RECORD_SHA256=$((Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash)"
"EVIDENCE_MANIFEST_SHA256=$((Get-FileHash -LiteralPath $evidenceManifestPath -Algorithm SHA256).Hash)"
"EVIDENCE_ROOT=$evidenceRoot"
"SIMULATION_COMPLETED=True"
"SIMULATION_RERUN=False"
"IMPLEMENTATION_MODIFIED=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_ORACLE_CELL_READY_FOR_MECHANICAL_AUDIT"
