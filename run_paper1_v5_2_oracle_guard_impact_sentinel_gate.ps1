$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$runId = "paper1_v5_2mv_s002_bw04_oracle"
$sentinelPath = Join-Path $projectRoot "analyze_paper1_v5_2_oracle_guard_impact_sentinel.py"
$contractPath = Join-Path $projectRoot "paper1_v5_2_oracle_guard_impact_sentinel_contract.json"
$trustPath = Join-Path $projectRoot "trust_metric.py"
$twinPath = Join-Path $projectRoot "twin_fed.py"
$envPath = Join-Path $projectRoot ".env"
$runRoot = Join-Path $projectRoot "runs\$runId"
$twinEventsPath = Join-Path $runRoot "twin\scores_events.parquet"
$oracleEventsPath = Join-Path $runRoot "oracle\oracle_events.parquet"
$mechanicalEvidenceRoot = Join-Path $projectRoot "frozen\v52audit\A20260903_223859"
$mechanicalReportPath = Join-Path $mechanicalEvidenceRoot "paper1_v5_2_oracle_mechanical_audit.json"
$mechanicalLogPath = Join-Path $mechanicalEvidenceRoot "paper1_v5_2_oracle_mechanical_audit.log"
$mechanicalGateRecordPath = Join-Path $mechanicalEvidenceRoot "paper1_v5_2_oracle_mechanical_audit_gate.json"
$mechanicalManifestPath = Join-Path $mechanicalEvidenceRoot "PAPER1_V5_2_MECHANICAL_AUDIT_SHA256.csv"

$expectedSentinelHash = "3AF3AC5259A475FDFD8B95460D97DB2B19237153879A23424920813DE5095475"
$expectedContractHash = "BD6489EA5B14D78B8825D48F129FAF440F7FA3DFC46DB118D74A5B5999EA43AE"
$expectedTrustHash = "936DD373A2D8A2F0B905604CA4C3DE61EC2CC889BA233AA150A24F44F2926FE6"
$expectedTwinHash = "9CD9FFAA32DCFE2F12ED161A8D62D2D97B2AB0B4D462FDA0E97E7F46572043A4"
$expectedEnvHash = "55A4FCB1ACB19D86CBE2DA4BCC4FE814170A14A5A637EC6CEC97D9C94195D694"
$expectedTwinEventsHash = "8BD205AA1F59AF3CEB19C83DA260EF43366C77ECF30919337B92A398CCB14447"
$expectedOracleEventsHash = "B754CC1CAFF4E2D67604D89D53F8D76AA54681387B4188D7A070B835C2FD74D8"
$expectedMechanicalReportHash = "BDB0B93BF7312B10549456E7646812C24FF6E6B00AB9FBD459145B306A8E1F24"
$expectedMechanicalLogHash = "51792E77FAA1952A4D3EB1815AA76374FCE073DB5EA551BF4C6DABB9DF77AD4C"
$expectedMechanicalGateRecordHash = "15E1126EFD9094A9DE57D75C24C282085F49587EE3DCBABB2EAFA3AB961CB313"
$expectedMechanicalManifestHash = "AB06C8BDE10825EDB362F4E90BA44BED838AB13030F3AE249F8EB4732399B677"

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
    if ($observed -ne $Expected.ToUpperInvariant()) {
        throw "Hash mismatch for $Path : $observed"
    }
}

Assert-Sha256 -Path $sentinelPath -Expected $expectedSentinelHash
Assert-Sha256 -Path $contractPath -Expected $expectedContractHash
Assert-Sha256 -Path $trustPath -Expected $expectedTrustHash
Assert-Sha256 -Path $twinPath -Expected $expectedTwinHash
Assert-Sha256 -Path $envPath -Expected $expectedEnvHash
Assert-Sha256 -Path $twinEventsPath -Expected $expectedTwinEventsHash
Assert-Sha256 -Path $oracleEventsPath -Expected $expectedOracleEventsHash
Assert-Sha256 -Path $mechanicalReportPath -Expected $expectedMechanicalReportHash
Assert-Sha256 -Path $mechanicalLogPath -Expected $expectedMechanicalLogHash
Assert-Sha256 -Path $mechanicalGateRecordPath -Expected $expectedMechanicalGateRecordHash
Assert-Sha256 -Path $mechanicalManifestPath -Expected $expectedMechanicalManifestHash

$contract = Get-Content -LiteralPath $contractPath -Raw |
    ConvertFrom-Json
if (
    $contract.schema -ne "paper1.v5_2.oracle_guard_impact_sentinel_contract.v1" -or
    $contract.authorized_run_id -ne $runId -or
    [int]$contract.authorized_cell_count -ne 1 -or
    -not [bool]$contract.authorization.existing_outputs_only -or
    -not [bool]$contract.authorization.event_level_only -or
    -not [bool]$contract.authorization.performance_outcome_columns_may_be_read -or
    -not [bool]$contract.authorization.performance_outcomes_may_be_inspected -or
    -not [bool]$contract.authorization.single_cell_exploratory_only -or
    [bool]$contract.authorization.simulation_rerun_authorized -or
    [bool]$contract.authorization.step_level_outcome_read_authorized -or
    [bool]$contract.authorization.calibration_authorized -or
    [bool]$contract.authorization.parameter_tuning_authorized -or
    [bool]$contract.authorization.guard_limit_change_authorized -or
    [bool]$contract.authorization.implementation_modification_authorized -or
    [bool]$contract.authorization.confirmatory_inference_authorized -or
    [bool]$contract.authorization.full_campaign_authorized
) {
    throw "The V5.2 sentinel contract is invalid or exceeds scope."
}

$mechanicalReport = Get-Content -LiteralPath $mechanicalReportPath -Raw |
    ConvertFrom-Json
if (
    $mechanicalReport.schema -ne "paper1.v5_2.oracle.mechanical.audit.v1" -or
    $mechanicalReport.run_id -ne $runId -or
    $mechanicalReport.status -ne "pass"
) {
    throw "The source mechanical audit is not a valid pass."
}

$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $MyInvocation.MyCommand.Path,
    [ref]$parseTokens,
    [ref]$parseErrors
)
if (@($parseErrors).Count -ne 0) {
    $parseErrors | ForEach-Object { $_.ToString() } | Out-Host
    throw "The guard-impact sentinel PowerShell gate contains parser errors."
}

$sentinelGateHash = Get-Sha256 -Path $MyInvocation.MyCommand.Path
$trustBefore = Get-Sha256 -Path $trustPath
$twinBefore = Get-Sha256 -Path $twinPath
$envBefore = Get-Sha256 -Path $envPath
$twinEventsBefore = Get-Sha256 -Path $twinEventsPath
$oracleEventsBefore = Get-Sha256 -Path $oracleEventsPath

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sentinelEvidenceRoot = Join-Path $projectRoot "frozen\v52sentinel\S$timestamp"
if (Test-Path -LiteralPath $sentinelEvidenceRoot) {
    throw "Sentinel evidence path already exists: $sentinelEvidenceRoot"
}
New-Item -ItemType Directory -Path $sentinelEvidenceRoot |
    Out-Null

$sentinelLog = Join-Path $sentinelEvidenceRoot "paper1_v5_2_oracle_guard_impact_sentinel.log"
$sentinelReport = Join-Path $sentinelEvidenceRoot "paper1_v5_2_oracle_guard_impact_sentinel.json"
$sentinelRecord = Join-Path $sentinelEvidenceRoot "paper1_v5_2_oracle_guard_impact_sentinel_gate.json"
$sentinelManifest = Join-Path $sentinelEvidenceRoot "PAPER1_V5_2_GUARD_IMPACT_SENTINEL_SHA256.csv"

$relativeReport = $sentinelReport.Substring($projectRoot.Length).TrimStart('\')
$containerReport = "/workspace/" + ($relativeReport -replace '\\', '/')

$sentinelCommand = (
    'docker compose --profile cosim run --rm --no-deps ' +
    '-e PAPER1_V5_2_SENTINEL_REPORT={0} dev ' +
    'python /workspace/analyze_paper1_v5_2_oracle_guard_impact_sentinel.py ' +
    '> "{1}" 2>&1'
) -f $containerReport, $sentinelLog

"STARTING_PAPER1_V5_2_ORACLE_GUARD_IMPACT_SENTINEL"
"RUN_ID=$runId"
"SENTINEL_SHA256=$expectedSentinelHash"
"SENTINEL_GATE_SHA256=$sentinelGateHash"
"CONTRACT_SHA256=$expectedContractHash"
"AUTHORIZED_SCOPE=ONE_FROZEN_CELL_EVENT_LEVEL_EXPLORATORY"
"SINGLE_CELL_EXPLORATORY_ONLY=True"
"EVENT_LEVEL_ONLY=True"
"SIMULATION_RERUN=False"
"STEP_LEVEL_OUTCOME_COLUMNS_READ=False"
"CALIBRATION_AUTHORIZED=False"
"PARAMETER_TUNING_AUTHORIZED=False"
"GUARD_LIMIT_CHANGE_AUTHORIZED=False"
"IMPLEMENTATION_MODIFICATION_AUTHORIZED=False"
"CONFIRMATORY_INFERENCE_AUTHORIZED=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=True"
"PERFORMANCE_OUTCOMES_INSPECTED=True"

& cmd.exe /d /c $sentinelCommand
$sentinelExitCode = $LASTEXITCODE

if ($sentinelExitCode -ne 0) {
    if (Test-Path -LiteralPath $sentinelLog -PathType Leaf) {
        Get-Content -LiteralPath $sentinelLog -Tail 240 |
            Out-Host
    }
    throw "V5.2 oracle guard-impact sentinel failed. Inspect $sentinelLog"
}

$sentinelOutput = Get-Content -LiteralPath $sentinelLog -Raw
if (-not $sentinelOutput.Contains("PAPER1_V5_2_ORACLE_GUARD_IMPACT_SENTINEL_OK")) {
    Get-Content -LiteralPath $sentinelLog -Tail 240 |
        Out-Host
    throw "The guard-impact sentinel success marker is missing."
}
foreach ($forbiddenMarker in @(
    "SIMULATION_RERUN=True",
    "STEP_LEVEL_OUTCOME_COLUMNS_READ=True",
    "CALIBRATION_PERFORMED=True",
    "PARAMETER_TUNING_PERFORMED=True",
    "GUARD_LIMIT_CHANGED=True",
    "IMPLEMENTATION_MODIFIED=True",
    "CONFIRMATORY_INFERENCE_PERFORMED=True",
    "FULL_CAMPAIGN_AUTHORIZED=True"
)) {
    if ($sentinelOutput.Contains($forbiddenMarker)) {
        throw "The sentinel exceeded scope: $forbiddenMarker"
    }
}

if (-not (Test-Path -LiteralPath $sentinelReport -PathType Leaf)) {
    throw "The guard-impact sentinel did not produce its report."
}
$report = Get-Content -LiteralPath $sentinelReport -Raw |
    ConvertFrom-Json
if (
    $report.schema -ne "paper1.v5_2.oracle.guard_impact_sentinel.v1" -or
    $report.run_id -ne $runId -or
    $report.status -ne "complete_pending_review" -or
    -not [bool]$report.scope.single_cell_exploratory_only -or
    -not [bool]$report.scope.event_level_only -or
    [bool]$report.scope.simulation_rerun -or
    [bool]$report.scope.step_level_outcome_columns_read -or
    [bool]$report.scope.calibration_performed -or
    [bool]$report.scope.parameter_tuning_performed -or
    [bool]$report.scope.guard_limit_changed -or
    [bool]$report.scope.implementation_modified -or
    [bool]$report.scope.confirmatory_inference_performed -or
    [bool]$report.scope.full_campaign_authorized -or
    -not [bool]$report.scope.performance_outcome_columns_read -or
    -not [bool]$report.scope.performance_outcomes_inspected
) {
    throw "The sentinel report is invalid or exceeds scope."
}

if ((Get-Sha256 -Path $trustPath) -ne $trustBefore) {
    throw "The sentinel changed trust_metric.py."
}
if ((Get-Sha256 -Path $twinPath) -ne $twinBefore) {
    throw "The sentinel changed twin_fed.py."
}
if ((Get-Sha256 -Path $envPath) -ne $envBefore) {
    throw "The sentinel changed .env."
}
if ((Get-Sha256 -Path $twinEventsPath) -ne $twinEventsBefore) {
    throw "The sentinel changed the frozen twin event output."
}
if ((Get-Sha256 -Path $oracleEventsPath) -ne $oracleEventsBefore) {
    throw "The sentinel changed the frozen oracle event output."
}

Copy-Item -LiteralPath $sentinelPath -Destination $sentinelEvidenceRoot
Copy-Item -LiteralPath $contractPath -Destination $sentinelEvidenceRoot
Copy-Item -LiteralPath $mechanicalReportPath -Destination $sentinelEvidenceRoot
Copy-Item -LiteralPath $mechanicalGateRecordPath -Destination $sentinelEvidenceRoot
Copy-Item -LiteralPath $mechanicalManifestPath -Destination $sentinelEvidenceRoot
Copy-Item -LiteralPath $MyInvocation.MyCommand.Path -Destination $sentinelEvidenceRoot

$record = [ordered]@{
    schema = "paper1.v5_2.oracle.guard_impact_sentinel_gate.v1"
    run_id = $runId
    status = "complete_pending_review"
    sentinel_sha256 = $expectedSentinelHash.ToLowerInvariant()
    sentinel_gate_sha256 = $sentinelGateHash.ToLowerInvariant()
    contract_sha256 = $expectedContractHash.ToLowerInvariant()
    report_sha256 = (Get-Sha256 -Path $sentinelReport).ToLowerInvariant()
    log_sha256 = (Get-Sha256 -Path $sentinelLog).ToLowerInvariant()
    source_mechanical_report_sha256 = $expectedMechanicalReportHash.ToLowerInvariant()
    source_mechanical_gate_record_sha256 = $expectedMechanicalGateRecordHash.ToLowerInvariant()
    twin_events_sha256 = $expectedTwinEventsHash.ToLowerInvariant()
    oracle_events_sha256 = $expectedOracleEventsHash.ToLowerInvariant()
    single_cell_exploratory_only = $true
    event_level_only = $true
    simulation_rerun = $false
    step_level_outcome_columns_read = $false
    calibration_performed = $false
    parameter_tuning_performed = $false
    guard_limit_changed = $false
    implementation_modified = $false
    confirmatory_inference_performed = $false
    full_campaign_authorized = $false
    performance_outcome_columns_read = $true
    performance_outcomes_inspected = $true
}
$record |
    ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $sentinelRecord -Encoding UTF8

$manifestRows = @(
    Get-ChildItem -LiteralPath $sentinelEvidenceRoot -File |
        Where-Object { $_.FullName -ne $sentinelManifest } |
        Sort-Object Name |
        ForEach-Object {
            [PSCustomObject]@{
                Name = $_.Name
                Length = $_.Length
                SHA256 = Get-Sha256 -Path $_.FullName
            }
        }
)
$manifestRows |
    Export-Csv -LiteralPath $sentinelManifest -NoTypeInformation

Get-Content -LiteralPath $sentinelLog

"SENTINEL_REPORT_SHA256=$(Get-Sha256 -Path $sentinelReport)"
"SENTINEL_LOG_SHA256=$(Get-Sha256 -Path $sentinelLog)"
"SENTINEL_GATE_SHA256=$sentinelGateHash"
"SENTINEL_GATE_RECORD_SHA256=$(Get-Sha256 -Path $sentinelRecord)"
"SENTINEL_EVIDENCE_MANIFEST_SHA256=$(Get-Sha256 -Path $sentinelManifest)"
"SENTINEL_EVIDENCE_ROOT=$sentinelEvidenceRoot"
"SINGLE_CELL_EXPLORATORY_ONLY=True"
"EVENT_LEVEL_ONLY=True"
"SIMULATION_RERUN=False"
"STEP_LEVEL_OUTCOME_COLUMNS_READ=False"
"CALIBRATION_PERFORMED=False"
"PARAMETER_TUNING_PERFORMED=False"
"GUARD_LIMIT_CHANGED=False"
"IMPLEMENTATION_MODIFIED=False"
"CONFIRMATORY_INFERENCE_PERFORMED=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=True"
"PERFORMANCE_OUTCOMES_INSPECTED=True"
"PAPER1_V5_2_ORACLE_GUARD_IMPACT_SENTINEL_COMPLETE"
