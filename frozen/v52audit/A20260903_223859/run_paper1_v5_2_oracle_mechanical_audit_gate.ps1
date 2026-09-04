$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$runId = "paper1_v5_2mv_s002_bw04_oracle"
$auditPath = Join-Path $projectRoot "audit_paper1_v5_2_oracle_mechanical.py"
$runnerPath = Join-Path $projectRoot "run_paper1_factor_campaign_v5_2_mechanical.ps1"
$contractPath = Join-Path $projectRoot "paper1_v5_2_repair\paper1_v5_2_mechanical_validation_contract.json"
$trustPath = Join-Path $projectRoot "trust_metric.py"
$twinPath = Join-Path $projectRoot "twin_fed.py"
$feederPath = Join-Path $projectRoot "feeder.npz"
$envPath = Join-Path $projectRoot ".env"
$runRoot = Join-Path $projectRoot "runs\$runId"
$cellEvidenceRoot = Join-Path $projectRoot "frozen\v52cell\C20260903_174122"
$gateRecordPath = Join-Path $cellEvidenceRoot "paper1_v5_2_oracle_mechanical_cell_record.json"
$gateManifestPath = Join-Path $cellEvidenceRoot "PAPER1_V5_2_ORACLE_CELL_GATE_SHA256.csv"
$cellManifestPath = Join-Path $runRoot "CELL_OUTPUT_SHA256SUMS.csv"

$expectedAuditHash = "D81AD40E8454CCB53FA9EC643F483CA2538764259EE8C03DD3B54814D62813AA"
$expectedRunnerHash = "5753DE4E708206D0A1F8669ADC52FA74BA5E4395318A24CCFBFAD6C1FCEB6629"
$expectedContractHash = "11B20715BD970988A25429BA645373382671C8D5713CBA5A836705F83B09256C"
$expectedTrustHash = "936DD373A2D8A2F0B905604CA4C3DE61EC2CC889BA233AA150A24F44F2926FE6"
$expectedTwinHash = "9CD9FFAA32DCFE2F12ED161A8D62D2D97B2AB0B4D462FDA0E97E7F46572043A4"
$expectedFeederHash = "9DF3426EA48C55F509E1F5F149E72F4E076DE7D9099980D97E721EFB94C8BD5D"
$expectedEnvHash = "55A4FCB1ACB19D86CBE2DA4BCC4FE814170A14A5A637EC6CEC97D9C94195D694"
$expectedGateRecordHash = "E48BE42D0D1A8B735BC02718D3A485EB8D4C1F962116FEF14533E0C64B22BFCD"
$expectedGateManifestHash = "CAE5B6A07227AD1AD28B6AEE8DA96F8FACA65ADF01DBF998DB599FBA480E36BE"

$expectedOutputHashes = [ordered]@{
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
    if ($observed -ne $Expected.ToUpperInvariant()) {
        throw "Hash mismatch for $Path : $observed"
    }
}

Assert-Sha256 -Path $auditPath -Expected $expectedAuditHash
Assert-Sha256 -Path $runnerPath -Expected $expectedRunnerHash
Assert-Sha256 -Path $contractPath -Expected $expectedContractHash
Assert-Sha256 -Path $trustPath -Expected $expectedTrustHash
Assert-Sha256 -Path $twinPath -Expected $expectedTwinHash
Assert-Sha256 -Path $feederPath -Expected $expectedFeederHash
Assert-Sha256 -Path $envPath -Expected $expectedEnvHash
Assert-Sha256 -Path $gateRecordPath -Expected $expectedGateRecordHash
Assert-Sha256 -Path $gateManifestPath -Expected $expectedGateManifestHash

if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) {
    throw "Completed V5.2 run directory is missing: $runRoot"
}

foreach ($entry in $expectedOutputHashes.GetEnumerator()) {
    Assert-Sha256 `
        -Path (Join-Path $runRoot $entry.Key) `
        -Expected $entry.Value
}

$gateRecord = Get-Content -LiteralPath $gateRecordPath -Raw |
    ConvertFrom-Json
if (
    $gateRecord.schema -ne "paper1.v5_2.oracle_mechanical_cell_gate.v1" -or
    $gateRecord.run_id -ne $runId -or
    -not [bool]$gateRecord.simulation_completed -or
    [bool]$gateRecord.simulation_rerun -or
    [bool]$gateRecord.implementation_modified -or
    -not [bool]$gateRecord.env_restored_exactly -or
    [bool]$gateRecord.performance_outcome_columns_read -or
    [bool]$gateRecord.performance_outcomes_inspected
) {
    throw "The finalized V5.2 cell gate record does not authorize this audit."
}

$auditGateHash = Get-Sha256 -Path $MyInvocation.MyCommand.Path

$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $MyInvocation.MyCommand.Path,
    [ref]$parseTokens,
    [ref]$parseErrors
)
if (@($parseErrors).Count -ne 0) {
    $parseErrors | ForEach-Object { $_.ToString() } | Out-Host
    throw "The mechanical-audit PowerShell gate contains parser errors."
}

$trustBefore = Get-Sha256 -Path $trustPath
$twinBefore = Get-Sha256 -Path $twinPath
$envBefore = Get-Sha256 -Path $envPath
$outputBefore = [ordered]@{}
foreach ($entry in $expectedOutputHashes.GetEnumerator()) {
    $outputBefore[$entry.Key] = Get-Sha256 -Path (
        Join-Path $runRoot $entry.Key
    )
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$auditEvidenceRoot = Join-Path $projectRoot "frozen\v52audit\A$timestamp"
if (Test-Path -LiteralPath $auditEvidenceRoot) {
    throw "Mechanical-audit evidence path already exists: $auditEvidenceRoot"
}
New-Item -ItemType Directory -Path $auditEvidenceRoot |
    Out-Null

$auditLog = Join-Path $auditEvidenceRoot "paper1_v5_2_oracle_mechanical_audit.log"
$auditReport = Join-Path $auditEvidenceRoot "paper1_v5_2_oracle_mechanical_audit.json"
$auditRecord = Join-Path $auditEvidenceRoot "paper1_v5_2_oracle_mechanical_audit_gate.json"
$auditManifest = Join-Path $auditEvidenceRoot "PAPER1_V5_2_MECHANICAL_AUDIT_SHA256.csv"

$relativeReport = $auditReport.Substring($projectRoot.Length).TrimStart('\')
$containerReport = "/workspace/" + ($relativeReport -replace '\\', '/')

$auditCommand = (
    'docker compose --profile cosim run --rm --no-deps ' +
    '-e PAPER1_V5_2_AUDIT_REPORT={0} dev ' +
    'python /workspace/audit_paper1_v5_2_oracle_mechanical.py ' +
    '> "{1}" 2>&1'
) -f $containerReport, $auditLog

"STARTING_PAPER1_V5_2_ORACLE_MECHANICAL_AUDIT"
"RUN_ID=$runId"
"AUDITOR_SHA256=$expectedAuditHash"
"AUDIT_GATE_SHA256=$auditGateHash"
"AUTHORIZED_SCOPE=MECHANICAL_AND_PROVENANCE_COLUMNS_ONLY"
"SIMULATION_RERUN=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"

& cmd.exe /d /c $auditCommand
$auditExitCode = $LASTEXITCODE

if ($auditExitCode -ne 0) {
    if (Test-Path -LiteralPath $auditLog -PathType Leaf) {
        Get-Content -LiteralPath $auditLog -Tail 240 |
            Out-Host
    }
    throw "V5.2 oracle mechanical audit failed. Inspect $auditLog"
}

$auditOutput = Get-Content -LiteralPath $auditLog -Raw
if (-not $auditOutput.Contains("PAPER1_V5_2_ORACLE_MECHANICAL_AUDIT_OK")) {
    Get-Content -LiteralPath $auditLog -Tail 240 |
        Out-Host
    throw "The V5.2 mechanical-audit success marker is missing."
}
if (
    $auditOutput.Contains("PERFORMANCE_OUTCOME_COLUMNS_READ=True") -or
    $auditOutput.Contains("PERFORMANCE_OUTCOMES_INSPECTED=True") -or
    $auditOutput.Contains("FULL_CAMPAIGN_AUTHORIZED=True") -or
    $auditOutput.Contains("CALIBRATION_AUTHORIZED=True")
) {
    throw "The mechanical audit exceeded its authorization boundary."
}

if (-not (Test-Path -LiteralPath $auditReport -PathType Leaf)) {
    throw "The mechanical audit did not produce its report."
}
$report = Get-Content -LiteralPath $auditReport -Raw |
    ConvertFrom-Json
if (
    $report.schema -ne "paper1.v5_2.oracle.mechanical.audit.v1" -or
    $report.run_id -ne $runId -or
    $report.status -ne "pass" -or
    [bool]$report.simulation_rerun -or
    [bool]$report.implementation_modified -or
    [bool]$report.full_campaign_authorized -or
    [bool]$report.calibration_authorized -or
    [bool]$report.performance_outcome_columns_read -or
    [bool]$report.performance_outcomes_inspected
) {
    throw "The V5.2 mechanical-audit report is invalid or exceeds scope."
}

if ((Get-Sha256 -Path $trustPath) -ne $trustBefore) {
    throw "The mechanical audit changed trust_metric.py."
}
if ((Get-Sha256 -Path $twinPath) -ne $twinBefore) {
    throw "The mechanical audit changed twin_fed.py."
}
if ((Get-Sha256 -Path $envPath) -ne $envBefore) {
    throw "The mechanical audit changed .env."
}
foreach ($entry in $outputBefore.GetEnumerator()) {
    if ((Get-Sha256 -Path (Join-Path $runRoot $entry.Key)) -ne $entry.Value) {
        throw "The mechanical audit changed a frozen run output: $($entry.Key)"
    }
}

Copy-Item -LiteralPath $auditPath -Destination $auditEvidenceRoot
Copy-Item -LiteralPath $contractPath -Destination $auditEvidenceRoot
Copy-Item -LiteralPath $gateRecordPath -Destination $auditEvidenceRoot
Copy-Item -LiteralPath $gateManifestPath -Destination $auditEvidenceRoot
Copy-Item -LiteralPath $cellManifestPath -Destination $auditEvidenceRoot
Copy-Item -LiteralPath $MyInvocation.MyCommand.Path -Destination $auditEvidenceRoot

$record = [ordered]@{
    schema = "paper1.v5_2.oracle.mechanical.audit_gate.v1"
    run_id = $runId
    status = "pass"
    auditor_sha256 = $expectedAuditHash.ToLowerInvariant()
    audit_gate_sha256 = $auditGateHash.ToLowerInvariant()
    report_sha256 = (Get-Sha256 -Path $auditReport).ToLowerInvariant()
    audit_log_sha256 = (Get-Sha256 -Path $auditLog).ToLowerInvariant()
    runner_sha256 = $expectedRunnerHash.ToLowerInvariant()
    contract_sha256 = $expectedContractHash.ToLowerInvariant()
    installed_trust_sha256 = $expectedTrustHash.ToLowerInvariant()
    installed_twin_sha256 = $expectedTwinHash.ToLowerInvariant()
    feeder_sha256 = $expectedFeederHash.ToLowerInvariant()
    cell_gate_record_sha256 = $expectedGateRecordHash.ToLowerInvariant()
    cell_gate_manifest_sha256 = $expectedGateManifestHash.ToLowerInvariant()
    mechanical_gates_passed = 13
    simulation_rerun = $false
    implementation_modified = $false
    full_campaign_authorized = $false
    calibration_authorized = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}
$record |
    ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $auditRecord -Encoding UTF8

$manifestRows = @(
    Get-ChildItem -LiteralPath $auditEvidenceRoot -File |
        Where-Object { $_.FullName -ne $auditManifest } |
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
    Export-Csv -LiteralPath $auditManifest -NoTypeInformation

Get-Content -LiteralPath $auditLog

"MECHANICAL_AUDIT_REPORT_SHA256=$(Get-Sha256 -Path $auditReport)"
"MECHANICAL_AUDIT_LOG_SHA256=$(Get-Sha256 -Path $auditLog)"
"MECHANICAL_AUDIT_GATE_SHA256=$auditGateHash"
"MECHANICAL_AUDIT_GATE_RECORD_SHA256=$(Get-Sha256 -Path $auditRecord)"
"MECHANICAL_AUDIT_EVIDENCE_MANIFEST_SHA256=$(Get-Sha256 -Path $auditManifest)"
"MECHANICAL_AUDIT_EVIDENCE_ROOT=$auditEvidenceRoot"
"MECHANICAL_GATES_PASSED=13"
"SIMULATION_RERUN=False"
"IMPLEMENTATION_MODIFIED=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_ORACLE_MECHANICAL_AUDIT_COMPLETE"
