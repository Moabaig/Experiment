$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$workspace = Join-Path $projectRoot "paper1_v5_2_repair"
$builder = Join-Path $projectRoot "build_paper1_v5_2_mechanical_runner.py"
$sourceRunner = Join-Path $projectRoot "run_paper1_factor_campaign_v5_1_mechanical.ps1"
$candidateRunner = Join-Path $projectRoot "run_paper1_factor_campaign_v5_2_mechanical.ps1"
$contractPath = Join-Path $workspace "paper1_v5_2_mechanical_validation_contract.json"
$buildReportPath = Join-Path $workspace "v5_2_mechanical_runner_build.json"

$expectedBuilderHash = "A3FB927879DF924F9EC518F57BDE4B1EE5CD5BF8C59D8E3D29FC9C0819312EEB"
$expectedSourceRunnerHash = "31CC460690323100FDCC10DF7162DB0A92D615035DBC6E9319C09D1610B1DAE9"
$expectedCandidateRunnerHash = "5753DE4E708206D0A1F8669ADC52FA74BA5E4395318A24CCFBFAD6C1FCEB6629"
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

Assert-Sha256 -Path $builder -Expected $expectedBuilderHash
Assert-Sha256 -Path $sourceRunner -Expected $expectedSourceRunnerHash
Assert-Sha256 -Path $candidateRunner -Expected $expectedCandidateRunnerHash
Assert-Sha256 -Path $contractPath -Expected $expectedContractHash
Assert-Sha256 -Path $buildReportPath -Expected $expectedBuildReportHash
Assert-Sha256 -Path (Join-Path $projectRoot "trust_metric.py") -Expected $expectedTrustHash
Assert-Sha256 -Path (Join-Path $projectRoot "twin_fed.py") -Expected $expectedTwinHash
Assert-Sha256 -Path (Join-Path $projectRoot ".env") -Expected $expectedEnvHash

$candidateText = Get-Content -LiteralPath $candidateRunner -Raw
if (-not $candidateText.Contains("paper1_factor_campaign_v5_2_mechanical")) {
    throw "V5.2 campaign-root token is missing from the generated runner."
}
if ($candidateText.Contains("paper1_factor_campaign_v5_1_mechanical")) {
    throw "Generated runner still targets the frozen V5.1 campaign root."
}

$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $candidateRunner,
    [ref]$parseTokens,
    [ref]$parseErrors
)
if (@($parseErrors).Count -ne 0) {
    $parseErrors | ForEach-Object { $_.ToString() } | Out-Host
    throw "Generated V5.2 runner has PowerShell parse errors."
}

$successfulPreflightLog = $null
$preflightOutput = $null
$preflightLogs = @(
    Get-ChildItem `
        -LiteralPath $workspace `
        -Filter "v5_2_runner_preflight_retry_*.log" `
        -File |
        Sort-Object LastWriteTime -Descending
)

foreach ($candidateLog in $preflightLogs) {
    $candidateOutput = Get-Content -LiteralPath $candidateLog.FullName -Raw
    if ($candidateOutput.Contains("PAPER1_V5_2_MECHANICAL_PREFLIGHT_OK")) {
        $successfulPreflightLog = $candidateLog
        $preflightOutput = $candidateOutput
        break
    }
}

if ($null -eq $successfulPreflightLog) {
    throw "No successful V5.2 validate-only preflight log was found."
}

if (
    $preflightOutput.Contains("FACTOR_CELL_COMPLETE") -or
    $preflightOutput.Contains("CO_SIMULATION_COMPLETE") -or
    $preflightOutput.Contains("MODE=Run")
) {
    throw "Selected preflight log contains a simulation-run marker."
}

$logNameMatch = [regex]::Match(
    $successfulPreflightLog.BaseName,
    '^v5_2_runner_preflight_retry_(\d{8}_\d{6})$'
)
if (-not $logNameMatch.Success) {
    throw "Cannot recover the gate timestamp from the preflight log name."
}
$runTimestamp = $logNameMatch.Groups[1].Value
$buildLog = Join-Path $workspace "v5_2_runner_rebuild_$runTimestamp.log"

if (-not (Test-Path -LiteralPath $buildLog -PathType Leaf)) {
    throw "Matching runner-build log is missing: $buildLog"
}
$buildOutput = Get-Content -LiteralPath $buildLog -Raw
if (-not $buildOutput.Contains("PAPER1_V5_2_MECHANICAL_RUNNER_BUILD_OK")) {
    throw "Matching runner-build success marker is missing."
}

$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
$buildReport = Get-Content -LiteralPath $buildReportPath -Raw | ConvertFrom-Json

if ($contract.runner.sha256.ToUpperInvariant() -ne $expectedCandidateRunnerHash) {
    throw "Contract does not identify the validated V5.2 runner."
}
if ($buildReport.candidate_runner_sha256.ToUpperInvariant() -ne $expectedCandidateRunnerHash) {
    throw "Build report does not identify the validated V5.2 runner."
}
if ($contract.authorized_cells.Count -ne 1) {
    throw "The V5.2 contract must authorize exactly one mechanical cell."
}
if ($contract.authorized_cells[0].run_id -ne "paper1_v5_2mv_s002_bw04_oracle") {
    throw "Unexpected authorized V5.2 run id."
}
if (
    [bool]$buildReport.simulation_started -or
    [bool]$buildReport.full_campaign_authorized -or
    [bool]$buildReport.calibration_authorized -or
    [bool]$buildReport.performance_outcome_columns_read -or
    [bool]$buildReport.performance_outcomes_inspected
) {
    throw "Build report exceeds the mechanical-validation authorization boundary."
}

$finalizedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$evidenceRoot = Join-Path $projectRoot "frozen\v52runner\G$finalizedAt"
if (Test-Path -LiteralPath $evidenceRoot) {
    throw "Evidence destination already exists: $evidenceRoot"
}
New-Item -ItemType Directory -Path $evidenceRoot | Out-Null

foreach ($sourcePath in @(
    $builder,
    $candidateRunner,
    $contractPath,
    $buildReportPath,
    $buildLog,
    $successfulPreflightLog.FullName
)) {
    Copy-Item -LiteralPath $sourcePath -Destination $evidenceRoot
}

$recordPath = Join-Path $evidenceRoot "v5_2_runner_gate_record.json"
$record = [ordered]@{
    schema = "paper1.v5_2.mechanical_runner_gate_record.v1"
    source_log_timestamp = $runTimestamp
    finalized_at = $finalizedAt
    builder_sha256 = $expectedBuilderHash.ToLowerInvariant()
    source_runner_sha256 = $expectedSourceRunnerHash.ToLowerInvariant()
    candidate_runner_sha256 = $expectedCandidateRunnerHash.ToLowerInvariant()
    contract_sha256 = $expectedContractHash.ToLowerInvariant()
    build_report_sha256 = $expectedBuildReportHash.ToLowerInvariant()
    preflight_log_sha256 = (
        Get-Sha256 -Path $successfulPreflightLog.FullName
    ).ToLowerInvariant()
    campaign_root = "frozen/paper1_factor_campaign_v5_2_mechanical"
    successful_marker = "PAPER1_V5_2_MECHANICAL_PREFLIGHT_OK"
    validation_log_reused = $true
    live_files_modified = $false
    simulation_rerun = $false
    full_campaign_authorized = $false
    calibration_authorized = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}
$record |
    ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $recordPath -Encoding UTF8

$manifestPath = Join-Path $evidenceRoot "PAPER1_V5_2_RUNNER_GATE_SHA256.csv"
$manifestRows = @(
    Get-ChildItem -LiteralPath $evidenceRoot -File |
        Where-Object { $_.FullName -ne $manifestPath } |
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
$manifestRows |
    Export-Csv -LiteralPath $manifestPath -NoTypeInformation

"PAPER1_V5_2_MECHANICAL_RUNNER_PREFLIGHT_OK"
"BUILDER_SHA256=$expectedBuilderHash"
"SOURCE_RUNNER_SHA256=$expectedSourceRunnerHash"
"RUNNER_CANDIDATE_SHA256=$expectedCandidateRunnerHash"
"CONTRACT_SHA256=$expectedContractHash"
"BUILD_REPORT_SHA256=$expectedBuildReportHash"
"PREFLIGHT_LOG_SHA256=$((Get-FileHash -LiteralPath $successfulPreflightLog.FullName -Algorithm SHA256).Hash)"
"GATE_RECORD_SHA256=$((Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash)"
"EVIDENCE_MANIFEST_SHA256=$((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash)"
"EVIDENCE_ROOT=$evidenceRoot"
"CAMPAIGN_ROOT=frozen\paper1_factor_campaign_v5_2_mechanical"
"VALIDATION_LOG_REUSED=True"
"ENV_UNCHANGED=True"
"LIVE_FILES_MODIFIED=False"
"SIMULATION_RERUN=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_MECHANICAL_RUNNER_GATE_COMPLETE"
