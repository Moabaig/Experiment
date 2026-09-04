$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$builder = Join-Path $projectRoot "build_paper1_v5_2_mechanical_runner.py"
$sourceRunner = Join-Path $projectRoot "run_paper1_factor_campaign_v5_1_mechanical.ps1"
$candidateRunner = Join-Path $projectRoot "run_paper1_factor_campaign_v5_2_mechanical.ps1"
$workspace = Join-Path $projectRoot "paper1_v5_2_repair"
$contractPath = Join-Path $workspace "paper1_v5_2_mechanical_validation_contract.json"
$buildReportPath = Join-Path $workspace "v5_2_mechanical_runner_build.json"

$expectedBuilderHash = "A3FB927879DF924F9EC518F57BDE4B1EE5CD5BF8C59D8E3D29FC9C0819312EEB"
$expectedSourceRunnerHash = "31CC460690323100FDCC10DF7162DB0A92D615035DBC6E9319C09D1610B1DAE9"
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
Assert-Sha256 -Path (Join-Path $projectRoot "trust_metric.py") -Expected $expectedTrustHash
Assert-Sha256 -Path (Join-Path $projectRoot "twin_fed.py") -Expected $expectedTwinHash
Assert-Sha256 -Path (Join-Path $projectRoot ".env") -Expected $expectedEnvHash

$trustHashBefore = Get-Sha256 -Path (Join-Path $projectRoot "trust_metric.py")
$twinHashBefore = Get-Sha256 -Path (Join-Path $projectRoot "twin_fed.py")
$envHashBefore = Get-Sha256 -Path (Join-Path $projectRoot ".env")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$buildLog = Join-Path $workspace "v5_2_runner_rebuild_$timestamp.log"
$preflightLog = Join-Path $workspace "v5_2_runner_preflight_retry_$timestamp.log"

New-Item -ItemType Directory -Path $workspace -Force | Out-Null

"STARTING_PAPER1_V5_2_MECHANICAL_RUNNER_REBUILD"
"BUILDER_SHA256=$expectedBuilderHash"
"SOURCE_RUNNER_SHA256=$expectedSourceRunnerHash"
"SIMULATION_STARTED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"

$buildCommand = (
    'docker compose --profile cosim run --rm --no-deps dev ' +
    'python /workspace/build_paper1_v5_2_mechanical_runner.py ' +
    '> "{0}" 2>&1'
) -f $buildLog

& cmd.exe /d /c $buildCommand
$buildExitCode = $LASTEXITCODE

if (Test-Path -LiteralPath $buildLog -PathType Leaf) {
    Get-Content -LiteralPath $buildLog | Out-Host
}

if ($buildExitCode -ne 0) {
    throw "Corrected V5.2 runner build failed. Inspect $buildLog"
}

$buildOutput = Get-Content -LiteralPath $buildLog -Raw
if (-not $buildOutput.Contains("PAPER1_V5_2_MECHANICAL_RUNNER_BUILD_OK")) {
    throw "Corrected V5.2 runner-build success marker is missing."
}

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

$runnerHash = Get-Sha256 -Path $candidateRunner
$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
$buildReport = Get-Content -LiteralPath $buildReportPath -Raw | ConvertFrom-Json

if ($contract.runner.sha256.ToUpperInvariant() -ne $runnerHash) {
    throw "Contract runner hash does not match the generated runner."
}
if ($buildReport.candidate_runner_sha256.ToUpperInvariant() -ne $runnerHash) {
    throw "Build-report runner hash does not match the generated runner."
}
if ($contract.authorized_cells.Count -ne 1) {
    throw "The V5.2 contract must authorize exactly one mechanical cell."
}
if ($contract.authorized_cells[0].run_id -ne "paper1_v5_2mv_s002_bw04_oracle") {
    throw "Unexpected authorized V5.2 run id."
}

"CORRECTED_CAMPAIGN_ROOT_VERIFIED=True"
"RUNNER_CANDIDATE_SHA256=$runnerHash"
"STARTING_VALIDATE_ONLY_PREFLIGHT"
"SIMULATION_STARTED=False"

$preflightCommand = (
    'powershell.exe -NoProfile -ExecutionPolicy Bypass ' +
    '-File "{0}" -Mode Validate ' +
    '-SeedFrom 2 -SeedTo 2 -BandwidthFrom 4 -BandwidthTo 4 ' +
    '> "{1}" 2>&1'
) -f $candidateRunner, $preflightLog

& cmd.exe /d /c $preflightCommand
$preflightExitCode = $LASTEXITCODE

if (Test-Path -LiteralPath $preflightLog -PathType Leaf) {
    Get-Content -LiteralPath $preflightLog | Out-Host
}

if ($preflightExitCode -ne 0) {
    throw "Corrected V5.2 validate-only preflight failed. Inspect $preflightLog"
}

$preflightOutput = Get-Content -LiteralPath $preflightLog -Raw
if (-not $preflightOutput.Contains("PAPER1_V5_2_MECHANICAL_BATCH_COMPLETE")) {
    throw "V5.2 validate-only completion marker is missing."
}
if (-not $preflightOutput.Contains("MODE=Validate")) {
    throw "V5.2 runner did not report Validate mode."
}
if (
    $preflightOutput.Contains("FACTOR_CELL_COMPLETE") -or
    $preflightOutput.Contains("CO_SIMULATION_COMPLETE")
) {
    throw "Validate-only gate unexpectedly emitted a simulation completion marker."
}

Assert-Sha256 -Path (Join-Path $projectRoot "trust_metric.py") -Expected $trustHashBefore
Assert-Sha256 -Path (Join-Path $projectRoot "twin_fed.py") -Expected $twinHashBefore
Assert-Sha256 -Path (Join-Path $projectRoot ".env") -Expected $envHashBefore

$evidenceRoot = Join-Path $projectRoot "frozen\v52runner\R$timestamp"
New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null

foreach ($sourcePath in @(
    $builder,
    $candidateRunner,
    $contractPath,
    $buildReportPath,
    $buildLog,
    $preflightLog
)) {
    Copy-Item -LiteralPath $sourcePath -Destination $evidenceRoot -Force
}

$recordPath = Join-Path $evidenceRoot "v5_2_runner_gate_record.json"
$record = [ordered]@{
    schema = "paper1.v5_2.mechanical_runner_gate_record.v1"
    builder_sha256 = $expectedBuilderHash.ToLowerInvariant()
    source_runner_sha256 = $expectedSourceRunnerHash.ToLowerInvariant()
    candidate_runner_sha256 = $runnerHash.ToLowerInvariant()
    campaign_root = "frozen/paper1_factor_campaign_v5_2_mechanical"
    mode = "Validate"
    authorized_run_id = "paper1_v5_2mv_s002_bw04_oracle"
    live_files_modified = $false
    simulation_started = $false
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
                SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
            }
        }
)
$manifestRows |
    Export-Csv -LiteralPath $manifestPath -NoTypeInformation

"PAPER1_V5_2_MECHANICAL_RUNNER_PREFLIGHT_OK"
"BUILDER_SHA256=$expectedBuilderHash"
"SOURCE_RUNNER_SHA256=$expectedSourceRunnerHash"
"RUNNER_CANDIDATE_SHA256=$runnerHash"
"CONTRACT_SHA256=$((Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash)"
"BUILD_REPORT_SHA256=$((Get-FileHash -LiteralPath $buildReportPath -Algorithm SHA256).Hash)"
"GATE_RECORD_SHA256=$((Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash)"
"EVIDENCE_MANIFEST_SHA256=$((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash)"
"EVIDENCE_ROOT=$evidenceRoot"
"CAMPAIGN_ROOT=frozen\paper1_factor_campaign_v5_2_mechanical"
"ENV_UNCHANGED=True"
"LIVE_FILES_MODIFIED=False"
"SIMULATION_STARTED=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_MECHANICAL_RUNNER_GATE_COMPLETE"
