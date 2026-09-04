[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Read-DotEnv {
    param([string]$Path)
    $result = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $parts = $trimmed -split '=', 2
        if ($parts.Count -eq 2) { $result[$parts[0].Trim()] = $parts[1].Trim() }
    }
    return $result
}

if (-not (Test-Path -LiteralPath '.\.env')) {
    throw 'Missing .env. Copy .env.example to .env and set the predeclared values.'
}

$cfg = Read-DotEnv '.\.env'
$requiredData = @('feeder.npz', 'patterns.npz', 'patterns.csv', 'scenarios.csv')
foreach ($file in $requiredData) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing required input: $file" }
}

$truthFile = if ($cfg.ContainsKey('TRUTH_FILE')) { $cfg['TRUTH_FILE'] } else { 'truth.npz' }
if (-not (Test-Path -LiteralPath $truthFile)) { throw "Missing truth file: $truthFile" }

$gamma = if ($cfg.ContainsKey('DRIFT_GAMMA')) { $cfg['DRIFT_GAMMA'] } else { '' }
$gammaValue = 0.0
if (-not [double]::TryParse($gamma, [ref]$gammaValue) -or $gammaValue -le 0.0) {
    throw 'DRIFT_GAMMA must be a predeclared positive number.'
}

$calibrationMode = if ($cfg.ContainsKey('CALIBRATION_MODE')) { $cfg['CALIBRATION_MODE'] } else { '0' }
if ($calibrationMode -ne '1') {
    $calibrationFile = if ($cfg.ContainsKey('CALIBRATION_FILE')) { $cfg['CALIBRATION_FILE'] } else { 'calibration.json' }
    if (-not (Test-Path -LiteralPath $calibrationFile)) {
        throw "Missing frozen calibration file: $calibrationFile"
    }
    $calibration = Get-Content -LiteralPath $calibrationFile -Raw | ConvertFrom-Json
    if ($calibration.schema -ne 'twin.calibration.v2') {
        throw "Unsupported calibration schema '$($calibration.schema)'. Regenerate it with the repaired calibrate_twin.py using oracle_scores.parquet."
    }
}

$runId = if ($cfg.ContainsKey('RUN_ID')) { $cfg['RUN_ID'] } else { 'real_001' }
$logPath = ".\run_$runId.log"
$powerSeed = if ($cfg.ContainsKey('POWER_SEED')) { $cfg['POWER_SEED'] } else { '1001' }
$stopTime = if ($cfg.ContainsKey('STOP_TIME')) { $cfg['STOP_TIME'] } else { '0' }
$allowLinearizedTelemetry = (
    $cfg.ContainsKey('ALLOW_LINEARIZED_TELEMETRY') -and
    $cfg['ALLOW_LINEARIZED_TELEMETRY'] -eq '1'
)

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'docker compose config failed' }

$truthValidationArgs = @(
    '/workspace/power_fed.py',
    '--feeder=/workspace/feeder.npz',
    "--truth=/workspace/$truthFile",
    "--seed=$powerSeed",
    '--dt=1',
    '--steps-per-event=12',
    "--stop-time=$stopTime",
    '--validate-only'
)
if ($allowLinearizedTelemetry) {
    $truthValidationArgs += '--allow-linearized-telemetry'
    Write-Warning 'LINEARIZED TELEMETRY ENABLED: this run is smoke/integration evidence only.'
}

docker compose --profile cosim run --rm --no-deps dev python @truthValidationArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Production truth validation failed. Physical truth must include z_true.'
}

Write-Host "Starting four-federate run: $runId"
cmd /d /c "docker compose --profile cosim up --abort-on-container-failure --remove-orphans broker power-fed net-fed twin-fed oracle-fed 2>&1" |
    Tee-Object -FilePath $logPath
$composeExit = $LASTEXITCODE

docker compose --profile cosim ps -a
if ($composeExit -ne 0) {
    throw "Co-simulation failed with exit code $composeExit. See $logPath"
}

Write-Host "CO_SIMULATION_COMPLETE run=$runId log=$logPath"
Write-Host "Inspect runs\$runId\{power,net,twin,oracle}\meta.json before analysis."

