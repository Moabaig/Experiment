[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Get-Location).Path
$workspace = Join-Path $root "paper1_v5_2_repair"

$liveTrust = Join-Path $root "trust_metric.py"
$liveTwin = Join-Path $root "twin_fed.py"
$envPath = Join-Path $root ".env"
$validator = Join-Path $root "validate_paper1_v5_2_installed_pair.py"
$builder = Join-Path $root "build_test_v5_2_twin_integration.py"

$trustCandidate = Join-Path `
    $workspace `
    "trust_metric_v5_2_candidate.py"

$twinCandidate = Join-Path `
    $workspace `
    "twin_fed_v5_2_candidate.py"

$rootCauseReport = Join-Path `
    $workspace `
    "v5_2_observability_ratcheting_diagnostic.json"

$estimatorReport = Join-Path `
    $workspace `
    "v5_2_estimator_candidate_tests.json"

$integrationReport = Join-Path `
    $workspace `
    "v5_2_twin_integration_tests.json"

$installedValidationReport = Join-Path `
    $workspace `
    "v5_2_installed_pair_validation.json"

$expectedLiveTrust = `
    "0A2627BDAACAD03E582BB039EEB2FB3AC73D33D20B77E96881EBCEEC64AAE437"
$expectedLiveTwin = `
    "39E6729AF233032AB9C58851C9682252F02D36EED739EB2EC769E165659DA34C"
$expectedTrustCandidate = `
    "936DD373A2D8A2F0B905604CA4C3DE61EC2CC889BA233AA150A24F44F2926FE6"
$expectedTwinCandidate = `
    "9CD9FFAA32DCFE2F12ED161A8D62D2D97B2AB0B4D462FDA0E97E7F46572043A4"
$expectedRootCauseReport = `
    "DEB3A8B15D8F6BF2CF942E06D45404B6C5459DB783C293D375EC0421A7C21AB2"
$expectedEstimatorReport = `
    "8A132C97EA5B41CE69127C05C377536D4441721583EB049737A129897390F9AE"
$expectedIntegrationReport = `
    "00E56B05EDD24581EBEF8DAECAFF9EED0ACBBEAE41029F43D97921823B4903A8"
$expectedBuilder = `
    "F66A91E285183A834CB8B82A4579545220DACCCAEA8BB28C10392E5207C0593D"
$expectedValidator = `
    "108B1717B5814456408F6287B58D8B881F46F4DE6B64B104E4909BB9F606F042"
$expectedEnv = `
    "55A4FCB1ACB19D86CBE2DA4BCC4FE814170A14A5A637EC6CEC97D9C94195D694"

function Get-HashUpper {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToUpperInvariant()
}

function Assert-Hash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Expected
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }

    $observed = Get-HashUpper -Path $Path
    if ($observed -ne $Expected.ToUpperInvariant()) {
        throw "Hash mismatch for $Path : $observed"
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Write-EvidenceManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EvidenceRoot
    )

    $manifestPath = Join-Path `
        $EvidenceRoot `
        "PAPER1_V5_2_INSTALL_SHA256.csv"

    $files = @(
        Get-ChildItem -LiteralPath $EvidenceRoot -File |
            Where-Object {
                $_.FullName -ne $manifestPath
            }
    )

    $rows = @(
        foreach ($file in $files) {
            [PSCustomObject]@{
                Name = $file.Name
                Length = $file.Length
                SHA256 = Get-HashUpper -Path $file.FullName
            }
        }
    )

    $rows |
        Sort-Object Name |
        Export-Csv `
            -LiteralPath $manifestPath `
            -NoTypeInformation `
            -Encoding UTF8

    return $manifestPath
}

Assert-Hash -Path $liveTrust -Expected $expectedLiveTrust
Assert-Hash -Path $liveTwin -Expected $expectedLiveTwin
Assert-Hash -Path $trustCandidate -Expected $expectedTrustCandidate
Assert-Hash -Path $twinCandidate -Expected $expectedTwinCandidate
Assert-Hash -Path $rootCauseReport -Expected $expectedRootCauseReport
Assert-Hash -Path $estimatorReport -Expected $expectedEstimatorReport
Assert-Hash -Path $integrationReport -Expected $expectedIntegrationReport
Assert-Hash -Path $builder -Expected $expectedBuilder
Assert-Hash -Path $validator -Expected $expectedValidator
Assert-Hash -Path $envPath -Expected $expectedEnv

$liveTrustBefore = Get-HashUpper -Path $liveTrust
$liveTwinBefore = Get-HashUpper -Path $liveTwin
$envBefore = Get-HashUpper -Path $envPath
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evidenceRoot = Join-Path `
    $root `
    ("frozen\v52install\I" + $timestamp)

New-Item -ItemType Directory -Path $evidenceRoot -Force |
    Out-Null

$trustBackup = Join-Path `
    $evidenceRoot `
    "trust_metric_v5_1_preinstall.py"
$twinBackup = Join-Path `
    $evidenceRoot `
    "twin_fed_v5_1_preinstall.py"
$envBackup = Join-Path $evidenceRoot "preinstall.env"

Copy-Item -LiteralPath $liveTrust -Destination $trustBackup
Copy-Item -LiteralPath $liveTwin -Destination $twinBackup
Copy-Item -LiteralPath $envPath -Destination $envBackup

foreach ($sourcePath in @(
    $trustCandidate,
    $twinCandidate,
    $rootCauseReport,
    $estimatorReport,
    $integrationReport,
    $builder,
    $validator
)) {
    Copy-Item `
        -LiteralPath $sourcePath `
        -Destination $evidenceRoot
}

$preinstallRecord = [ordered]@{
    schema = "paper1.v5_2.preinstall_record.v1"
    created_utc = [DateTime]::UtcNow.ToString("o")
    previous_trust_sha256 = $liveTrustBefore
    previous_twin_sha256 = $liveTwinBefore
    candidate_trust_sha256 = $expectedTrustCandidate
    candidate_twin_sha256 = $expectedTwinCandidate
    environment_sha256 = $envBefore
    source_integration_report_sha256 = $expectedIntegrationReport
    simulation_rerun = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}

$preinstallRecordPath = Join-Path `
    $evidenceRoot `
    "paper1_v5_2_preinstall_record.json"

Write-Utf8NoBom `
    -Path $preinstallRecordPath `
    -Text (($preinstallRecord | ConvertTo-Json -Depth 6) + "`n")

"STARTING_PAPER1_V5_2_TRANSACTIONAL_INSTALL"
"PREVIOUS_TRUST_SHA256=$liveTrustBefore"
"PREVIOUS_TWIN_SHA256=$liveTwinBefore"
"CANDIDATE_TRUST_SHA256=$expectedTrustCandidate"
"CANDIDATE_TWIN_SHA256=$expectedTwinCandidate"
"EVIDENCE_ROOT=$evidenceRoot"
"AUTHORIZATION_SCOPE=V5_2_MECHANICAL_PAIR_ONLY"
"SIMULATION_RERUN=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"

$installationStarted = $false
$validationLog = Join-Path `
    $evidenceRoot `
    "paper1_v5_2_installed_pair_validation.log"

try {
    $installationStarted = $true

    Copy-Item `
        -LiteralPath $trustCandidate `
        -Destination $liveTrust `
        -Force

    Copy-Item `
        -LiteralPath $twinCandidate `
        -Destination $liveTwin `
        -Force

    Assert-Hash -Path $liveTrust -Expected $expectedTrustCandidate
    Assert-Hash -Path $liveTwin -Expected $expectedTwinCandidate

    $validationCommand = (
        'docker compose --profile cosim run --rm --no-deps ' +
        '-e OPENBLAS_NUM_THREADS=1 ' +
        '-e OMP_NUM_THREADS=1 ' +
        'dev python /workspace/validate_paper1_v5_2_installed_pair.py ' +
        '> "{0}" 2>&1'
    ) -f $validationLog

    & cmd.exe /d /c $validationCommand
    $validationExitCode = $LASTEXITCODE

    if ($validationExitCode -ne 0) {
        throw "Installed-pair validator exited with code $validationExitCode"
    }

    $validationOutput = Get-Content `
        -LiteralPath $validationLog `
        -Raw

    if (
        -not $validationOutput.Contains(
            "PAPER1_V5_2_INSTALLED_PAIR_READY_FOR_MECHANICAL_CELL_GATE"
        )
    ) {
        throw "Installed-pair validation success marker is missing."
    }

    if (
        -not (
            Test-Path `
                -LiteralPath $installedValidationReport `
                -PathType Leaf
        )
    ) {
        throw "Installed-pair validation report was not created."
    }

    if ((Get-HashUpper -Path $envPath) -ne $envBefore) {
        throw "Installed-pair validation unexpectedly changed .env."
    }
}
catch {
    $failureMessage = $_.Exception.Message

    if ($installationStarted) {
        Copy-Item `
            -LiteralPath $trustBackup `
            -Destination $liveTrust `
            -Force
        Copy-Item `
            -LiteralPath $twinBackup `
            -Destination $liveTwin `
            -Force
        Copy-Item `
            -LiteralPath $envBackup `
            -Destination $envPath `
            -Force
    }

    $rollbackTrust = Get-HashUpper -Path $liveTrust
    $rollbackTwin = Get-HashUpper -Path $liveTwin
    $rollbackEnv = Get-HashUpper -Path $envPath

    $rollbackVerified = (
        $rollbackTrust -eq $liveTrustBefore -and
        $rollbackTwin -eq $liveTwinBefore -and
        $rollbackEnv -eq $envBefore
    )

    $failureRecord = [ordered]@{
        schema = "paper1.v5_2.install_failure.v1"
        created_utc = [DateTime]::UtcNow.ToString("o")
        failure = $failureMessage
        rollback_verified = $rollbackVerified
        restored_trust_sha256 = $rollbackTrust
        restored_twin_sha256 = $rollbackTwin
        restored_environment_sha256 = $rollbackEnv
        simulation_rerun = $false
        performance_outcome_columns_read = $false
        performance_outcomes_inspected = $false
    }

    $failureRecordPath = Join-Path `
        $evidenceRoot `
        "paper1_v5_2_install_failure.json"

    Write-Utf8NoBom `
        -Path $failureRecordPath `
        -Text (($failureRecord | ConvertTo-Json -Depth 6) + "`n")

    $failureManifest = Write-EvidenceManifest `
        -EvidenceRoot $evidenceRoot

    if (-not $rollbackVerified) {
        throw (
            "V5.2 installation failed and rollback verification also failed. " +
            "Inspect $evidenceRoot"
        )
    }

    throw (
        "V5.2 installation failed; the exact V5.1 pair and .env were restored. " +
        "Cause: $failureMessage. Evidence: $failureRecordPath. " +
        "Manifest: $failureManifest"
    )
}

Copy-Item `
    -LiteralPath $installedValidationReport `
    -Destination $evidenceRoot `
    -Force

$installRecord = [ordered]@{
    schema = "paper1.v5_2.install_record.v1"
    created_utc = [DateTime]::UtcNow.ToString("o")
    previous_trust_sha256 = $liveTrustBefore
    previous_twin_sha256 = $liveTwinBefore
    installed_trust_sha256 = Get-HashUpper -Path $liveTrust
    installed_twin_sha256 = Get-HashUpper -Path $liveTwin
    installed_validation_report_sha256 = Get-HashUpper -Path $installedValidationReport
    installed_validation_log_sha256 = Get-HashUpper -Path $validationLog
    environment_sha256_before = $envBefore
    environment_sha256_after = Get-HashUpper -Path $envPath
    installation_completed = $true
    rollback_required = $false
    authorization_scope = "v5_2_mechanical_pair_only"
    simulation_authorized = $false
    full_campaign_authorized = $false
    calibration_authorized = $false
    performance_outcome_columns_read = $false
    performance_outcomes_inspected = $false
}

$installRecordPath = Join-Path `
    $evidenceRoot `
    "paper1_v5_2_install_record.json"

Write-Utf8NoBom `
    -Path $installRecordPath `
    -Text (($installRecord | ConvertTo-Json -Depth 6) + "`n")

$manifestPath = Write-EvidenceManifest `
    -EvidenceRoot $evidenceRoot

Get-Content -LiteralPath $validationLog

"PREVIOUS_TRUST_SHA256=$liveTrustBefore"
"PREVIOUS_TWIN_SHA256=$liveTwinBefore"
"INSTALLED_TRUST_SHA256=$(Get-HashUpper -Path $liveTrust)"
"INSTALLED_TWIN_SHA256=$(Get-HashUpper -Path $liveTwin)"
"VALIDATOR_SHA256=$(Get-HashUpper -Path $validator)"
"VALIDATION_REPORT_SHA256=$(Get-HashUpper -Path $installedValidationReport)"
"INSTALL_RECORD_SHA256=$(Get-HashUpper -Path $installRecordPath)"
"INSTALL_MANIFEST_SHA256=$(Get-HashUpper -Path $manifestPath)"
"INSTALL_EVIDENCE=$evidenceRoot"
"INSTALLATION_COMPLETED=True"
"ROLLBACK_REQUIRED=False"
"AUTHORIZATION_SCOPE=V5_2_MECHANICAL_PAIR_ONLY"
"ENV_UNCHANGED=True"
"SIMULATION_AUTHORIZED=False"
"SIMULATION_RERUN=False"
"FULL_CAMPAIGN_AUTHORIZED=False"
"CALIBRATION_AUTHORIZED=False"
"PERFORMANCE_OUTCOME_COLUMNS_READ=False"
"PERFORMANCE_OUTCOMES_INSPECTED=False"
"PAPER1_V5_2_TRANSACTIONAL_INSTALL_AND_VALIDATION_COMPLETE"
