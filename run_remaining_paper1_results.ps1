[CmdletBinding()]
param(
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$bundleRoot = $PSScriptRoot
Set-Location $bundleRoot

$expectedDesignHash = `
    "f4c2b422b3fb113f1e33bd19aafd89f1591f18f1c5272d88a02d84a3d8d154f1"

function Get-HashLower {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (
        Get-FileHash `
            -LiteralPath $Path `
            -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}

function Assert-LastExitCode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Message Exit code: $LASTEXITCODE"
    }
}

function Install-FrozenFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing supplied campaign file: $Source"
    }

    if (Test-Path -LiteralPath $Destination) {
        $sourceHash = Get-HashLower -Path $Source
        $destinationHash = Get-HashLower -Path $Destination

        if ($sourceHash -ne $destinationHash) {
            throw "Existing campaign file conflicts with supplied file: $Destination"
        }

        "FROZEN_FILE_ALREADY_INSTALLED=$Destination"
        return
    }

    Copy-Item `
        -LiteralPath $Source `
        -Destination $Destination

    "FROZEN_FILE_INSTALLED=$Destination"
}

$transcriptStarted = $false
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path `
    $bundleRoot `
    "paper1_remaining_results_$timestamp.log"

try {
    Start-Transcript -LiteralPath $logPath | Out-Null
    $transcriptStarted = $true

    "PAPER1_REMAINING_RESULTS_STARTED"
    "BUNDLE_ROOT=$bundleRoot"
    "LOG=$logPath"

    # ------------------------------------------------------------
    # 1. Required package and experiment inputs
    # ------------------------------------------------------------

    $requiredPaths = @(
        ".\.env",
        ".\docker-compose.yml",
        ".\run_experiment.ps1",
        ".\run_opendss_exporter.ps1",
        ".\validate_truth.ps1",
        ".\calibration.v2.json",
        ".\gamma.frozen.v2.txt",
        ".\W.frozen.v2.npy",
        ".\physical_design.production.v1.json",
        ".\feeder.npz",
        ".\patterns.npz",
        ".\scenarios.csv",
        ".\twin_fed.py",
        ".\runs\calibration_v2_final_001\twin\scores_events.parquet",
        ".\runs\calibration_v2_final_001\oracle\oracle_events.parquet",
        ".\paper1_diagnostics_v1\freeze_paper1_thresholds.py",
        ".\paper1_diagnostics_v1\production_overlay\factor_design.paper1.v4.json",
        ".\paper1_diagnostics_v1\production_overlay\verify_paper1_factor_design_v4.py",
        ".\paper1_diagnostics_v1\production_overlay\run_paper1_factor_campaign_v4.ps1",
        ".\paper1_publication_results_v1\paper1_final_results.py",
        ".\paper1_publication_results_v1\diagnostic_inputs",
        ".\paper1_publication_results_v1\Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.tex"
    )

    foreach ($path in $requiredPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw (
                "Required Paper-1 input is missing: $path. " +
                "Extract Paper1_Campaign_Ready_Finalization_v11.zip " +
                "into the experiment-bundle root first."
            )
        }
    }

    docker version | Out-Null
    Assert-LastExitCode "Docker is not available."

    docker compose version | Out-Null
    Assert-LastExitCode "Docker Compose is not available."

    "PAPER1_REQUIRED_INPUTS_OK"

    # ------------------------------------------------------------
    # 2. Install and verify the frozen v4 campaign overlay
    # ------------------------------------------------------------

    Install-FrozenFile `
        -Source ".\paper1_diagnostics_v1\production_overlay\factor_design.paper1.v4.json" `
        -Destination ".\factor_design.paper1.v4.json"

    Install-FrozenFile `
        -Source ".\paper1_diagnostics_v1\production_overlay\verify_paper1_factor_design_v4.py" `
        -Destination ".\verify_paper1_factor_design_v4.py"

    Install-FrozenFile `
        -Source ".\paper1_diagnostics_v1\production_overlay\run_paper1_factor_campaign_v4.ps1" `
        -Destination ".\run_paper1_factor_campaign_v4.ps1"

    $designHash = Get-HashLower `
        -Path ".\factor_design.paper1.v4.json"

    if ($designHash -ne $expectedDesignHash) {
        throw (
            "Paper-1 v4 design hash mismatch. Expected " +
            "$expectedDesignHash but found $designHash"
        )
    }

    "PAPER1_V4_DESIGN_HASH_OK=$designHash"

    # ------------------------------------------------------------
    # 3. Freeze or audit calibration-only matched-FAR thresholds
    # ------------------------------------------------------------

    $thresholdPath = ".\paper1_matched_far_thresholds.v1.json"

    if (-not (Test-Path -LiteralPath $thresholdPath)) {
        docker compose --profile cosim run --rm --no-deps dev `
            python /workspace/paper1_diagnostics_v1/freeze_paper1_thresholds.py `
            --calibration-run /workspace/runs/calibration_v2_final_001 `
            --calibration-json /workspace/calibration.v2.json `
            --output /workspace/paper1_matched_far_thresholds.v1.json `
            --target-far 0.01

        Assert-LastExitCode `
            "Paper-1 matched-FAR threshold freeze failed."
    }
    else {
        "MATCHED_FAR_THRESHOLD_ALREADY_EXISTS_AUDITING"
    }

    $threshold = Get-Content `
        -LiteralPath $thresholdPath `
        -Raw |
        ConvertFrom-Json

    if (
        $threshold.schema -ne "paper1.matched_far.thresholds.v1" -or
        $threshold.source_split -ne "calibration_only" -or
        [double]$threshold.target_far -ne 0.01 -or
        [double]$threshold.quantile -ne 0.99 -or
        $threshold.quantile_method -ne "higher" -or
        $threshold.alarm_rule -ne "score > threshold" -or
        [int]$threshold.population -ne 117 -or
        $null -eq $threshold.thresholds.s -or
        $null -eq $threshold.thresholds.chi2 -or
        $threshold.proposed_threshold_exact_match -ne $true
    ) {
        throw "The matched-FAR threshold contract is invalid."
    }

    $calibrationHash = Get-HashLower `
        -Path ".\calibration.v2.json"

    $twinCalibrationHash = Get-HashLower `
        -Path ".\runs\calibration_v2_final_001\twin\scores_events.parquet"

    $oracleCalibrationHash = Get-HashLower `
        -Path ".\runs\calibration_v2_final_001\oracle\oracle_events.parquet"

    if (
        $calibrationHash -ne
        $threshold.sources.calibration_json_sha256
    ) {
        throw "Threshold calibration JSON source hash mismatch."
    }

    if (
        $twinCalibrationHash -ne
        $threshold.sources.twin_events_sha256
    ) {
        throw "Threshold Twin-event source hash mismatch."
    }

    if (
        $oracleCalibrationHash -ne
        $threshold.sources.oracle_events_sha256
    ) {
        throw "Threshold Oracle-event source hash mismatch."
    }

    $thresholdHash = Get-HashLower -Path $thresholdPath

    "PAPER1_MATCHED_FAR_THRESHOLD_AUDIT_OK"
    "THRESHOLD_SHA256=$thresholdHash"
    "S_THRESHOLD=$($threshold.thresholds.s.threshold)"
    "CHI2_THRESHOLD=$($threshold.thresholds.chi2.threshold)"

    # ------------------------------------------------------------
    # 4. Full preflight
    # ------------------------------------------------------------

    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File .\run_paper1_factor_campaign_v4.ps1 `
        -Mode Validate

    Assert-LastExitCode "Paper-1 v4 preflight failed."
    "PAPER1_PREFLIGHT_COMPLETE"

    # ------------------------------------------------------------
    # 5. Restartable 150-cell campaign in six five-seed batches
    # ------------------------------------------------------------

    $batches = @(
        [PSCustomObject]@{ From = 2;  To = 6  },
        [PSCustomObject]@{ From = 7;  To = 11 },
        [PSCustomObject]@{ From = 12; To = 16 },
        [PSCustomObject]@{ From = 17; To = 21 },
        [PSCustomObject]@{ From = 22; To = 26 },
        [PSCustomObject]@{ From = 27; To = 31 }
    )

    foreach ($batch in $batches) {
        "STARTING_PAPER1_BATCH=$($batch.From)-$($batch.To)"

        powershell.exe `
            -NoProfile `
            -ExecutionPolicy Bypass `
            -File .\run_paper1_factor_campaign_v4.ps1 `
            -Mode All `
            -SeedFrom $batch.From `
            -SeedTo $batch.To `
            -BandwidthFrom 0 `
            -BandwidthTo 4

        Assert-LastExitCode (
            "Paper-1 factor batch failed: " +
            "$($batch.From)-$($batch.To)"
        )

        "PAPER1_BATCH_COMPLETE=$($batch.From)-$($batch.To)"
    }

    # ------------------------------------------------------------
    # 6. Final fail-closed analysis and publication rendering
    # ------------------------------------------------------------

    docker compose --profile cosim run --rm --no-deps dev `
        python /workspace/paper1_publication_results_v1/paper1_final_results.py `
        confirmatory `
        --runs-root /workspace/runs `
        --design /workspace/factor_design.paper1.v4.json `
        --thresholds /workspace/paper1_matched_far_thresholds.v1.json `
        --static-results /workspace/paper1_publication_results_v1/diagnostic_inputs `
        --output /workspace/paper1_publication_results_v1/paper1_generated `
        --confirmatory-seeds 2-31 `
        --bootstrap-draws 2000

    Assert-LastExitCode `
        "Paper-1 final confirmatory analysis failed."

    $outputRoot = `
        ".\paper1_publication_results_v1\paper1_generated"

    $manifestPath = Join-Path `
        $outputRoot `
        "paper1_analysis_manifest.json"

    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Final analysis manifest was not generated."
    }

    $manifest = Get-Content `
        -LiteralPath $manifestPath `
        -Raw |
        ConvertFrom-Json

    if (
        $manifest.status -ne "confirmatory_complete" -or
        [int]$manifest.cells_found -ne 150 -or
        $manifest.qualification_seed_excluded -ne $true -or
        $manifest.threshold_source_split -ne "calibration_only" -or
        [double]$manifest.target_far -ne 0.01 -or
        $manifest.design_sha256 -ne $designHash -or
        $manifest.threshold_sha256 -ne $thresholdHash
    ) {
        throw "The final Paper-1 evidence manifest is not confirmatory-complete."
    }

    $requiredFinalOutputs = @(
        "$outputRoot\paper1_confirmatory_headline.json",
        "$outputRoot\latex\paper1_confirmatory_results.tex",
        "$outputRoot\latex\paper1_confirmatory_abstract.tex",
        "$outputRoot\latex\paper1_confirmatory_discussion.tex",
        "$outputRoot\latex\paper1_confirmatory_conclusion.tex",
        "$outputRoot\figures\fig_confirmatory_matched_far.pdf",
        "$outputRoot\figures\fig_confirmatory_roc_by_bandwidth.pdf",
        "$outputRoot\figures\fig_confirmatory_residual_silence.pdf",
        "$outputRoot\figures\fig_confirmatory_silent_latency.pdf",
        "$outputRoot\figures\fig_confirmatory_network_conditions.pdf",
        "$outputRoot\figures\fig_confirmatory_collinearity_q.pdf"
    )

    foreach ($path in $requiredFinalOutputs) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Final publication output is missing: $path"
        }
    }

    "PAPER1_FINAL_RESULTS_GATE_OK"
    "STATUS=$($manifest.status)"
    "CELLS_FOUND=$($manifest.cells_found)"
    "QUALIFICATION_SEED_EXCLUDED=$($manifest.qualification_seed_excluded)"

    # ------------------------------------------------------------
    # 7. Compile with the installed IEEE LaTeX system when present
    # ------------------------------------------------------------

    $manuscriptDir = `
        ".\paper1_publication_results_v1"

    $manuscriptName = `
        "Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.tex"

    if (-not $SkipCompile) {
        Push-Location $manuscriptDir

        try {
            $latexmk = Get-Command `
                latexmk `
                -ErrorAction SilentlyContinue

            $pdflatex = Get-Command `
                pdflatex `
                -ErrorAction SilentlyContinue

            if ($null -ne $latexmk) {
                & $latexmk.Source `
                    -pdf `
                    -interaction=nonstopmode `
                    -halt-on-error `
                    $manuscriptName

                Assert-LastExitCode `
                    "Final IEEE LaTeX compilation failed."
            }
            elseif ($null -ne $pdflatex) {
                foreach ($pass in 1..2) {
                    & $pdflatex.Source `
                        -interaction=nonstopmode `
                        -halt-on-error `
                        $manuscriptName

                    Assert-LastExitCode `
                        "Final IEEE LaTeX pass $pass failed."
                }
            }
            else {
                Write-Warning (
                    "LaTeX was not found. Results are complete, " +
                    "but the final PDF was not compiled."
                )
            }
        }
        finally {
            Pop-Location
        }
    }

    # ------------------------------------------------------------
    # 8. Seal final manuscript and generated results
    # ------------------------------------------------------------

    $finalFreezeDir = `
        ".\frozen\paper1_final_results_v11"

    if (-not (Test-Path -LiteralPath $finalFreezeDir)) {
        New-Item `
            -ItemType Directory `
            -Path $finalFreezeDir |
            Out-Null

        Copy-Item `
            -LiteralPath (
                Join-Path $manuscriptDir $manuscriptName
            ) `
            -Destination $finalFreezeDir

        Copy-Item `
            -LiteralPath ".\factor_design.paper1.v4.json" `
            -Destination $finalFreezeDir

        Copy-Item `
            -LiteralPath $thresholdPath `
            -Destination $finalFreezeDir

        Copy-Item `
            -LiteralPath $outputRoot `
            -Destination (
                Join-Path $finalFreezeDir "paper1_generated"
            ) `
            -Recurse

        $finalPdf = Join-Path `
            $manuscriptDir `
            "Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.pdf"

        if (Test-Path -LiteralPath $finalPdf) {
            Copy-Item `
                -LiteralPath $finalPdf `
                -Destination $finalFreezeDir
        }

        $filesForHashing = @(
            Get-ChildItem `
                -LiteralPath $finalFreezeDir `
                -Recurse `
                -File |
                Where-Object {
                    $_.Name -ne "FINAL_PAPER1_SHA256SUMS.csv"
                }
        )

        $hashRows = foreach ($file in $filesForHashing) {
            Get-FileHash `
                -LiteralPath $file.FullName `
                -Algorithm SHA256
        }

        $hashRows |
            Export-Csv `
                -LiteralPath (
                    Join-Path `
                        $finalFreezeDir `
                        "FINAL_PAPER1_SHA256SUMS.csv"
                ) `
                -NoTypeInformation
    }
    else {
        "PAPER1_FINAL_ARCHIVE_ALREADY_EXISTS=$finalFreezeDir"
    }

    "PAPER1_FINAL_RESULTS_AND_MANUSCRIPT_SEALED_OK"
    "FINAL_ARCHIVE=$finalFreezeDir"

    Get-Content `
        -LiteralPath (
            Join-Path `
                $outputRoot `
                "paper1_confirmatory_headline.json"
        )
}
finally {
    Set-Location $bundleRoot

    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
