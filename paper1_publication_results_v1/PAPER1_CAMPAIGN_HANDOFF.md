# Paper 1 final campaign handoff

## Current status

The v11 manuscript, mathematical formulation, static information-geometry
diagnostics, publication figures, statistical analysis code, and fail-closed
LaTeX integration are complete. Detector-performance claims remain gated on
the predeclared 150-cell campaign: 30 unseen physical seeds (indices 2--31)
crossed with five bandwidth levels. Seed index 1 / physical seed 81001 is a
qualification run and must remain excluded.

## 1. Install the v4 campaign files

From the real experiment-bundle root in PowerShell, copy the supplied files
only if the corresponding v4 files are not already installed:

```powershell
$ErrorActionPreference = "Stop"

Copy-Item `
    .\paper1_diagnostics_v1\production_overlay\factor_design.paper1.v4.json `
    .\factor_design.paper1.v4.json

Copy-Item `
    .\paper1_diagnostics_v1\production_overlay\verify_paper1_factor_design_v4.py `
    .\verify_paper1_factor_design_v4.py

Copy-Item `
    .\paper1_diagnostics_v1\production_overlay\run_paper1_factor_campaign_v4.ps1 `
    .\run_paper1_factor_campaign_v4.ps1
```

Do not replace the earlier production design; the Paper-1 v4 design is a
separate frozen contract.

## 2. Freeze the matched-FAR thresholds once

This command must be run before any confirmatory output is inspected. The
freezer refuses to overwrite an existing file.

```powershell
$ErrorActionPreference = "Stop"

docker compose --profile cosim run --rm --no-deps dev `
    python /workspace/paper1_diagnostics_v1/freeze_paper1_thresholds.py `
    --calibration-run /workspace/runs/calibration_v2_final_001 `
    --calibration-json /workspace/calibration.v2.json `
    --output /workspace/paper1_matched_far_thresholds.v1.json `
    --target-far 0.01

if ($LASTEXITCODE -ne 0) {
    throw "Paper-1 matched-FAR threshold freeze failed."
}

Get-FileHash `
    .\paper1_matched_far_thresholds.v1.json `
    -Algorithm SHA256
```

If the threshold file already exists, audit it rather than regenerating it.

## 3. Run the full preflight

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_paper1_factor_campaign_v4.ps1 `
    -Mode Validate

if ($LASTEXITCODE -ne 0) {
    throw "Paper-1 v4 preflight failed."
}
```

The expected terminal marker is `PAPER1_FACTOR_V4_FULL_PREFLIGHT_OK`, with
30 seeds, five bandwidth levels, and 150 cells.

## 4. Execute restartable five-seed batches

Each command exports/validates truth and runs all five bandwidth levels. A
completed cell is verified and skipped on restart.

```powershell
$ErrorActionPreference = "Stop"

$batches = @(
    @(2, 6),
    @(7, 11),
    @(12, 16),
    @(17, 21),
    @(22, 26),
    @(27, 31)
)

foreach ($batch in $batches) {
    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File .\run_paper1_factor_campaign_v4.ps1 `
        -Mode All `
        -SeedFrom $batch[0] `
        -SeedTo $batch[1] `
        -BandwidthFrom 0 `
        -BandwidthTo 4

    if ($LASTEXITCODE -ne 0) {
        throw "Paper-1 batch failed: $($batch[0])-$($batch[1])"
    }
}

"PAPER1_150_CELL_CAMPAIGN_FINISHED"
```

## 5. Generate final figures, tables, and manuscript prose

```powershell
$ErrorActionPreference = "Stop"

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

if ($LASTEXITCODE -ne 0) {
    throw "Paper-1 final analysis failed or the evidence gate did not pass."
}

$manifest = Get-Content `
    .\paper1_publication_results_v1\paper1_generated\paper1_analysis_manifest.json `
    -Raw | ConvertFrom-Json

if (
    $manifest.status -ne "confirmatory_complete" -or
    [int]$manifest.cells_found -ne 150 -or
    -not $manifest.qualification_seed_excluded
) {
    throw "Paper-1 confirmatory manifest is not final."
}

"PAPER1_FINAL_RESULTS_READY"
Get-Content `
    .\paper1_publication_results_v1\paper1_generated\paper1_confirmatory_headline.json
```

The analyzer will then create the final results fragment, four narrative
fragments, seed-clustered tables, matched-FAR and latency figures,
availability--geometry collinearity diagnostics, F3 Arm-G comparisons, and a
SHA-256 manifest. It will not render final outputs from a partial campaign.

## 6. Compile the paper

Compile
`Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.tex` with a full
IEEE LaTeX installation. The generated directory must remain adjacent to the
manuscript. Run LaTeX twice so cross-references resolve. Replace the anonymous
author block only after the target venue and review mode are fixed.

## Interpretation boundary

- F1 is the measured residual-silence rate, not an assumed property.
- F2 uses paired seed-level comparison with estimator-matched chi-square.
- F3 uses Arm G full metric versus residual+B1/B2; Arm C alone cannot pass F3.
- Report Spearman(B1, u) and matched-pair fraction q before interpreting F3.
- Keep unfavorable or null outcomes. Do not recalibrate, change gamma, or
  select bandwidth levels after inspecting campaign output.
