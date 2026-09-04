# Full Factor Matrix Extension v3

This package extends the sealed production bundle for the post-primary factor
campaign declared in `factor_design.production.v3.json`.

It does **not** change or supersede the sealed `eval_v2_seed001` result. That
run remains the unfavorable primary evaluation. Calibration, gamma, weight,
and alarm thresholds remain frozen.

## Factor implementation

Each packet uses

`B_effective = minimum(B_pattern, bandwidth_cap_bps)`.

The network and twin federates apply the same transformation. Existing
pattern-starved channels therefore remain starved, while all otherwise
feasible channels are crossed with five predeclared service-rate caps:

| ID | Cap (bit/s) | Meaning |
|---|---:|---|
| `bw00_floor` | 0.5 | below `b_min`; floor endpoint |
| `bw01_10kbps` | 10,000 | lower moderate endpoint |
| `bw02_100kbps` | 100,000 | upper moderate endpoint |
| `bw03_1mbps` | 1,000,000 | ample endpoint |
| `bw04_oracle` | 1,000,000,000,000 | uncapped-pattern endpoint |

Thirty independent seed blocks are paired across bandwidth. A physical truth
is generated once per seed and reused at all five bandwidth levels. Power-noise
and network seeds are independent of the physical seed and are also held fixed
within each five-level block.

The primary mean exposure and conservative delta exposure are computed from
the same packet realization. The mean-form frozen alarm remains primary.

## Installation

From the existing bundle root, after extracting this package:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\OpenDSS_Full_Factor_Matrix_Extension_v3\install_factor_extension_v3.ps1 `
    -BundleRoot .
```

The installer refuses unknown source revisions and creates the recoverable
backup `frozen\pre_factor_v3_source_backup` before replacing anything.

## Mandatory preflight

The first preflight rebuilds the image because `net_fed.cc` changed, runs all
regression tests, verifies the factor contract, and validates all five twin
configurations:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_factor_matrix_v3.ps1 `
    -Mode Validate
```

Do not start truth generation unless it prints
`FACTOR_V3_FULL_PREFLIGHT_OK`.

## Execution in resumable batches

Generate one truth block first:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_factor_matrix_v3.ps1 `
    -Mode Truth `
    -SeedFrom 1 `
    -SeedTo 1
```

Then run its five bandwidth cells:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_factor_matrix_v3.ps1 `
    -Mode Run `
    -SeedFrom 1 `
    -SeedTo 1 `
    -BandwidthFrom 0 `
    -BandwidthTo 4
```

After auditing seed 1, proceed in small seed ranges, for example `2–5`, then
`6–10`. Completed, matching cells are validated and skipped. Incomplete or
conflicting directories cause a hard stop and are never overwritten.

`-Mode All` generates each missing truth and runs the selected cells, but small
batches are recommended for recoverability.

## Statistical status

- Unit of analysis: event, never individual timestep.
- Cluster inference by seed.
- Holm correction across the five bandwidth levels.
- Cluster bootstrap: 2,000 resamples.
- Primary comparator: estimator-matched chi-square.
- Arm G is the F3 verdict; Arm C alone cannot establish the claim.
- Secondary and delta-form scores cannot replace the frozen primary result.
- No recalibration, gamma change, or threshold change is permitted.

The full campaign comprises 150 co-simulation cells. Based on the observed
single-cell runtime, plan for multiple days of sequential execution and several
gigabytes of output. Verify free disk space before starting.
