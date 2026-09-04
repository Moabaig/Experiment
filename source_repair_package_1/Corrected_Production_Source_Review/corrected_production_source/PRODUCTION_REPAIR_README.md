# Production-source repair and migration

## Outcome

This package repairs three correctness defects found during review and adds
fail-closed production checks. The previously generated `calibration.json`
(`twin.calibration.v1`), `calibration_final_001`, and `eval_seed001` are useful
diagnostic evidence only. They must not be reported as paper results.

The supplied `truth_exporter_review.zip` is not a production truth exporter.
It contains only `estimate_Q.py`, `starvation.py`, and
`make_smoke_truth.py`. No OpenDSS export source or feeder model was supplied,
so this package cannot regenerate physical telemetry. Obtain the actual
OpenDSS exporter/model and make it emit `z_true` before starting a new
calibration or evaluation.

## Correctness repairs

1. `oracle_fed.py` now aggregates event fields by their declared semantics:
   detector/error statistics use maxima, `T` and coverage use minima, and
   state/solver booleans have explicit summaries. It verifies
   `event_T = exp(-event_s)`.
2. `calibrate_twin.py` now requires step-level
   `oracle/oracle_scores.parquet`. After selecting normalizers it recomputes
   each step score and uses `max_step(r/r0 + u/u0)` per event. Separate event
   maxima of `r` and `u` cannot reconstruct that statistic.
3. `power_fed.py` requires physical `truth.z_true` by default. The previous
   `H_telemetry @ x_true` fallback is available only through the explicit
   `--allow-linearized-telemetry` smoke-test flag and is recorded in metadata.
4. `twin_fed.py` accepts only `twin.calibration.v2`, preventing accidental
   reuse of the invalidated v1 artifact.
5. `run_experiment.ps1` performs truth validation before starting HELICS and
   checks the frozen calibration schema in production mode.

The threshold FAR is calibrated on **nominal/ample events**, not on all events
with a negative Oracle label. Report both populations explicitly; they answer
different questions.

## Install the repaired files

Back up the current six files, then copy the replacements from this package
into the experiment-bundle root:

```powershell
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$backup = ".\frozen\source_before_repair_$stamp"
New-Item -ItemType Directory -Path $backup | Out-Null

$files = @(
    "calibrate_twin.py",
    "oracle_fed.py",
    "power_fed.py",
    "twin_fed.py",
    "docker-compose.yml",
    "run_experiment.ps1"
)

foreach ($file in $files) {
    Copy-Item ".\$file" $backup
    Copy-Item ".\corrected_production_source\$file" ".\$file" -Force
}
```

Adjust `corrected_production_source` if this package was extracted elsewhere.

## Required new truth contract

Each real truth split must contain at least:

- `x_true`: `(13200, 491)` physical state trajectory;
- `z_true`: `(13200, 45)` noiseless physical solver measurement outputs, in
  the frozen feeder telemetry-channel order (measurement noise is added later
  by `power_fed.py` from the frozen seed);
- `time`: exactly `1..13200` seconds;
- `event_id`: `floor(step_index / 12)`;
- `W`, physical-family metadata, trajectory IDs, and provenance metadata.

`z_true` must be exported from the power solver under the event's actual
topology and parameters. Do not manufacture it by multiplying the frozen
linear measurement matrix by `x_true`.

Validate every new split before a run:

```powershell
docker compose --profile cosim run --rm --no-deps dev `
    python /workspace/power_fed.py `
    --feeder=/workspace/feeder.npz `
    --truth=/workspace/truth.calibration.v2.npz `
    --dt=1 `
    --steps-per-event=12 `
    --stop-time=0 `
    --seed=11001 `
    --validate-only
```

The output must report `telemetry_source=truth.z_true`.

## Re-run calibration from untouched development truth

Because physical telemetry changes the Twin trajectory and Oracle error, both
`DRIFT_GAMMA` and Twin calibration must be refrozen. Do not reuse
`gamma.frozen.txt` or `calibration.json` from the prior runs.

1. Generate a new calibration truth split with physical `z_true`.
2. Use only that split to predeclare/select and freeze `DRIFT_GAMMA`.
3. Run a new full calibration co-simulation with `CALIBRATION_MODE=1`.
4. Freeze Twin calibration from the **step-level** Oracle table:

```powershell
$calRunId = "calibration_v2_001"

docker compose --profile cosim run --rm --no-deps dev `
    python /workspace/calibrate_twin.py `
    --input "/workspace/runs/$calRunId/oracle/oracle_scores.parquet" `
    --output /workspace/calibration.json `
    --scalarization lmax `
    --far 0.01 `
    --nominal-trust 0.90 `
    --moderate-target-trust 0.70
```

Confirm `calibration.json` contains:

```text
"schema": "twin.calibration.v2"
"event_score_construction": "max_step(r/r0 + u/u0)"
"calibration_population": "nominal_ample_events"
```

Then run Twin validation:

```powershell
docker compose --profile cosim run --rm --no-deps `
    -e CALIBRATION_MODE=0 `
    dev python /workspace/twin_fed.py `
    --feeder /workspace/feeder.npz `
    --patterns /workspace/patterns.npz `
    --scenarios /workspace/scenarios.csv `
    --calibration /workspace/calibration.json `
    --validate-only
```

## New evaluation only

Generate new, untouched evaluation truth seeds with physical `z_true`. Never
use an evaluation split to choose gamma, normalizers, hybrid coefficients, or
thresholds. Give each run a new ID and archive input/output hashes as before.

The old `eval_seed001` has already been inspected during diagnosis and must not
be reused as confirmatory evidence.

## Smoke-only fallback

For a quarantined smoke fixture only, set:

```text
ALLOW_LINEARIZED_TELEMETRY=1
```

The run will be marked `telemetry_source=linearized_smoke_fallback`. Such a run
is software-integration evidence, never experimental evidence.

## Regression tests

Run from the directory containing the repaired source files:

```powershell
docker compose --profile cosim run --rm --no-deps dev `
    python -m unittest discover -s /workspace/tests -v
```

Seven regression tests should pass.
