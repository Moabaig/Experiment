# OpenDSS production truth exporter

This package exports nonlinear IEEE 123-bus physical truth for the DT
co-simulation. It enforces the frozen 491-state ordering from `feeder.npz` and
records 45 physical telemetry channels at every solved step.

## What is exact and what is newly declared

The following contracts are recovered exactly from the supplied production
inputs:

- IEEE 123-bus master SHA-256:
  `C92A69D9B218B1B2646EC7911783826229309038E72F16B848304C0457C0A54D`
- 278 OpenDSS Y-nodes, merged into 247 frozen supernodes
- 491-state definition and ordering
- 45-channel telemetry definition and ordering
- pinned OpenDSSDirect.py 0.9.4, dss-python 0.15.7, and backend 0.14.5

The missing historical exporter could not be recovered. Consequently, the
event mix and drift magnitudes in `physical_design.production.v1.json` are a
new, explicit reconstruction. Review and freeze that JSON before generating
confirmatory data. Do not claim it reproduces the missing exporter's physical
scenario distribution.

`z_true` is not a linear surrogate. Each row is obtained from the converged
OpenDSS voltage and current state and is transformed into the frozen estimator
coordinates by

```text
z_true(x) = h_OpenDSS(x) + H_telemetry @ x0 - h_OpenDSS(x0)
```

The unshifted nonlinear measurements are retained as `z_physical`.

## One-time Docker image build

Run these commands in the experiment-bundle directory after copying this
package's files into it:

```powershell
$ErrorActionPreference = "Stop"

docker build `
    --file .\Dockerfile.opendss-exporter `
    --tag dt-opendss-exporter:0.1 `
    .

if ($LASTEXITCODE -ne 0) {
    throw "OpenDSS exporter image build failed."
}
```

Verify the pinned runtime without fragile nested shell quoting:

```powershell
$versionCode = @'
import importlib.metadata as m
import opendssdirect

print("OPENDSSDIRECT=", m.version("OpenDSSDirect.py"))
print("DSS_PYTHON=", m.version("dss-python"))
print("DSS_BACKEND=", m.version("dss-python-backend"))
print("OPENDSS_IMPORT_OK")
'@

docker run --rm `
    --entrypoint python `
    dt-opendss-exporter:0.1 `
    -c $versionCode

if ($LASTEXITCODE -ne 0) {
    throw "Pinned OpenDSS runtime verification failed."
}
```

## Freeze the physical design

Read `physical_design.production.v1.json`, make any scientifically justified
changes before looking at new outcomes, then record its hash:

```powershell
Get-FileHash `
    .\physical_design.production.v1.json, `
    .\export_opendss_truth.py, `
    .\feeder.npz, `
    .\opendss\123Bus\IEEE123Master.dss `
    -Algorithm SHA256 |
    Format-Table -AutoSize
```

## Validate the model-to-feeder contract

The `-WeightSource` file must contain the frozen 491-by-491 `W` array. The
existing calibration truth may be used only as a source of that already-frozen
array; none of its trajectories, labels, or telemetry are reused.

```powershell
.\run_opendss_exporter.ps1 `
    -Mode Validate `
    -WeightSource .\truth.calibration.npz `
    -Role calibration.v2 `
    -Seed 51031
```

Required marker:

```text
OPENDSS_EXPORTER_VALIDATE_OK ... states=491 telemetry=45 ...
```

Run the packaged real-OpenDSS repeatability test once:

```powershell
docker run --rm `
    --mount "type=bind,source=$((Get-Location).Path),target=/workspace" `
    --workdir /workspace `
    dt-opendss-exporter:0.1 `
    /workspace/self_test_opendss_exporter.py

if ($LASTEXITCODE -ne 0) {
    throw "OpenDSS exporter self-test failed."
}
```

Required marker: `OPENDSS_EXPORTER_SELF_TEST_OK`.

## Generate new calibration truth

Use a new physical seed because the old calibration/evaluation artifacts were
generated without physical `z_true` and are not confirmatory inputs.

```powershell
.\run_opendss_exporter.ps1 `
    -Mode Export `
    -WeightSource .\truth.calibration.npz `
    -Role calibration.v2 `
    -Seed 51031 `
    -Output .\truth.calibration.v2.npz
```

Expected marker:

```text
OPENDSS_TRUTH_EXPORT_OK ... steps=13200 events=1100 states=491 telemetry=45 ...
```

Validate the artifact:

```powershell
.\validate_truth.ps1 `
    -Truth .\truth.calibration.v2.npz `
    -Role calibration.v2 `
    -Seed 51031
```

## Generate the first disjoint evaluation truth

```powershell
.\run_opendss_exporter.ps1 `
    -Mode Export `
    -WeightSource .\truth.calibration.npz `
    -Role eval.v2.seed001 `
    -Seed 81001 `
    -Output .\truth.eval.v2.seed001.npz

.\validate_truth.ps1 `
    -Truth .\truth.eval.v2.seed001.npz `
    -Role eval.v2.seed001 `
    -Seed 81001 `
    -DisjointWith .\truth.calibration.v2.npz
```

Required marker:

```text
OPENDSS_TRUTH_VALIDATE_OK ... states=491 telemetry=45 ...
```

## Inspect provenance and hashes

```powershell
$truthFiles = @(
    ".\truth.calibration.v2.npz",
    ".\truth.eval.v2.seed001.npz"
)

Get-FileHash $truthFiles -Algorithm SHA256 |
    Format-Table -AutoSize

$inspectCode = @'
import json
import sys
import numpy as np

for path in sys.argv[1:]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta"].item())
        families, counts = np.unique(data["drift_family"], return_counts=True)
        print("=" * 72)
        print("FILE=", path)
        print("X_SHAPE=", data["x_true"].shape)
        print("Z_SHAPE=", data["z_true"].shape)
        print("PHYSICAL_SEED=", data["physical_seed"].item())
        print("EVENT_COUNTS=", dict(zip(families.tolist(), counts.tolist())))
        print("META=", json.dumps(meta, indent=2, sort_keys=True))
'@

docker run --rm `
    --mount "type=bind,source=$((Get-Location).Path),target=/workspace" `
    --workdir /workspace `
    --entrypoint python `
    dt-opendss-exporter:0.1 `
    -c $inspectCode `
    /workspace/truth.calibration.v2.npz `
    /workspace/truth.eval.v2.seed001.npz
```

## Mandatory downstream reset

After generating these v2 truth files, rerun the calibration-only gamma pilot,
freeze a new gamma, run the corrected step-level `calibrate_twin.py`, and then
generate fresh evaluation results. Do not reuse `calibration_final_001`,
`eval_seed001`, or the v1 `calibration.json` as confirmatory evidence.
