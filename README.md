# Production ns-3/HELICS digital-twin experiment bundle

This bundle implements the four-federate chain required by the experimental
design:

```text
power_fed.py -> net_fed.cc -> twin_fed.py -> oracle_fed.py
```

The broker expects exactly four federates. Endpoints are global and fixed:
`net_fed/in`, `twin_fed/in`, and `oracle_fed/in`.

## What is production and what is not

- `power_fed.py` is a production playback boundary for a real OpenDSS,
  pandapower, or Simscape trajectory.
- `net_fed.cc` creates 45 real ns-3 point-to-point links with NetDevices,
  DropTail queues, serialization rates, propagation delays, and independent
  per-packet loss realizations.
- `twin_fed.py` is the validated age-aware estimator/trust federate.
- `oracle_fed.py` is the only component allowed to create labels.
- `make_smoke_truth.py` is plumbing-only and must never supply paper results.

The repository cannot truthfully manufacture the real physical trajectory or
the predeclared drift threshold. Supply `truth.npz` as described in
`truth_contract.md`, and set `DRIFT_GAMMA` before any real run.

## Required files in this directory

Copy these already-generated inputs beside the bundle:

- `feeder.npz`
- `patterns.npz`
- `patterns.csv` (641,300 rows)
- `scenarios.csv`
- real `truth.npz`
- frozen `calibration.json` for evaluation runs

`trust_metric.py` is included because `twin_fed.py` imports it directly.
If `patterns.csv` is absent, recreate it without changing `patterns.npz`:

```powershell
docker compose run --rm --no-deps dev python /workspace/export_patterns.py `
  --input /workspace/patterns.npz --output /workspace/patterns.csv
```

## 1. Build and inspect

From PowerShell in this directory:

```powershell
Copy-Item .\.env.example .\.env
docker compose config --quiet
docker compose --profile cosim config --services
docker compose --progress plain --profile cosim build `
  2>&1 | Tee-Object .\docker_build.log
```

Expected services:

```text
dev
broker
power-fed
net-fed
twin-fed
oracle-fed
```

The Dockerfile intentionally builds HELICS and ns-3 with one C++ compiler job
to avoid the Docker Desktop/WSL2 memory failure encountered previously.

## 2. Static/data validation

For real truth, replace `0.01` below with the predeclared gamma:

```powershell
docker compose run --rm --no-deps dev python /workspace/power_fed.py `
  --feeder /workspace/feeder.npz --truth /workspace/truth.npz --seed 1001 --validate-only

docker compose run --rm --no-deps dev python /workspace/oracle_fed.py `
  --feeder /workspace/feeder.npz --truth /workspace/truth.npz --gamma 0.01 --validate-only

docker compose run --rm --no-deps dev python /workspace/twin_fed.py `
  --feeder /workspace/feeder.npz --patterns /workspace/patterns.npz `
  --scenarios /workspace/scenarios.csv --allow-uncalibrated --validate-only

docker compose run --rm --no-deps dev `
  /opt/ns-allinone-3.35/ns-3.35/build/scratch/net_fed --PrintHelp
```

The full horizon must report `patterns=1100`, `steps=13200`, and
`stop=13200s`.

## 3. Two-event route smoke test

Generate quarantined truth:

```powershell
docker compose run --rm --no-deps dev python /workspace/make_smoke_truth.py `
  --feeder /workspace/feeder.npz --output /workspace/truth.smoke.npz --steps 24
```

Set the following in `.env`:

```dotenv
RUN_ID=smoke_001
TRUTH_FILE=truth.smoke.npz
CALIBRATION_MODE=1
STOP_TIME=24
DRIFT_GAMMA=0.01
POWER_SEED=1001
NETWORK_SEED=2001
NETWORK_RUN=1
```

Then run:

```powershell
.\run_experiment.ps1
```

Success requires all four `meta.json` files to report `complete`, 24 oracle
rows, two oracle event rows, no malformed protocol messages, and no missing
twin score.

## 4. Dedicated calibration run

Calibration and evaluation trajectories must be disjoint.

1. Export a real calibration `truth.npz` containing labeled drift and nominal
   intervals across ample and moderate communication conditions.
2. Set `CALIBRATION_MODE=1`, a calibration-only `RUN_ID`, `STOP_TIME=0`, and the
   predeclared `DRIFT_GAMMA`.
3. Run `run_experiment.ps1`.
4. Freeze constants from the resulting event table:

```powershell
docker compose run --rm --no-deps dev python /workspace/calibrate_twin.py `
  --input /workspace/runs/calibration_001/oracle/oracle_events.parquet `
  --output /workspace/calibration.json --scalarization lmax
```

Archive the generated calibration JSON and its source hash. Re-run preflight G1
because `u0` changes the epsilon feasibility bound.

## 5. Real evaluation run

Set `.env` with the real files and independent seeds:

```dotenv
RUN_ID=eval_seed001
TRUTH_FILE=truth.eval.seed001.npz
CALIBRATION_FILE=calibration.json
CALIBRATION_MODE=0
STOP_TIME=0
DRIFT_GAMMA=<predeclared positive value>
POWER_SEED=1001
NETWORK_SEED=2001
NETWORK_RUN=1
```

Run:

```powershell
.\run_experiment.ps1
```

Expected full-run counts before packet loss:

- 13,200 logical updates
- 1,100 events
- 594,000 power telemetry packets (`13,200 x 45`)
- 13,200 twin score messages
- 13,200 oracle step rows and 1,100 event rows

## Timing contract

The dependency chain is acyclic. Network, twin, and oracle federates enable
HELICS `wait_for_current_time_update`; this forces upstream work at a logical
time to finish before the dependent is granted that same time. The network
uses packet `event_id`, not receiver-loop position, to select impairment rows.
This prevents artificial one-step age and event-boundary pattern drift.

## Scientific stop conditions

Do not treat a run as evidence if any of the following occurs:

- calibration and evaluation trajectories overlap;
- `DRIFT_GAMMA` was chosen after inspecting evaluation results;
- physical and network seeds are coupled;
- a meta file is not `complete`;
- oracle reports a missing/duplicate/malformed score;
- twin reports malformed telemetry;
- a smoke truth file is used outside integration testing;
- Arm T is used for the primary F3 verdict.
