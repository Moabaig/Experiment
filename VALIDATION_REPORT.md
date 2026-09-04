# Validation report

## Passed locally

- Python bytecode compilation for all federates and utilities.
- Eight protocol/calibration/bundle unit tests.
- `make_smoke_truth.py`: 24 updates, 491 states, 45 telemetry channels.
- `power_fed.py --validate-only` against the smoke truth contract.
- `oracle_fed.py --validate-only` with a positive predeclared gamma.
- `twin_fed.py --validate-only` against the supplied feeder, 1,100 patterns,
  13,200 updates, and a structurally valid frozen calibration.
- `export_patterns.py`: exactly 641,300 rows, events 0–1099, channels 0–582.
- JSON Schema and Docker Compose YAML parsing.
- Exact AUC tie convention and event-label aggregation tests.

## Requires the target Docker toolchain

`net_fed.cc` includes the ns-3.35 and HELICS 3.6.1 headers that are not present
in this workspace. Its final compile/link/runtime check must therefore occur in
the supplied Dockerfile on the Windows Docker Desktop host. The Dockerfile
contains explicit build-time checks for the executable and all shared-library
dependencies.

## Intentionally not fabricated

- real OpenDSS/pandapower/Simscape `truth.npz`;
- the predeclared oracle `DRIFT_GAMMA`;
- frozen `calibration.json` derived from a disjoint real calibration run;
- a paper-result initial state derived from oracle evaluation truth.

These study inputs are scientifically consequential and cannot be manufactured
from `feeder.npz` or communication patterns alone.
