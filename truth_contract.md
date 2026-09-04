# Real power-trajectory contract (`truth.npz`)

`power_fed.py` is a playback boundary around a real OpenDSS, pandapower, or
Simscape run. It deliberately does not synthesize production physics. Export
the following arrays from the selected solver before starting HELICS.

## Required arrays

| Key | Shape | Meaning |
|---|---:|---|
| `x_true` | `(T, 491)` | Oracle state at each 1 s update, ordered exactly like `feeder.npz:H` columns. |
| `time` | `(T,)` | `1, 2, ..., T` seconds. The current federates require a regular 1 s grid. |
| `event_id` | `(T,)` | `step_index // 12`; each event therefore contains exactly 12 updates. |

## Strongly recommended arrays

| Key | Shape | Meaning |
|---|---:|---|
| `z_true` | `(T, 45)` | Noiseless physical telemetry in `H_telemetry` row order. Use this for nonlinear OpenDSS/EMT outputs. If omitted, `power_fed.py` uses the linearized `H_telemetry @ x_true`. |
| `W` | `(491, 491)` | Frozen symmetric positive-semidefinite matrix defining the oracle norm. If absent, `oracle_fed.py` uses identity and records that choice. |
| `drift_family` | `(T,)` or `(N_events,)` | `nominal`, `load_ramp`, `topology_change`, or `parameter_change`. |
| `is_nominal` | `(T,)` or `(N_events,)` | Whether the underlying physical scenario is nominal; this is for calibration selection, not the final oracle label. |
| `trajectory_id` | `(T,)` or `(N_events,)` | Stable external OpenDSS/Simscape trajectory identifier. |

## Label independence

The file must not contain detector-derived labels. The oracle alone computes

\[
d(t)=\sqrt{(x(t)-\hat{x}(t))^\top W(x(t)-\hat{x}(t))},\qquad
y(t)=\mathbf{1}\{d(t)>\gamma\}.
\]

Choose and freeze `gamma` before evaluation. Do not derive it from residuals,
alarms, `s`, `T`, B1, B2, or the evaluation ROC.

## Reproducibility rules

- Use a physical-trajectory seed independent of the network seed.
- Reuse the same truth file while sweeping communication conditions.
- Save the OpenDSS/Simscape model version, solver settings, state/channel order,
  seed, and export code alongside the NPZ.
- Keep calibration trajectories and evaluation trajectories disjoint.

`make_smoke_truth.py` creates a small linear plumbing fixture. Its output is
explicitly quarantined and must never be used in the paper.
