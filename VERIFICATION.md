# Verification record

The exporter was verified on 2026-08-29 with the pinned OpenDSS runtime.

## Frozen contract

```text
OPENDSS_EXPORTER_VALIDATE_OK
nodes=278
supernodes=247
states=491
telemetry=45
loads=91
load_groups=35,67,76,97,108
jacobian_max_error=1.86e-09
telemetry_offset_norm=0.417658763
```

## Repeatability and nonlinear telemetry

Two independent 240-step invocations with the same seed reproduced all nine
scientific arrays exactly. The NPZ container hashes differ because provenance
contains the creation time; the numeric and categorical arrays do not.

```text
OPENDSS_EXPORTER_SELF_TEST_OK
repeatable_arrays=9
events=20
steps=240
nonlinear_vs_linear_max_diff=0.302135953
```

## Full production horizon

A complete 1,100-event, 13,200-step export completed without a convergence,
mapping, ordering, or finite-value failure.

```text
OPENDSS_TRUTH_VALIDATE_OK
steps=13200
events=1100
states=491
telemetry=45
affine_max_error=1.07e-07
families=load_ramp,nominal,parameter_change,topology_change
```

Exact event allocation from the frozen design:

| Family | Events |
|---|---:|
| nominal | 605 |
| load_ramp | 198 |
| parameter_change | 165 |
| topology_change | 132 |

