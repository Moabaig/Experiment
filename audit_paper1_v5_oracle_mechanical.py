from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root = Path("/workspace")
run_id = "paper1_v5mv_s002_bw04_oracle"
run_root = root / "runs" / run_id

step_path = run_root / "twin" / "scores.parquet"
event_path = run_root / "twin" / "scores_events.parquet"
record_path = (
    run_root /
    "cell_record.paper1.v5.mechanical.json"
)
output_path = (
    root /
    "paper1_v5_oracle_mechanical_audit.json"
)

expected_steps = 13200
expected_events = 1100
steps_per_event = 12

old_absolute_guard_limit = (
    50.0 * math.sqrt(491.0) * 0.01
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def bool_values(series, name):
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    require(
        np.all(np.isfinite(values)),
        f"{name} contains nonfinite values",
    )
    require(
        np.all(np.isin(values, [0.0, 1.0])),
        f"{name} contains values other than zero or one",
    )
    return values > 0.5

def numeric_values(frame, name):
    return pd.to_numeric(
        frame[name],
        errors="coerce",
    ).to_numpy(dtype=float)

def finite_reduce(values, operation):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    if operation == "max":
        return float(np.max(values))
    if operation == "min":
        return float(np.min(values))

    raise ValueError(operation)

def assert_close(actual, expected, name):
    if math.isnan(expected):
        require(
            math.isnan(actual),
            f"{name}: expected NaN but found {actual}",
        )
    else:
        require(
            math.isclose(
                actual,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ),
            f"{name}: {actual} != {expected}",
        )

# Verify completion metadata without examining outcomes.
for service in ("power", "net", "twin", "oracle"):
    meta_path = run_root / service / "meta.json"
    require(
        meta_path.exists(),
        f"missing metadata: {meta_path}",
    )

    meta = json.loads(
        meta_path.read_text(encoding="utf-8-sig")
    )

    require(
        meta.get("status") == "complete",
        f"incomplete service metadata: {meta_path}",
    )

require(record_path.exists(), "missing V5 cell record")

record = json.loads(
    record_path.read_text(encoding="utf-8-sig")
)

require(
    record.get("schema") ==
    "twin.factor.cell.record.paper1.v5.mechanical",
    "wrong mechanical cell-record schema",
)

require(
    record.get("run_id") == run_id,
    "cell-record run identifier mismatch",
)

require(step_path.exists(), "missing step results")
require(event_path.exists(), "missing event results")

step_new_fields = [
    "hold_reason",
    "state_update_accepted_step",
    "bootstrap_accept_step",
    "solve_inexact_hold_step",
    "nonfinite_candidate_hold_step",
    "jump_guard_hold_step",
    "candidate_norm",
    "previous_norm",
    "jump_norm",
    "jump_limit",
    "external_received_count",
    "external_total",
    "external_support_fraction",
    "pseudo_received_count",
    "pseudo_only_step",
    "external_support_present_step",
    "no_received_measurements_step",
]

event_new_fields = [
    "solve_exact_fraction",
    "solve_exact_all",
    "state_update_accepted_fraction",
    "bootstrap_accept_fraction",
    "solve_inexact_hold_fraction",
    "nonfinite_candidate_hold_fraction",
    "jump_guard_hold_fraction",
    "pseudo_only_fraction",
    "external_support_present_fraction",
    "no_received_measurements_fraction",
    "candidate_norm_max",
    "previous_norm_max",
    "jump_norm_max",
    "jump_limit_min",
    "external_received_count_min",
    "external_total",
    "external_support_fraction_min",
    "pseudo_received_count_max",
]

step_schema = set(
    pq.ParquetFile(step_path).schema_arrow.names
)
event_schema = set(
    pq.ParquetFile(event_path).schema_arrow.names
)

missing_step = sorted(
    set(step_new_fields) - step_schema
)
missing_event = sorted(
    set(event_new_fields) - event_schema
)

require(
    not missing_step,
    f"missing step fields: {missing_step}",
)
require(
    not missing_event,
    f"missing event fields: {missing_event}",
)

# Read mechanical diagnostics only.
step_columns = [
    "step_index",
    "event_id",
    "held",
    "solve_exact",
] + step_new_fields

event_columns = [
    "event_id",
] + event_new_fields

steps = pd.read_parquet(
    step_path,
    columns=step_columns,
)
events = pd.read_parquet(
    event_path,
    columns=event_columns,
)

require(
    len(steps) == expected_steps,
    f"step count {len(steps)} != {expected_steps}",
)
require(
    len(events) == expected_events,
    f"event count {len(events)} != {expected_events}",
)
require(
    events["event_id"].is_unique,
    "event results contain duplicate event identifiers",
)

event_sizes = steps.groupby("event_id").size()

require(
    len(event_sizes) == expected_events,
    "wrong number of step-level event groups",
)
require(
    np.all(event_sizes.to_numpy() == steps_per_event),
    "one or more events do not contain 12 steps",
)

held = bool_values(steps["held"], "held")
solve_exact = bool_values(
    steps["solve_exact"],
    "solve_exact",
)
accepted = bool_values(
    steps["state_update_accepted_step"],
    "state_update_accepted_step",
)
bootstrap = bool_values(
    steps["bootstrap_accept_step"],
    "bootstrap_accept_step",
)
solve_inexact_hold = bool_values(
    steps["solve_inexact_hold_step"],
    "solve_inexact_hold_step",
)
nonfinite_hold = bool_values(
    steps["nonfinite_candidate_hold_step"],
    "nonfinite_candidate_hold_step",
)
jump_hold = bool_values(
    steps["jump_guard_hold_step"],
    "jump_guard_hold_step",
)
pseudo_only = bool_values(
    steps["pseudo_only_step"],
    "pseudo_only_step",
)
external_support = bool_values(
    steps["external_support_present_step"],
    "external_support_present_step",
)
no_received = bool_values(
    steps["no_received_measurements_step"],
    "no_received_measurements_step",
)

reasons = steps["hold_reason"]

require(
    not reasons.isna().any(),
    "hold_reason contains missing values",
)

reasons = reasons.astype(str).to_numpy()

allowed_reasons = {
    "bootstrap_accept",
    "accepted",
    "solve_inexact",
    "nonfinite_candidate",
    "jump_guard",
}

require(
    set(reasons).issubset(allowed_reasons),
    f"unexpected hold reasons: {sorted(set(reasons) - allowed_reasons)}",
)

require(
    np.array_equal(accepted, ~held),
    "accepted/held complement is inconsistent",
)
require(
    np.array_equal(
        bootstrap,
        reasons == "bootstrap_accept",
    ),
    "bootstrap reason indicator is inconsistent",
)
require(
    np.array_equal(
        solve_inexact_hold,
        reasons == "solve_inexact",
    ),
    "solve-inexact indicator is inconsistent",
)
require(
    np.array_equal(
        nonfinite_hold,
        reasons == "nonfinite_candidate",
    ),
    "nonfinite-candidate indicator is inconsistent",
)
require(
    np.array_equal(
        jump_hold,
        reasons == "jump_guard",
    ),
    "jump-guard indicator is inconsistent",
)
require(
    np.array_equal(
        accepted,
        np.isin(
            reasons,
            ["bootstrap_accept", "accepted"],
        ),
    ),
    "accepted-state reasons are inconsistent",
)
require(
    np.array_equal(~solve_exact, solve_inexact_hold),
    "inexact solutions were not held consistently",
)

bootstrap_indices = np.flatnonzero(bootstrap)

require(
    len(bootstrap_indices) == 1,
    f"expected one bootstrap acceptance; found {len(bootstrap_indices)}",
)

bootstrap_index = int(bootstrap_indices[0])

require(
    solve_exact[bootstrap_index],
    "bootstrap solution was not exact",
)
require(
    accepted[bootstrap_index],
    "bootstrap state was not accepted",
)
require(
    not held[bootstrap_index],
    "bootstrap state was marked held",
)

candidate_norm = numeric_values(
    steps,
    "candidate_norm",
)
previous_norm = numeric_values(
    steps,
    "previous_norm",
)
jump_norm = numeric_values(
    steps,
    "jump_norm",
)
jump_limit = numeric_values(
    steps,
    "jump_limit",
)

bootstrap_candidate_norm = float(
    candidate_norm[bootstrap_index]
)

require(
    math.isfinite(bootstrap_candidate_norm),
    "bootstrap candidate norm is nonfinite",
)
require(
    bootstrap_candidate_norm >
    old_absolute_guard_limit,
    "bootstrap candidate does not reproduce the old-guard test",
)
require(
    np.all(np.isfinite(candidate_norm[accepted])),
    "an accepted state has a nonfinite candidate norm",
)

post_bootstrap_accepted = accepted & ~bootstrap

require(
    np.all(np.isfinite(jump_norm[post_bootstrap_accepted])),
    "accepted post-bootstrap step has nonfinite jump norm",
)
require(
    np.all(np.isfinite(jump_limit[post_bootstrap_accepted])),
    "accepted post-bootstrap step has nonfinite jump limit",
)
require(
    np.all(
        jump_norm[post_bootstrap_accepted]
        <= jump_limit[post_bootstrap_accepted] + 1e-12
    ),
    "an accepted increment exceeds its jump limit",
)

if np.any(jump_hold):
    require(
        np.all(np.isfinite(jump_norm[jump_hold])),
        "jump hold has nonfinite jump norm",
    )
    require(
        np.all(np.isfinite(jump_limit[jump_hold])),
        "jump hold has nonfinite jump limit",
    )
    require(
        np.all(
            jump_norm[jump_hold] >
            jump_limit[jump_hold]
        ),
        "jump-guard hold does not exceed its limit",
    )

solve_exact_fraction = float(np.mean(solve_exact))
accepted_fraction = float(np.mean(accepted))
held_fraction = float(np.mean(held))

require(
    solve_exact_fraction >= 0.95,
    f"oracle exact-solve fraction is too low: {solve_exact_fraction}",
)
require(
    accepted_fraction >= 0.95,
    f"oracle accepted-state fraction is too low: {accepted_fraction}",
)
require(
    held_fraction <= 0.05,
    f"oracle held fraction is too high: {held_fraction}",
)

external_count = numeric_values(
    steps,
    "external_received_count",
)
external_total = numeric_values(
    steps,
    "external_total",
)
external_fraction = numeric_values(
    steps,
    "external_support_fraction",
)
pseudo_count = numeric_values(
    steps,
    "pseudo_received_count",
)

require(
    np.all(np.isfinite(external_count)),
    "external received count is nonfinite",
)
require(
    np.all(np.isfinite(external_total)),
    "external total is nonfinite",
)
require(
    np.all(external_total > 0),
    "external total must be positive",
)
require(
    np.all(external_count >= 0),
    "external received count is negative",
)
require(
    np.all(external_count <= external_total),
    "external received count exceeds total",
)
require(
    np.all(np.isfinite(external_fraction)),
    "external support fraction is nonfinite",
)
require(
    np.allclose(
        external_fraction,
        external_count / external_total,
        rtol=1e-12,
        atol=1e-15,
    ),
    "external support fraction is inconsistent",
)
require(
    np.all(np.isfinite(pseudo_count)),
    "pseudo received count is nonfinite",
)
require(
    np.all(pseudo_count >= 0),
    "pseudo received count is negative",
)
require(
    np.array_equal(
        external_support,
        external_count > 0,
    ),
    "external-support indicator is inconsistent",
)
require(
    np.array_equal(
        pseudo_only,
        (external_count == 0) & (pseudo_count > 0),
    ),
    "pseudo-only indicator is inconsistent",
)
require(
    np.array_equal(
        no_received,
        (external_count == 0) & (pseudo_count == 0),
    ),
    "no-received-measurements indicator is inconsistent",
)
require(
    not np.any(pseudo_only),
    "oracle cell unexpectedly contains pseudo-only steps",
)

# Verify real event aggregation using diagnostics only.
event_rows = events.set_index("event_id")

fraction_mapping = {
    "solve_exact_fraction": solve_exact,
    "state_update_accepted_fraction": accepted,
    "bootstrap_accept_fraction": bootstrap,
    "solve_inexact_hold_fraction": solve_inexact_hold,
    "nonfinite_candidate_hold_fraction": nonfinite_hold,
    "jump_guard_hold_fraction": jump_hold,
    "pseudo_only_fraction": pseudo_only,
    "external_support_present_fraction": external_support,
    "no_received_measurements_fraction": no_received,
}

step_event_ids = pd.to_numeric(
    steps["event_id"],
    errors="raise",
).to_numpy(dtype=int)

for event_id in sorted(np.unique(step_event_ids)):
    mask = step_event_ids == event_id

    require(
        event_id in event_rows.index,
        f"event {event_id} missing from event results",
    )

    event_row = event_rows.loc[event_id]

    for event_column, values in fraction_mapping.items():
        actual = float(event_row[event_column])
        expected = float(np.mean(values[mask]))
        assert_close(
            actual,
            expected,
            f"event {event_id} {event_column}",
        )

    event_exact_all = bool(event_row["solve_exact_all"])

    require(
        event_exact_all == bool(np.all(solve_exact[mask])),
        f"event {event_id} solve_exact_all mismatch",
    )

    reductions = {
        "candidate_norm_max": (
            candidate_norm,
            "max",
        ),
        "previous_norm_max": (
            previous_norm,
            "max",
        ),
        "jump_norm_max": (
            jump_norm,
            "max",
        ),
        "jump_limit_min": (
            jump_limit,
            "min",
        ),
        "external_received_count_min": (
            external_count,
            "min",
        ),
        "external_total": (
            external_total,
            "max",
        ),
        "external_support_fraction_min": (
            external_fraction,
            "min",
        ),
        "pseudo_received_count_max": (
            pseudo_count,
            "max",
        ),
    }

    for event_column, (values, operation) in reductions.items():
        actual = float(event_row[event_column])
        expected = finite_reduce(
            values[mask],
            operation,
        )
        assert_close(
            actual,
            expected,
            f"event {event_id} {event_column}",
        )

reason_counts = {
    reason: int(np.sum(reasons == reason))
    for reason in sorted(set(reasons))
}

report = {
    "schema": "paper1.v5.oracle.mechanical.audit.v1",
    "run_id": run_id,
    "performance_outcomes_inspected": False,
    "steps": int(len(steps)),
    "events": int(len(events)),
    "steps_per_event": steps_per_event,
    "solve_exact_fraction": solve_exact_fraction,
    "state_update_accepted_fraction": accepted_fraction,
    "held_fraction": held_fraction,
    "bootstrap_accept_count": int(np.sum(bootstrap)),
    "bootstrap_step_index": int(
        steps.iloc[bootstrap_index]["step_index"]
    ),
    "bootstrap_candidate_norm": bootstrap_candidate_norm,
    "old_absolute_guard_limit": old_absolute_guard_limit,
    "bootstrap_exceeds_old_absolute_guard": True,
    "jump_guard_hold_count": int(np.sum(jump_hold)),
    "nonfinite_candidate_hold_count": int(
        np.sum(nonfinite_hold)
    ),
    "solve_inexact_hold_count": int(
        np.sum(solve_inexact_hold)
    ),
    "pseudo_only_step_count": int(np.sum(pseudo_only)),
    "external_support_present_fraction": float(
        np.mean(external_support)
    ),
    "hold_reason_counts": reason_counts,
    "event_aggregation_verified": True,
    "mechanical_gate_passed": True,
}

output_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_ORACLE_MECHANICAL_GATE_OK")
print("STEPS=", len(steps))
print("EVENTS=", len(events))
print("SOLVE_EXACT_FRACTION=", solve_exact_fraction)
print("STATE_UPDATE_ACCEPTED_FRACTION=", accepted_fraction)
print("HELD_FRACTION=", held_fraction)
print("BOOTSTRAP_ACCEPT_COUNT=", int(np.sum(bootstrap)))
print("BOOTSTRAP_CANDIDATE_NORM=", bootstrap_candidate_norm)
print("OLD_ABSOLUTE_GUARD_LIMIT=", old_absolute_guard_limit)
print("HOLD_REASON_COUNTS=", json.dumps(reason_counts, sort_keys=True))
print("PSEUDO_ONLY_STEP_COUNT=", int(np.sum(pseudo_only)))
print("EVENT_AGGREGATION_VERIFIED=True")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("OUTPUT=", output_path)