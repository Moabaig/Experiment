from pathlib import Path
from collections import Counter
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root = Path("/workspace")
run_id = "paper1_v5mv_s002_bw04_oracle"

step_path = (
    root / "runs" / run_id /
    "twin" / "scores.parquet"
)

output_path = (
    root /
    "paper1_v5_oracle_hold_diagnostic.json"
)

required = [
    "step_index",
    "event_id",
    "hold_reason",
    "held",
    "solve_exact",
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

schema = set(
    pq.ParquetFile(step_path).schema_arrow.names
)

missing = sorted(set(required) - schema)

if missing:
    raise RuntimeError(
        f"missing mechanical diagnostic columns: {missing}"
    )

frame = pd.read_parquet(
    step_path,
    columns=required,
)

if len(frame) != 13200:
    raise RuntimeError(
        f"wrong step count: {len(frame)}"
    )

def boolean(name):
    series = frame[name]

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isin(values, [0.0, 1.0]))
    ):
        raise RuntimeError(
            f"{name} is not a valid Boolean column"
        )

    return values > 0.5

def numeric(name):
    return pd.to_numeric(
        frame[name],
        errors="coerce",
    ).to_numpy(dtype=float)

def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }

    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }

def maximum_consecutive_run(mask):
    maximum = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return int(maximum)

held = boolean("held")
exact = boolean("solve_exact")
accepted = boolean(
    "state_update_accepted_step"
)
bootstrap = boolean(
    "bootstrap_accept_step"
)
inexact_hold = boolean(
    "solve_inexact_hold_step"
)
nonfinite_hold = boolean(
    "nonfinite_candidate_hold_step"
)
jump_hold = boolean(
    "jump_guard_hold_step"
)
pseudo_only = boolean(
    "pseudo_only_step"
)
external_support = boolean(
    "external_support_present_step"
)
no_received = boolean(
    "no_received_measurements_step"
)

candidate_norm = numeric("candidate_norm")
previous_norm = numeric("previous_norm")
jump_norm = numeric("jump_norm")
jump_limit = numeric("jump_limit")
external_count = numeric(
    "external_received_count"
)
external_total = numeric("external_total")
external_fraction = numeric(
    "external_support_fraction"
)
pseudo_count = numeric(
    "pseudo_received_count"
)

step_index = numeric("step_index").astype(int)
event_id = numeric("event_id").astype(int)

reasons = (
    frame["hold_reason"]
    .fillna("<missing>")
    .astype(str)
    .to_numpy()
)

reason_counts = {
    key: int(value)
    for key, value in sorted(
        Counter(reasons).items()
    )
}

if not np.array_equal(accepted, ~held):
    raise RuntimeError(
        "accepted and held columns are inconsistent"
    )

if not np.array_equal(
    inexact_hold,
    reasons == "solve_inexact",
):
    raise RuntimeError(
        "solve-inexact reason logging is inconsistent"
    )

if not np.array_equal(
    nonfinite_hold,
    reasons == "nonfinite_candidate",
):
    raise RuntimeError(
        "nonfinite reason logging is inconsistent"
    )

if not np.array_equal(
    jump_hold,
    reasons == "jump_guard",
):
    raise RuntimeError(
        "jump-guard reason logging is inconsistent"
    )

if not np.array_equal(~exact, inexact_hold):
    raise RuntimeError(
        "inexact solutions are not held consistently"
    )

bootstrap_indices = np.flatnonzero(bootstrap)

if len(bootstrap_indices) != 1:
    raise RuntimeError(
        "expected exactly one bootstrap acceptance; "
        f"found {len(bootstrap_indices)}"
    )

bootstrap_index = int(bootstrap_indices[0])
old_guard_limit = (
    50.0 * math.sqrt(491.0) * 0.01
)

finite_guard = (
    jump_hold
    & np.isfinite(jump_norm)
    & np.isfinite(jump_limit)
    & (jump_limit > 0)
)

jump_ratio = (
    jump_norm[finite_guard]
    / jump_limit[finite_guard]
)

affected_events, affected_counts = np.unique(
    event_id[jump_hold],
    return_counts=True,
)

reason_external_support = {}

for reason in sorted(set(reasons)):
    mask = reasons == reason

    reason_external_support[reason] = {
        "steps": int(np.sum(mask)),
        "external_received_mean": float(
            np.mean(external_count[mask])
        ),
        "external_received_minimum": float(
            np.min(external_count[mask])
        ),
        "external_support_fraction_mean": float(
            np.mean(external_fraction[mask])
        ),
        "pseudo_only_fraction": float(
            np.mean(pseudo_only[mask])
        ),
    }

accepted_above_old_limit = (
    accepted
    & np.isfinite(candidate_norm)
    & (candidate_norm > old_guard_limit)
)

eligible_before_jump_guard = (
    exact
    & np.isfinite(candidate_norm)
)

report = {
    "schema":
        "paper1.v5.oracle.hold.diagnostic.v1",
    "run_id": run_id,
    "performance_outcomes_inspected": False,
    "steps": int(len(frame)),
    "accepted_count": int(np.sum(accepted)),
    "held_count": int(np.sum(held)),
    "accepted_fraction": float(
        np.mean(accepted)
    ),
    "held_fraction": float(np.mean(held)),
    "steps_short_of_95_percent": int(
        max(
            0,
            math.ceil(0.95 * len(frame))
            - int(np.sum(accepted)),
        )
    ),
    "solve_exact_count": int(np.sum(exact)),
    "solve_exact_fraction": float(
        np.mean(exact)
    ),
    "bootstrap_accept_count": int(
        np.sum(bootstrap)
    ),
    "bootstrap_step_index": int(
        step_index[bootstrap_index]
    ),
    "bootstrap_candidate_norm": float(
        candidate_norm[bootstrap_index]
    ),
    "old_absolute_guard_limit": float(
        old_guard_limit
    ),
    "bootstrap_exceeds_old_guard": bool(
        candidate_norm[bootstrap_index]
        > old_guard_limit
    ),
    "accepted_above_old_guard_count": int(
        np.sum(accepted_above_old_limit)
    ),
    "solve_inexact_hold_count": int(
        np.sum(inexact_hold)
    ),
    "nonfinite_candidate_hold_count": int(
        np.sum(nonfinite_hold)
    ),
    "jump_guard_hold_count": int(
        np.sum(jump_hold)
    ),
    "eligible_before_jump_guard_count": int(
        np.sum(eligible_before_jump_guard)
    ),
    "jump_guard_affected_event_count": int(
        len(affected_events)
    ),
    "jump_guard_maximum_holds_in_one_event": int(
        np.max(affected_counts)
        if len(affected_counts)
        else 0
    ),
    "jump_guard_maximum_consecutive_steps": (
        maximum_consecutive_run(jump_hold)
    ),
    "jump_guard_step_indices_first_30": [
        int(value)
        for value in step_index[jump_hold][:30]
    ],
    "jump_norm_statistics":
        statistics(jump_norm[jump_hold]),
    "jump_limit_statistics":
        statistics(jump_limit[jump_hold]),
    "jump_to_limit_ratio_statistics":
        statistics(jump_ratio),
    "candidate_norm_accepted_statistics":
        statistics(candidate_norm[accepted]),
    "previous_norm_jump_hold_statistics":
        statistics(previous_norm[jump_hold]),
    "candidate_norm_jump_hold_statistics":
        statistics(candidate_norm[jump_hold]),
    "pseudo_only_step_count": int(
        np.sum(pseudo_only)
    ),
    "external_support_present_fraction": float(
        np.mean(external_support)
    ),
    "no_received_measurements_count": int(
        np.sum(no_received)
    ),
    "hold_reason_counts": reason_counts,
    "reason_external_support":
        reason_external_support,
}

output_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_ORACLE_HOLD_DIAGNOSTIC_OK")
print("STEPS=", len(frame))
print("ACCEPTED_COUNT=", int(np.sum(accepted)))
print("HELD_COUNT=", int(np.sum(held)))
print(
    "ACCEPTED_FRACTION=",
    float(np.mean(accepted)),
)
print(
    "STEPS_SHORT_OF_95_PERCENT=",
    report["steps_short_of_95_percent"],
)
print(
    "SOLVE_EXACT_FRACTION=",
    float(np.mean(exact)),
)
print(
    "HOLD_REASON_COUNTS=",
    json.dumps(reason_counts, sort_keys=True),
)
print(
    "BOOTSTRAP_CANDIDATE_NORM=",
    report["bootstrap_candidate_norm"],
)
print(
    "OLD_ABSOLUTE_GUARD_LIMIT=",
    old_guard_limit,
)
print(
    "ACCEPTED_ABOVE_OLD_GUARD_COUNT=",
    report["accepted_above_old_guard_count"],
)
print(
    "JUMP_GUARD_AFFECTED_EVENTS=",
    report["jump_guard_affected_event_count"],
)
print(
    "JUMP_GUARD_MAX_CONSECUTIVE_STEPS=",
    report["jump_guard_maximum_consecutive_steps"],
)
print(
    "JUMP_TO_LIMIT_RATIO_STATISTICS=",
    json.dumps(
        report[
            "jump_to_limit_ratio_statistics"
        ],
        sort_keys=True,
    ),
)
print(
    "PSEUDO_ONLY_STEP_COUNT=",
    report["pseudo_only_step_count"],
)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("OUTPUT=", output_path)