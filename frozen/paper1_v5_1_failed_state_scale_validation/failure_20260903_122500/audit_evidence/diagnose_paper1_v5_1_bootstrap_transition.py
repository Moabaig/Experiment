import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ID = "paper1_v5_1mv_s002_bw04_oracle"
SCORES_PATH = Path("/workspace/runs") / RUN_ID / "twin" / "scores.parquet"
REPORT_PATH = Path("/workspace/paper1_v5_1_bootstrap_transition_diagnostic.json")

COLUMNS = [
    "time",
    "step_index",
    "event_id",
    "step_in_event",
    "held",
    "solve_exact",
    "estimator_reliable",
    "estimator_solver",
    "estimator_rcond",
    "estimator_effective_rows",
    "estimator_rank",
    "estimator_condition",
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
    "external_support_present_step",
    "pseudo_received_count",
    "pseudo_only_step",
    "no_received_measurements_step",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bool_array(frame, column):
    return frame[column].fillna(False).astype(bool).to_numpy()


def first_position(mask):
    positions = np.flatnonzero(mask)
    return None if len(positions) == 0 else int(positions[0])


def native(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def row_dictionary(frame, position):
    return {
        column: native(frame.iloc[position][column])
        for column in frame.columns
    }


require(SCORES_PATH.is_file(), f"missing step parquet: {SCORES_PATH}")

scores = pd.read_parquet(SCORES_PATH, columns=COLUMNS)
require(len(scores) == 13200, f"unexpected step count: {len(scores)}")

step_index = pd.to_numeric(
    scores["step_index"], errors="raise"
).to_numpy(dtype=np.int64)

require(
    np.array_equal(step_index, np.arange(len(scores), dtype=np.int64)),
    "step_index is not the exact zero-based sequence",
)

bootstrap = bool_array(scores, "bootstrap_accept_step")
reliable = bool_array(scores, "estimator_reliable")
solve_exact = bool_array(scores, "solve_exact")
accepted = bool_array(scores, "state_update_accepted_step")
held = bool_array(scores, "held")

bootstrap_positions = [
    int(value) for value in np.flatnonzero(bootstrap)
]

first_bootstrap = first_position(bootstrap)
first_reliable = first_position(reliable)
first_solve_exact = first_position(solve_exact)
first_accepted = first_position(accepted)

if first_bootstrap is None:
    window_start = 0
    window_end = min(20, len(scores))
else:
    window_start = max(0, first_bootstrap - 10)
    window_end = min(len(scores), first_bootstrap + 11)

preview_columns = [
    "step_index",
    "event_id",
    "step_in_event",
    "held",
    "solve_exact",
    "estimator_reliable",
    "estimator_rank",
    "estimator_effective_rows",
    "estimator_condition",
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
    "no_received_measurements_step",
]

preview = [
    row_dictionary(scores[preview_columns], position)
    for position in range(window_start, window_end)
]

pre_bootstrap_accepted = (
    None
    if first_bootstrap is None
    else int(np.count_nonzero(accepted[:first_bootstrap]))
)

pre_bootstrap_reliable = (
    None
    if first_bootstrap is None
    else int(np.count_nonzero(reliable[:first_bootstrap]))
)

pre_bootstrap_reasons = {}
if first_bootstrap is not None:
    counts = (
        scores.iloc[:first_bootstrap]["hold_reason"]
        .fillna("<null>")
        .astype(str)
        .value_counts()
    )
    pre_bootstrap_reasons = {
        str(key): int(value)
        for key, value in counts.items()
    }

first_reliable_bootstrap_invariant = bool(
    len(bootstrap_positions) == 1
    and first_bootstrap is not None
    and first_bootstrap == first_reliable
    and first_bootstrap == first_accepted
    and reliable[first_bootstrap]
    and solve_exact[first_bootstrap]
    and accepted[first_bootstrap]
    and not held[first_bootstrap]
    and not np.any(accepted[:first_bootstrap])
    and not np.any(reliable[:first_bootstrap])
)

if first_reliable_bootstrap_invariant:
    classification = (
        "VALID_FIRST_RELIABLE_BOOTSTRAP_AUDITOR_STEP_ZERO_ASSUMPTION_INVALID"
    )
elif len(bootstrap_positions) == 0:
    classification = "BOOTSTRAP_MARKER_MISSING"
elif len(bootstrap_positions) > 1:
    classification = "MULTIPLE_BOOTSTRAP_MARKERS"
else:
    classification = "BOOTSTRAP_TRANSITION_REQUIRES_FURTHER_DIAGNOSIS"

report = {
    "schema": "paper1.v5_1.bootstrap.transition.diagnostic.v1",
    "run_id": RUN_ID,
    "performance_outcome_columns_read": False,
    "steps": int(len(scores)),
    "bootstrap_count": int(len(bootstrap_positions)),
    "bootstrap_positions": bootstrap_positions,
    "first_bootstrap_position": first_bootstrap,
    "first_bootstrap_step_index": (
        None
        if first_bootstrap is None
        else int(step_index[first_bootstrap])
    ),
    "first_reliable_position": first_reliable,
    "first_solve_exact_position": first_solve_exact,
    "first_accepted_position": first_accepted,
    "pre_bootstrap_accepted_count": pre_bootstrap_accepted,
    "pre_bootstrap_reliable_count": pre_bootstrap_reliable,
    "pre_bootstrap_hold_reason_counts": pre_bootstrap_reasons,
    "step_zero_assumption_supported": bool(first_bootstrap == 0),
    "first_reliable_bootstrap_invariant_supported": (
        first_reliable_bootstrap_invariant
    ),
    "classification": classification,
    "transition_window_start": int(window_start),
    "transition_window_end_exclusive": int(window_end),
    "transition_rows": preview,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_1_BOOTSTRAP_TRANSITION_DIAGNOSTIC")
print("RUN_ID=", RUN_ID)
print("STEPS=", len(scores))
print("BOOTSTRAP_COUNT=", len(bootstrap_positions))
print("BOOTSTRAP_POSITIONS=", json.dumps(bootstrap_positions))
print("FIRST_BOOTSTRAP_POSITION=", first_bootstrap)
print(
    "FIRST_BOOTSTRAP_STEP_INDEX=",
    None if first_bootstrap is None else int(step_index[first_bootstrap]),
)
print("FIRST_RELIABLE_POSITION=", first_reliable)
print("FIRST_SOLVE_EXACT_POSITION=", first_solve_exact)
print("FIRST_ACCEPTED_POSITION=", first_accepted)
print("PRE_BOOTSTRAP_ACCEPTED_COUNT=", pre_bootstrap_accepted)
print("PRE_BOOTSTRAP_RELIABLE_COUNT=", pre_bootstrap_reliable)
print(
    "PRE_BOOTSTRAP_HOLD_REASON_COUNTS=",
    json.dumps(pre_bootstrap_reasons, sort_keys=True),
)
print("STEP_ZERO_ASSUMPTION_SUPPORTED=", first_bootstrap == 0)
print(
    "FIRST_RELIABLE_BOOTSTRAP_INVARIANT_SUPPORTED=",
    first_reliable_bootstrap_invariant,
)
print("DIAGNOSTIC_CLASSIFICATION=", classification)
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("TRANSITION_ROWS_BEGIN")
for row in preview:
    print(json.dumps(row, sort_keys=True, allow_nan=False))
print("TRANSITION_ROWS_END")
print(
    "REPORT_SHA256=",
    hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest().upper(),
)
print("PAPER1_V5_1_BOOTSTRAP_TRANSITION_DIAGNOSTIC_COMPLETE")
