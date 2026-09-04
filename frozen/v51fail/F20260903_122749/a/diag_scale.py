from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/workspace")
RUN_ID = "paper1_v5_1mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID
STEP_PATH = RUN_ROOT / "twin" / "scores.parquet"
EVENT_PATH = RUN_ROOT / "twin" / "scores_events.parquet"
TWIN_PATH = ROOT / "twin_fed.py"
REPORT_PATH = ROOT / "paper1_v5_1_guard_scale_diagnostic.json"

N = 491
HOLD_FACTOR = 50.0
OMEGA = 0.02

EXPECTED_COMPONENT_SCALE = OMEGA
EXPECTED_VECTOR_SCALE = math.sqrt(N) * OMEGA
EXPECTED_JUMP_LIMIT = HOLD_FACTOR * EXPECTED_VECTOR_SCALE


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def numbers(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def close_all(values, expected, atol=1e-15):
    return bool(
        np.allclose(
            values,
            expected,
            rtol=1e-12,
            atol=atol,
        )
    )


def unique_finite(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return [float(value) for value in np.unique(values)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


require(STEP_PATH.is_file(), f"missing step parquet: {STEP_PATH}")
require(EVENT_PATH.is_file(), f"missing event parquet: {EVENT_PATH}")
require(TWIN_PATH.is_file(), f"missing twin source: {TWIN_PATH}")

step = pd.read_parquet(
    STEP_PATH,
    columns=[
        "step_index",
        "model_increment_scale",
        "jump_limit",
        "jump_guard_policy",
    ],
)

event = pd.read_parquet(
    EVENT_PATH,
    columns=[
        "event_id",
        "model_increment_scale",
        "jump_limit_min",
        "jump_guard_policy",
    ],
)

step_scale = numbers(step["model_increment_scale"])
step_limit = numbers(step["jump_limit"])
event_scale = numbers(event["model_increment_scale"])
event_limit = numbers(event["jump_limit_min"])

require(
    np.all(np.isfinite(step_scale)),
    "step model_increment_scale contains nonfinite values",
)
require(
    np.all(np.isfinite(step_limit)),
    "step jump_limit contains nonfinite values",
)
require(
    np.all(np.isfinite(event_scale)),
    "event model_increment_scale contains nonfinite values",
)
require(
    np.all(np.isfinite(event_limit)),
    "event jump_limit contains nonfinite values",
)

step_scale_equals_component = close_all(
    step_scale,
    EXPECTED_COMPONENT_SCALE,
)
step_scale_equals_vector = close_all(
    step_scale,
    EXPECTED_VECTOR_SCALE,
)
step_limit_equals_hold_times_scale = close_all(
    step_limit,
    HOLD_FACTOR * step_scale,
)
step_limit_equals_formula = close_all(
    step_limit,
    EXPECTED_JUMP_LIMIT,
)

event_scale_equals_step_scale = close_all(
    event_scale,
    float(step_scale[0]),
)
event_limit_equals_step_limit = close_all(
    event_limit,
    float(step_limit[0]),
)

policy_step_constant = bool(
    (
        step["jump_guard_policy"].astype(str)
        == "fixed_model_increment"
    ).all()
)

policy_event_constant = bool(
    (
        event["jump_guard_policy"].astype(str)
        == "fixed_model_increment"
    ).all()
)

if (
    step_scale_equals_vector
    and step_limit_equals_hold_times_scale
    and step_limit_equals_formula
):
    classification = "MODEL_INCREMENT_SCALE_IS_FULL_STATE_VECTOR_NORM"
elif (
    step_scale_equals_component
    and close_all(
        step_limit,
        HOLD_FACTOR * math.sqrt(N) * step_scale,
    )
):
    classification = "MODEL_INCREMENT_SCALE_IS_PER_STATE_COMPONENT"
else:
    classification = "GUARD_SCALE_REQUIRES_FURTHER_DIAGNOSIS"

source_lines = TWIN_PATH.read_text(
    encoding="utf-8-sig"
).splitlines()

source_context = []
patterns = (
    "model_increment_scale",
    "jump_limit",
    "fixed_model_increment",
)

for number, line in enumerate(source_lines, start=1):
    if any(pattern in line for pattern in patterns):
        source_context.append({
            "line": int(number),
            "text": line,
        })

report = {
    "schema": "paper1.v5_1.guard.scale.diagnostic.v1",
    "run_id": RUN_ID,
    "steps": int(len(step)),
    "events": int(len(event)),
    "n": N,
    "hold_factor": HOLD_FACTOR,
    "omega": OMEGA,
    "expected_component_scale": EXPECTED_COMPONENT_SCALE,
    "expected_vector_scale": EXPECTED_VECTOR_SCALE,
    "expected_jump_limit": EXPECTED_JUMP_LIMIT,
    "step_scale_unique": unique_finite(step_scale),
    "step_limit_unique": unique_finite(step_limit),
    "event_scale_unique": unique_finite(event_scale),
    "event_limit_unique": unique_finite(event_limit),
    "step_scale_equals_component": step_scale_equals_component,
    "step_scale_equals_vector": step_scale_equals_vector,
    "step_limit_equals_hold_times_scale": (
        step_limit_equals_hold_times_scale
    ),
    "step_limit_equals_formula": step_limit_equals_formula,
    "event_scale_equals_step_scale": event_scale_equals_step_scale,
    "event_limit_equals_step_limit": event_limit_equals_step_limit,
    "step_policy_constant": policy_step_constant,
    "event_policy_constant": policy_event_constant,
    "classification": classification,
    "twin_sha256": sha256(TWIN_PATH),
    "source_context": source_context,
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_1_GUARD_SCALE_DIAGNOSTIC")
print("RUN_ID=", RUN_ID)
print("STEPS=", len(step))
print("EVENTS=", len(event))
print("OMEGA=", OMEGA)
print("EXPECTED_COMPONENT_SCALE=", EXPECTED_COMPONENT_SCALE)
print("EXPECTED_VECTOR_SCALE=", EXPECTED_VECTOR_SCALE)
print("EXPECTED_JUMP_LIMIT=", EXPECTED_JUMP_LIMIT)
print("STEP_SCALE_UNIQUE=", json.dumps(unique_finite(step_scale)))
print("STEP_LIMIT_UNIQUE=", json.dumps(unique_finite(step_limit)))
print("EVENT_SCALE_UNIQUE=", json.dumps(unique_finite(event_scale)))
print("EVENT_LIMIT_UNIQUE=", json.dumps(unique_finite(event_limit)))
print("STEP_SCALE_EQUALS_COMPONENT=", step_scale_equals_component)
print("STEP_SCALE_EQUALS_VECTOR=", step_scale_equals_vector)
print(
    "STEP_LIMIT_EQUALS_HOLD_TIMES_SCALE=",
    step_limit_equals_hold_times_scale,
)
print("STEP_LIMIT_EQUALS_FORMULA=", step_limit_equals_formula)
print("EVENT_SCALE_EQUALS_STEP_SCALE=", event_scale_equals_step_scale)
print("EVENT_LIMIT_EQUALS_STEP_LIMIT=", event_limit_equals_step_limit)
print("STEP_POLICY_CONSTANT=", policy_step_constant)
print("EVENT_POLICY_CONSTANT=", policy_event_constant)
print("CLASSIFICATION=", classification)
print("TWIN_SHA256=", sha256(TWIN_PATH))
print("SOURCE_CONTEXT_BEGIN")
for item in source_context:
    print(f'{item["line"]:05d}: {item["text"]}')
print("SOURCE_CONTEXT_END")
print("REPORT_SHA256=", sha256(REPORT_PATH))
print("SIMULATION_RERUN=False")
print("IMPLEMENTATION_MODIFIED=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_1_GUARD_SCALE_DIAGNOSTIC_COMPLETE")
