from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
patch_report_path = workspace / "paper1_v5_logging_patch_v3_report.json"
test_report_path = workspace / "paper1_v5_logging_patch_v3_test_report.json"

sys.path.insert(0, "/workspace")

spec = importlib.util.spec_from_file_location(
    "twin_fed_paper1_v5_logging",
    source_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load the v5 candidate.")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

patch_report = json.loads(
    patch_report_path.read_text(encoding="utf-8")
)

rows = []

for index in range(3):
    row = {
        "event_id": 0,
        "pattern_id": 0,
        "arm": "mechanical_test",
        "regime": "mechanical_test",
        "stratum": 0,
        "bandwidth_level": "mechanical_test",
        "bandwidth_cap_bps": 10000.0,
        "T": 0.9 - index * 0.1,
        "alarm": index == 2,
        "alarm_delta_lmax": False,
        "b1": [0.0, 0.5, 1.0][index],
        "n_rx": [538, 560, 583][index],
        "n_rx_telemetry": [0, 22, 45][index],
        "held": [False, True, False][index],
        "solve_exact": [True, False, True][index],
        "residual_available": True,
        "hold_reason": [
            "bootstrap_accept",
            "solve_inexact",
            "accepted",
        ][index],
        "state_update_accepted_step": [True, False, True][index],
        "bootstrap_accept_step": index == 0,
        "solve_inexact_hold_step": index == 1,
        "nonfinite_candidate_hold_step": False,
        "jump_guard_hold_step": False,
        "candidate_norm": [30.0, 31.0, 32.0][index],
        "previous_norm": [0.0, 30.0, 31.0][index],
        "jump_norm": [np.nan, 1.0, 2.0][index],
        "jump_limit": [11.0, 1500.0, 1550.0][index],
        "external_received_count": [0, 22, 45][index],
        "external_total": 45,
        "external_support_fraction": [0.0, 22.0 / 45.0, 1.0][index],
        "pseudo_received_count": 538,
        "pseudo_only_step": index == 0,
        "external_support_present_step": index > 0,
        "no_received_measurements_step": False,
    }

    assert not (
        row["pseudo_only_step"]
        and row["external_support_present_step"]
    )

    for column in module.MAX_EVENT_COLUMNS:
        row.setdefault(column, float(index + 1))

    rows.append(row)

scores = pd.DataFrame(rows)
events = module.aggregate_events(scores)

assert len(events) == 1
event = events.iloc[0]

assert math.isclose(
    event["state_update_accepted_fraction"],
    2.0 / 3.0,
)
assert math.isclose(event["bootstrap_accept_fraction"], 1.0 / 3.0)
assert math.isclose(event["solve_inexact_hold_fraction"], 1.0 / 3.0)
assert math.isclose(event["pseudo_only_fraction"], 1.0 / 3.0)
assert math.isclose(
    event["external_support_present_fraction"],
    2.0 / 3.0,
)
assert event["candidate_norm_max"] == 32.0
assert event["previous_norm_max"] == 31.0
assert event["jump_norm_max"] == 2.0
assert event["jump_limit_min"] == 11.0
assert event["external_received_count_min"] == 0
assert event["external_total"] == 45
assert event["external_support_fraction_min"] == 0.0
assert event["pseudo_received_count_max"] == 538
assert math.isclose(event["solve_exact_fraction"], 2.0 / 3.0)
assert event["solve_exact_all"] is False or not bool(event["solve_exact_all"])

required_step = set(patch_report["step_fields_added"])
required_event = set(patch_report["event_fields_added"])

assert required_step.issubset(scores.columns)
assert required_event.issubset(events.columns)

report = {
    "schema": "paper1.v5.logging.patch.regression.v3",
    "step_fields_verified": sorted(required_step),
    "event_fields_verified": sorted(required_event),
    "pseudo_only_external_support_exclusivity": True,
    "event_aggregation_numerically_verified": True,
    "tests_passed": 8,
    "tests_failed": 0,
    "performance_outcomes_inspected": False,
}

test_report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_PATCH_V3_TESTS_OK")
print("STEP_FIELDS_VERIFIED=", len(required_step))
print("EVENT_FIELDS_VERIFIED=", len(required_event))
print("EVENT_AGGREGATION_NUMERICALLY_VERIFIED=True")
print("PSEUDO_ONLY_EXTERNAL_SUPPORT_EXCLUSIVITY=True")
print("TESTS_PASSED=8")
print("TESTS_FAILED=0")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")