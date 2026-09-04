from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
patch_report_path = workspace / "paper1_v5_guard_patch_report.json"
test_report_path = workspace / "paper1_v5_guard_test_report.json"

sys.path.insert(0, "/workspace")

patch_report = json.loads(
    patch_report_path.read_text(encoding="utf-8")
)

class_name = patch_report["patched_class"]
method_name = patch_report["patched_method"]

spec = importlib.util.spec_from_file_location(
    "twin_fed_paper1_v5_candidate",
    source_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load the patched v5 candidate.")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

estimator_class = getattr(module, class_name)


class FakeMetric:
    def __init__(self):
        self.candidate = None
        self.exact = True

    def set_result(self, candidate, exact):
        self.candidate = np.asarray(candidate, dtype=float).copy()
        self.exact = bool(exact)

    def estimate(self, z, rx, gamma):
        return self.candidate.copy(), self.exact


def make_estimator():
    estimator = estimator_class.__new__(estimator_class)
    estimator.n = 491
    estimator.n_telemetry = 45
    estimator.hold_factor = 50.0
    estimator.x_previous = np.zeros(estimator.n, dtype=float)
    estimator.has_valid_estimate = False
    estimator.metric = FakeMetric()
    return estimator


def invoke(estimator, candidate, exact=True, external=True, pseudo=True):
    estimator.metric.set_result(candidate, exact)

    rx = np.zeros(60, dtype=bool)

    if external:
        rx[:estimator.n_telemetry] = True

    if pseudo:
        rx[estimator.n_telemetry:] = True

    z = np.zeros(len(rx), dtype=float)
    gamma = np.ones(len(rx), dtype=float)

    method = getattr(estimator, method_name)
    state, held, solved_exactly = method(z, rx, gamma)

    return np.asarray(state), bool(held), bool(solved_exactly)


results = []

estimator = make_estimator()

bootstrap_candidate = np.full(
    estimator.n,
    31.0 / math.sqrt(estimator.n),
)

state, held, exact = invoke(
    estimator,
    bootstrap_candidate,
    exact=True,
    external=True,
    pseudo=True,
)

assert exact is True
assert held is False
assert estimator.has_valid_estimate is True
assert estimator.last_hold_reason == "bootstrap_accept"
assert np.allclose(state, bootstrap_candidate)
assert math.isclose(
    np.linalg.norm(state),
    31.0,
    rel_tol=1e-12,
    abs_tol=1e-12,
)

results.append({
    "test": "valid_norm_31_bootstrap",
    "passed": True,
    "held": held,
    "reason": estimator.last_hold_reason,
    "state_norm": float(np.linalg.norm(state)),
})

accepted_state = bootstrap_candidate + 0.001

state, held, exact = invoke(
    estimator,
    accepted_state,
    exact=True,
    external=True,
    pseudo=True,
)

assert exact is True
assert held is False
assert estimator.last_hold_reason == "accepted"
assert np.allclose(state, accepted_state)

results.append({
    "test": "small_post_bootstrap_increment",
    "passed": True,
    "held": held,
    "reason": estimator.last_hold_reason,
    "jump_norm": estimator.last_jump_norm,
    "jump_limit": estimator.last_jump_limit,
})

previous_state = estimator.x_previous.copy()

state, held, exact = invoke(
    estimator,
    accepted_state + 0.002,
    exact=False,
    external=True,
    pseudo=True,
)

assert exact is False
assert held is True
assert estimator.last_hold_reason == "solve_inexact"
assert np.allclose(state, previous_state)
assert np.allclose(estimator.x_previous, previous_state)

results.append({
    "test": "inexact_solution_rejected",
    "passed": True,
    "held": held,
    "reason": estimator.last_hold_reason,
})

nonfinite_candidate = accepted_state.copy()
nonfinite_candidate[0] = np.nan

state, held, exact = invoke(
    estimator,
    nonfinite_candidate,
    exact=True,
    external=True,
    pseudo=True,
)

assert exact is True
assert held is True
assert estimator.last_hold_reason == "nonfinite_candidate"
assert np.allclose(state, previous_state)

results.append({
    "test": "nonfinite_candidate_rejected",
    "passed": True,
    "held": held,
    "reason": estimator.last_hold_reason,
})

extreme_candidate = previous_state.copy()
extreme_candidate[0] += 2000.0

state, held, exact = invoke(
    estimator,
    extreme_candidate,
    exact=True,
    external=True,
    pseudo=True,
)

assert exact is True
assert held is True
assert estimator.last_hold_reason == "jump_guard"
assert estimator.last_jump_norm > estimator.last_jump_limit
assert np.allclose(state, previous_state)

results.append({
    "test": "extreme_increment_rejected",
    "passed": True,
    "held": held,
    "reason": estimator.last_hold_reason,
    "jump_norm": estimator.last_jump_norm,
    "jump_limit": estimator.last_jump_limit,
})

pseudo_estimator = make_estimator()

state, held, exact = invoke(
    pseudo_estimator,
    bootstrap_candidate,
    exact=True,
    external=False,
    pseudo=True,
)

assert exact is True
assert held is False
assert pseudo_estimator.last_pseudo_only is True
assert pseudo_estimator.last_external_received_count == 0
assert pseudo_estimator.last_pseudo_received_count > 0
assert (
    pseudo_estimator.last_external_support_state
    == "pseudo_only"
)

results.append({
    "test": "pseudo_only_support_identified",
    "passed": True,
    "held": held,
    "reason": pseudo_estimator.last_hold_reason,
    "external_support_state":
        pseudo_estimator.last_external_support_state,
})

report = {
    "schema": "paper1.v5.guard.regression.v1",
    "patched_class": class_name,
    "patched_method": method_name,
    "tests": results,
    "tests_passed": len(results),
    "tests_failed": 0,
    "performance_outcomes_inspected": False,
}

test_report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_GUARD_REGRESSION_TESTS_OK")
print("TESTS_PASSED=", len(results))
print("TESTS_FAILED=0")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")

for result in results:
    print(
        result["test"],
        "PASS",
        "held=",
        result.get("held"),
        "reason=",
        result.get("reason"),
    )