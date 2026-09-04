from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path("/workspace")
WORKSPACE = ROOT / "paper1_v5_1_solver_repair_workspace"
TRUST = WORKSPACE / "trust_metric.v5_1.candidate.py"
TWIN = WORKSPACE / "twin_fed.v5_1.candidate.py"

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"Could not load {path}",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

sys.path.insert(0, str(ROOT))

trust_module = load_module(TRUST, "trust_metric")
twin_module = load_module(TWIN, "paper1_v5_1_twin_candidate")

n = 491
m = 583
n_telemetry = 45
omega = 0.02
hold_factor = 50.0
expected_limit = hold_factor * math.sqrt(n) * omega

class StubMetric:
    def __init__(self, candidate, reliable):
        self.candidate = np.asarray(candidate, dtype=float)
        self.reliable = bool(reliable)
        self.cfg = SimpleNamespace(omega=omega)

    def estimate(self, z, rx, gamma):
        self.last_estimator_solver = "weighted_lstsq_svd"
        self.last_estimator_rcond = float(
            np.sqrt(np.finfo(float).eps)
        )
        self.last_estimator_effective_rows = int(
            np.count_nonzero(rx)
        )
        self.last_estimator_rank = n if self.reliable else n - 1
        self.last_estimator_condition = (
            1000.0 if self.reliable else float("inf")
        )
        self.last_estimator_singular_max = 10.0
        self.last_estimator_singular_min = (
            0.01 if self.reliable else 0.0
        )
        self.last_estimator_residual_norm = 0.5
        return self.candidate.copy(), self.reliable

def make_twin(previous, candidate, reliable, valid):
    twin = object.__new__(twin_module.ProductionTwin)
    twin.n = n
    twin.n_telemetry = n_telemetry
    twin.hold_factor = hold_factor
    twin.x_previous = np.asarray(previous, dtype=float).copy()
    twin.has_valid_estimate = bool(valid)
    twin.metric = StubMetric(candidate, reliable)
    return twin

z = np.zeros(m)
rx = np.ones(m, dtype=bool)
gamma = np.ones(m)

tests = []

bootstrap = np.full(n, 1206.0 / math.sqrt(n))
twin = make_twin(np.zeros(n), bootstrap, True, False)
estimate, held, reliable = twin._estimate(z, rx, gamma)
tests.append(("reliable_bootstrap_accept", not held and reliable))
tests.append((
    "fixed_jump_limit_value",
    math.isclose(
        twin.last_jump_limit,
        expected_limit,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ),
))
tests.append((
    "fixed_guard_policy_logged",
    twin.last_jump_guard_policy == "fixed_model_increment",
))

previous = np.full(n, 30.8 / math.sqrt(n))
small_candidate = previous.copy()
small_candidate[0] += expected_limit * 0.25
twin = make_twin(previous, small_candidate, True, True)
_, held, reliable = twin._estimate(z, rx, gamma)
tests.append(("small_increment_accepted", not held and reliable))

large_candidate = previous.copy()
large_candidate[0] += expected_limit * 1.10
twin = make_twin(previous, large_candidate, True, True)
_, held, reliable = twin._estimate(z, rx, gamma)
tests.append((
    "large_increment_rejected",
    held
    and reliable
    and twin.last_hold_reason == "jump_guard",
))

huge_previous = np.full(n, 1.0e6)
huge_candidate = huge_previous.copy()
huge_candidate[0] += expected_limit * 1.10
twin = make_twin(
    huge_previous,
    huge_candidate,
    True,
    True,
)
_, held, _ = twin._estimate(z, rx, gamma)
tests.append((
    "guard_does_not_expand_with_previous_norm",
    held
    and math.isclose(
        twin.last_jump_limit,
        expected_limit,
        rel_tol=1e-12,
    ),
))

twin = make_twin(previous, previous, False, True)
_, held, reliable = twin._estimate(z, rx, gamma)
tests.append((
    "unreliable_solver_rejected",
    held
    and not reliable
    and twin.last_hold_reason == "solve_inexact",
))

nonfinite = previous.copy()
nonfinite[0] = np.nan
twin = make_twin(previous, nonfinite, True, True)
_, held, reliable = twin._estimate(z, rx, gamma)
tests.append((
    "nonfinite_candidate_rejected",
    held
    and reliable
    and twin.last_hold_reason == "nonfinite_candidate",
))

tests.append((
    "solver_name_logged",
    twin.last_estimator_solver == "weighted_lstsq_svd",
))
tests.append((
    "solver_rank_logged",
    twin.last_estimator_rank == n,
))

source = TWIN.read_text(encoding="utf-8-sig")
tree = ast.parse(source)

def named_function(function_name):
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
    ]
    require(
        len(matches) == 1,
        f"Expected one {function_name}; found {len(matches)}",
    )
    return matches[0]

def named_method(class_name, method_name):
    matches = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            matches.extend(
                child for child in node.body
                if isinstance(child, ast.FunctionDef)
                and child.name == method_name
            )
    require(
        len(matches) == 1,
        f"Expected one {class_name}.{method_name}; found {len(matches)}",
    )
    return matches[0]

def dict_keys(function_node, variable_name):
    matches = []
    for node in ast.walk(function_node):
        target = None
        value = None

        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value

        if (
            isinstance(target, ast.Name)
            and target.id == variable_name
            and isinstance(value, ast.Dict)
        ):
            matches.append(value)

    require(
        len(matches) == 1,
        f"Expected one {variable_name} dictionary; found {len(matches)}",
    )

    result = set()
    for key in matches[0].keys:
        require(
            isinstance(key, ast.Constant)
            and isinstance(key.value, str),
            f"Nonliteral key in {variable_name}",
        )
        result.add(key.value)
    return result

step_keys = dict_keys(
    named_method("ProductionTwin", "update"),
    "row",
)
publication_keys = dict_keys(
    named_function("publish_score"),
    "payload",
)
event_keys = dict_keys(
    named_function("aggregate_events"),
    "row",
)

required_step_fields = {
    "estimator_reliable",
    "estimator_solver",
    "estimator_rcond",
    "estimator_effective_rows",
    "estimator_rank",
    "estimator_condition",
    "estimator_singular_max",
    "estimator_singular_min",
    "estimator_residual_norm",
    "jump_guard_policy",
    "model_increment_scale",
}

required_event_fields = {
    "estimator_reliable_fraction",
    "estimator_reliable_all",
    "estimator_solver",
    "estimator_rcond",
    "estimator_effective_rows_min",
    "estimator_effective_rows_max",
    "estimator_rank_min",
    "estimator_condition_max",
    "estimator_singular_max_max",
    "estimator_singular_min_min",
    "estimator_residual_norm_max",
    "jump_guard_policy",
    "model_increment_scale",
}

tests.append((
    "step_schema_complete",
    required_step_fields <= step_keys,
))
tests.append((
    "publication_schema_complete",
    required_step_fields <= publication_keys,
))
tests.append((
    "event_schema_complete",
    required_event_fields <= event_keys,
))

event_rows = []

for index in range(2):
    row = {
        "event_id": 0,
        "pattern_id": 0,
        "arm": "mechanical",
        "regime": "mechanical",
        "stratum": 0,
        "bandwidth_level": "bw04_oracle",
        "bandwidth_cap_bps": 1.0e12,
        "T": 0.9 - index * 0.1,
        "alarm": False,
        "alarm_delta_lmax": False,
        "b1": 1.0,
        "n_rx": 583,
        "n_rx_telemetry": 45,
        "held": bool(index),
        "solve_exact": not bool(index),
        "estimator_reliable": not bool(index),
        "estimator_solver": "weighted_lstsq_svd",
        "estimator_rcond": np.sqrt(np.finfo(float).eps),
        "estimator_effective_rows": 550 - index * 50,
        "estimator_rank": 491 - index,
        "estimator_condition": 10.0 + index * 10.0,
        "estimator_singular_max": 5.0 + index,
        "estimator_singular_min": 0.2 - index * 0.1,
        "estimator_residual_norm": 1.0 + index,
        "state_update_accepted_step": not bool(index),
        "bootstrap_accept_step": index == 0,
        "solve_inexact_hold_step": index == 1,
        "nonfinite_candidate_hold_step": False,
        "jump_guard_hold_step": False,
        "pseudo_only_step": False,
        "external_support_present_step": True,
        "no_received_measurements_step": False,
        "candidate_norm": 30.8,
        "previous_norm": 30.7,
        "jump_norm": 0.1,
        "jump_limit": expected_limit,
        "jump_guard_policy": "fixed_model_increment",
        "model_increment_scale": math.sqrt(n) * omega,
        "external_received_count": 45,
        "external_total": 45,
        "external_support_fraction": 1.0,
        "pseudo_received_count": 538,
        "residual_available": True,
    }

    for column in twin_module.MAX_EVENT_COLUMNS:
        row.setdefault(column, float(index + 1))

    event_rows.append(row)

events = twin_module.aggregate_events(pd.DataFrame(event_rows))
require(len(events) == 1, "Event aggregation returned wrong row count.")

event = events.iloc[0]

tests.append((
    "event_reliable_fraction_correct",
    math.isclose(
        float(event["estimator_reliable_fraction"]),
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ),
))
tests.append((
    "event_rank_min_correct",
    int(event["estimator_rank_min"]) == 490,
))
tests.append((
    "event_condition_max_correct",
    math.isclose(
        float(event["estimator_condition_max"]),
        20.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ),
))
tests.append((
    "event_guard_policy_correct",
    event["jump_guard_policy"] == "fixed_model_increment",
))

failed = [name for name, passed in tests if not passed]

for name, passed in tests:
    print(name, "PASS" if passed else "FAIL")

if failed:
    raise RuntimeError(f"V5.1 twin tests failed: {failed}")

print("PAPER1_V5_1_TWIN_GUARD_LOGGING_TESTS_OK")
print("TESTS_PASSED=", len(tests))
print("TESTS_FAILED=0")
print("FIXED_JUMP_LIMIT=", expected_limit)
print("JUMP_GUARD_POLICY=fixed_model_increment")
print("STEP_FIELDS_VERIFIED=", len(required_step_fields))
print("EVENT_FIELDS_VERIFIED=", len(required_event_fields))
print("LIVE_FILES_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")