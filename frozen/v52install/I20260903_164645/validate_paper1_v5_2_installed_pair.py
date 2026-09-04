from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("PAPER1_V5_2_ROOT", "/workspace"))
LIVE_TRUST = ROOT / "trust_metric.py"
LIVE_TWIN = ROOT / "twin_fed.py"
FEEDER = ROOT / "feeder.npz"
REPORT = (
    ROOT
    / "paper1_v5_2_repair"
    / "v5_2_installed_pair_validation.json"
)

EXPECTED_TRUST_SHA256 = (
    "936dd373a2d8a2f0b905604ca4c3de61"
    "ec2cc889ba233aa150a24f44f2926fe6"
)
EXPECTED_TWIN_SHA256 = (
    "9cd9ffaa32dcfe2f12ed161a8d62d2d9"
    "7b2ab0b4d462fda0e97e7f46572043a4"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"cannot load {path}",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_function(
    tree: ast.AST,
    class_name: str | None,
    function_name: str,
):
    matches = []
    if class_name is None:
        matches = [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
    else:
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matches.extend(
                    item
                    for item in node.body
                    if isinstance(
                        item,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and item.name == function_name
                )
    require(
        len(matches) == 1,
        f"expected one {class_name or '<module>'}.{function_name}",
    )
    return matches[0]


def find_named_dict(function: ast.AST, target_name: str) -> ast.Dict:
    matches = []
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == target_name
            and isinstance(value, ast.Dict)
        ):
            matches.append(value)
    require(
        len(matches) == 1,
        f"expected one dictionary named {target_name}",
    )
    return matches[0]


def dictionary_keys(
    source: str,
    class_name: str | None,
    function_name: str,
    target_name: str,
) -> set[str]:
    tree = ast.parse(source)
    function = find_function(tree, class_name, function_name)
    dictionary = find_named_dict(function, target_name)
    return {
        key.value
        for key in dictionary.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
    }


def dotted_name(node: ast.AST) -> str | None:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


for required_path in (LIVE_TRUST, LIVE_TWIN, FEEDER):
    require(
        required_path.is_file(),
        f"required file is missing: {required_path}",
    )

trust_hash = sha256(LIVE_TRUST)
twin_hash = sha256(LIVE_TWIN)
require(
    trust_hash == EXPECTED_TRUST_SHA256,
    f"installed trust hash mismatch: {trust_hash}",
)
require(
    twin_hash == EXPECTED_TWIN_SHA256,
    f"installed twin hash mismatch: {twin_hash}",
)

trust_source = LIVE_TRUST.read_text(encoding="utf-8-sig")
twin_source = LIVE_TWIN.read_text(encoding="utf-8-sig")
compile(trust_source, str(LIVE_TRUST), "exec")
compile(twin_source, str(LIVE_TWIN), "exec")

trust_tree = ast.parse(trust_source)
twin_tree = ast.parse(twin_source)
trust_estimate_node = find_function(
    trust_tree,
    "TrustMetric",
    "estimate",
)
twin_estimate_node = find_function(
    twin_tree,
    "ProductionTwin",
    "_estimate",
)
trust_estimate_text = (
    ast.get_source_segment(trust_source, trust_estimate_node) or ""
)
twin_estimate_text = (
    ast.get_source_segment(twin_source, twin_estimate_node) or ""
)

solver_calls = {
    name
    for call in ast.walk(trust_estimate_node)
    if isinstance(call, ast.Call)
    for name in [dotted_name(call.func)]
    if name is not None
}

trust_module = load_module("trust_metric", LIVE_TRUST)
twin_module = load_module("twin_fed_v5_2_installed", LIVE_TWIN)

with np.load(FEEDER, allow_pickle=False) as feeder:
    h = np.asarray(feeder["H"], dtype=float)
    sigma2 = np.asarray(feeder["sigma2"], dtype=float)
    q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])

metric = trust_module.TrustMetric(
    h,
    sigma2,
    q,
    trust_module.MetricConfig(
        n_telemetry=n_telemetry,
        ridge=0.0,
    ),
)

n = h.shape[1]
m = h.shape[0]
rng = np.random.default_rng(52004)
tests: list[dict] = []


def check(name: str, condition: bool, detail: dict) -> None:
    require(condition, f"{name} failed: {detail}")
    tests.append({"name": name, "status": "PASS", **detail})
    print(name, "PASS")


check(
    "direct_svd_without_normal_equations",
    "np.linalg.lstsq" in solver_calls
    and "np.linalg.solve" not in solver_calls
    and "Hr.T @ (Hr" not in trust_estimate_text,
    {"solver_calls": sorted(solver_calls)},
)

q_eigenvalues = np.linalg.eigvalsh((q + q.T) / 2.0)
check(
    "process_covariance_positive_definite",
    bool(
        np.all(np.isfinite(q_eigenvalues))
        and float(q_eigenvalues[0]) > 0.0
        and metric.process_cholesky.shape == (n, n)
    ),
    {
        "q_eigenvalue_min": float(q_eigenvalues[0]),
        "process_rms_increment": float(metric.process_rms_increment),
    },
)


def make_twin(initial_state: np.ndarray, active_metric=metric):
    twin = twin_module.ProductionTwin.__new__(
        twin_module.ProductionTwin
    )
    twin.metric = active_metric
    twin.x_previous = np.asarray(
        initial_state,
        dtype=float,
    ).copy()
    twin.n = n
    twin.n_telemetry = n_telemetry
    twin.hold_factor = 50.0
    return twin


all_rx = np.ones(m, dtype=bool)
pseudo_rx = np.ones(m, dtype=bool)
pseudo_rx[:n_telemetry] = False
gamma = np.ones(m, dtype=float)
state = rng.normal(scale=0.02, size=n)
exact_z = h @ state
twin = make_twin(np.zeros(n, dtype=float))

estimate, held, reliable = twin._estimate(
    exact_z,
    pseudo_rx,
    gamma,
)
check(
    "pseudo_only_bootstrap_rejected",
    held
    and not reliable
    and twin.last_hold_reason == "solve_inexact"
    and not hasattr(twin, "has_valid_estimate")
    and twin.last_measurement_rank == 489,
    {
        "hold_reason": twin.last_hold_reason,
        "measurement_rank": twin.last_measurement_rank,
    },
)

estimate, held, reliable = twin._estimate(
    exact_z,
    all_rx,
    gamma,
)
bootstrap_error = float(np.linalg.norm(estimate - state))
check(
    "full_design_bootstrap_exact_recovery",
    not held
    and reliable
    and twin.last_hold_reason == "bootstrap_accept"
    and twin.has_valid_estimate
    and bootstrap_error < 1e-8
    and twin.last_measurement_rank == n,
    {
        "bootstrap_error": bootstrap_error,
        "measurement_rank": twin.last_measurement_rank,
    },
)

state_before = twin.x_previous.copy()
estimate, held, reliable = twin._estimate(
    h @ state_before,
    pseudo_rx,
    gamma,
)
check(
    "pseudo_only_postbootstrap_explicit_hold",
    held
    and reliable
    and twin.last_hold_reason == "prior_only_hold"
    and np.linalg.norm(estimate - state_before) < 1e-12
    and twin.last_measurement_rank == 489
    and twin.last_posterior_rank == n,
    {
        "hold_reason": twin.last_hold_reason,
        "measurement_rank": twin.last_measurement_rank,
        "posterior_rank": twin.last_posterior_rank,
    },
)

process_coordinates = rng.normal(size=n)
process_coordinates *= 0.5 / np.linalg.norm(process_coordinates)
truth = state_before + metric.process_cholesky @ process_coordinates
mixed_z = h @ state_before
mixed_z[:n_telemetry] = (h @ truth)[:n_telemetry]
estimate, held, reliable = twin._estimate(
    mixed_z,
    all_rx,
    gamma,
)
check(
    "small_network_supported_q_update_accepted",
    not held
    and reliable
    and twin.last_hold_reason == "accepted"
    and twin.last_process_prior_active
    and twin.last_process_increment_mahalanobis
    < twin.last_process_guard_limit,
    {
        "process_increment_mahalanobis": (
            twin.last_process_increment_mahalanobis
        ),
        "process_guard_limit": twin.last_process_guard_limit,
    },
)


class GuardStub:
    process_rms_increment = metric.process_rms_increment
    last_process_prior_active = True
    last_process_increment_norm = 0.1
    last_process_increment_mahalanobis = 51.0
    last_estimator_solver = "q_prior_augmented_lstsq_svd"
    last_estimator_mode = "q_prior_innovation"
    last_estimator_rcond = math.sqrt(np.finfo(float).eps)
    last_estimator_effective_rows = m
    last_estimator_rank = n
    last_estimator_condition = 2.0
    last_estimator_singular_max = 2.0
    last_estimator_singular_min = 1.0
    last_estimator_residual_norm = 0.0
    last_estimator_augmented_residual_norm = 51.0
    last_measurement_rank = n
    last_measurement_condition = 2.0
    last_measurement_singular_max = 2.0
    last_measurement_singular_min = 1.0
    last_measurement_full_rank = True
    last_posterior_rank = n
    last_posterior_condition = 2.0
    last_posterior_singular_max = 2.0
    last_posterior_singular_min = 1.0
    last_posterior_reliable = True

    def estimate(self, z, rx, gamma, *, prior_state=None):
        require(prior_state is not None, "guard test requires prior")
        direction = np.zeros(n, dtype=float)
        direction[0] = 0.1
        return np.asarray(prior_state, dtype=float) + direction, True


guarded_twin = make_twin(state_before, GuardStub())
guarded_twin.has_valid_estimate = True
estimate, held, reliable = guarded_twin._estimate(
    exact_z,
    all_rx,
    gamma,
)
check(
    "catastrophic_q_coordinate_update_rejected",
    held
    and reliable
    and guarded_twin.last_hold_reason == "process_guard"
    and np.array_equal(estimate, state_before),
    {
        "process_guard_statistic": (
            guarded_twin.last_process_guard_statistic
        ),
        "process_guard_limit": guarded_twin.last_process_guard_limit,
    },
)

step_keys = dictionary_keys(
    twin_source,
    "ProductionTwin",
    "update",
    "row",
)
event_keys = dictionary_keys(
    twin_source,
    None,
    "aggregate_events",
    "row",
)
publication_keys = dictionary_keys(
    twin_source,
    None,
    "publish_score",
    "payload",
)

required_step = {
    "estimator_mode",
    "measurement_rank",
    "measurement_condition",
    "measurement_full_rank",
    "posterior_rank",
    "posterior_condition",
    "posterior_reliable",
    "process_prior_active",
    "process_rms_increment",
    "process_increment_norm",
    "process_increment_mahalanobis",
    "process_guard_statistic",
    "process_guard_limit",
    "process_guard_policy",
    "prior_only_hold_step",
    "process_diagnostic_invalid_hold_step",
    "process_guard_hold_step",
}
required_event = {
    "estimator_mode_first",
    "estimator_mode_last",
    "measurement_rank_min",
    "measurement_full_rank_fraction",
    "posterior_rank_min",
    "posterior_reliable_fraction",
    "process_prior_active_fraction",
    "process_rms_increment",
    "process_increment_mahalanobis_max",
    "process_guard_limit",
    "process_guard_policy",
    "prior_only_hold_fraction",
    "process_diagnostic_invalid_hold_fraction",
    "process_guard_hold_fraction",
}

check(
    "step_schema_complete",
    required_step <= step_keys,
    {"verified_fields": len(required_step)},
)
check(
    "event_schema_complete",
    required_event <= event_keys,
    {"verified_fields": len(required_event)},
)
check(
    "publication_schema_complete",
    required_step <= publication_keys,
    {"verified_fields": len(required_step)},
)

publication_node = find_function(
    twin_tree,
    None,
    "publish_score",
)
publication_text = (
    ast.get_source_segment(twin_source, publication_node) or ""
)
check(
    "publication_fields_read_from_step_row",
    "self.last_" not in publication_text
    and all(
        f'row["{field}"]' in publication_text
        for field in required_step
    ),
    {"verified_fields": len(required_step)},
)

check(
    "prior_and_q_process_guard_wiring_present",
    "prior_state=prior_state" in twin_estimate_text
    and (
        "process_increment_mahalanobis > process_guard_limit"
        in twin_estimate_text
    )
    and "math.sqrt(self.n) * omega" not in twin_estimate_text
    and "jump_norm > jump_limit" not in twin_estimate_text,
    {"guard_policy": "q_process_mahalanobis"},
)

require(
    sha256(LIVE_TRUST) == trust_hash,
    "installed trust changed during validation",
)
require(
    sha256(LIVE_TWIN) == twin_hash,
    "installed twin changed during validation",
)

report = {
    "schema": "paper1.v5_2.installed_pair_validation.v1",
    "installed_trust_sha256": trust_hash,
    "installed_twin_sha256": twin_hash,
    "estimator_mode": (
        "q_prior_innovation_after_measurement_bootstrap"
    ),
    "process_guard_policy": "q_process_mahalanobis",
    "legacy_euclidean_jump_guard_active": False,
    "tests_passed": len(tests),
    "tests_failed": 0,
    "tests": tests,
    "simulation_rerun": False,
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_2_INSTALLED_PAIR_VALIDATION_OK")
print("TESTS_PASSED=", len(tests))
print("TESTS_FAILED=0")
print("INSTALLED_TRUST_SHA256=", trust_hash)
print("INSTALLED_TWIN_SHA256=", twin_hash)
print("REPORT_SHA256=", sha256(REPORT))
print("ESTIMATOR_MODE=q_prior_innovation_after_measurement_bootstrap")
print("PROCESS_GUARD_POLICY=q_process_mahalanobis")
print("LEGACY_EUCLIDEAN_JUMP_GUARD_ACTIVE=False")
print("SIMULATION_RERUN=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_2_INSTALLED_PAIR_READY_FOR_MECHANICAL_CELL_GATE")
