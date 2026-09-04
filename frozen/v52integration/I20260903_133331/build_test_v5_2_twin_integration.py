from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(os.environ.get("PAPER1_V5_2_ROOT", "/workspace"))
WORKSPACE = ROOT / "paper1_v5_2_repair"
LIVE_TRUST = ROOT / "trust_metric.py"
LIVE_TWIN = ROOT / "twin_fed.py"
TRUST_CANDIDATE = WORKSPACE / "trust_metric_v5_2_candidate.py"
TWIN_CANDIDATE = WORKSPACE / "twin_fed_v5_2_candidate.py"
REPORT_PATH = WORKSPACE / "v5_2_twin_integration_tests.json"

EXPECTED_LIVE_TRUST_SHA256 = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)
EXPECTED_LIVE_TWIN_SHA256 = (
    "39e6729af233032ab9c58851c968225"
    "2f02d36eed739eb2ec769e165659da34c"
)
EXPECTED_TRUST_CANDIDATE_SHA256 = (
    "936dd373a2d8a2f0b905604ca4c3de61"
    "ec2cc889ba233aa150a24f44f2926fe6"
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


def find_function(tree: ast.AST, class_name: str | None, function_name: str):
    if class_name is None:
        matches = [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
    else:
        matches = []
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matches.extend(
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == function_name
                )
    require(
        len(matches) == 1,
        f"expected one {class_name or '<module>'}.{function_name}",
    )
    return matches[0]


def replace_method(source: str, class_name: str, method_name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = find_function(tree, class_name, method_name)
    require(node.end_lineno is not None, "AST method end is unavailable")
    lines = source.splitlines(keepends=True)
    return "".join(
        lines[: node.lineno - 1]
        + [replacement.rstrip() + "\n"]
        + lines[node.end_lineno :]
    )


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
        if isinstance(target, ast.Name) and target.id == target_name and isinstance(value, ast.Dict):
            matches.append(value)
    require(len(matches) == 1, f"expected one dictionary named {target_name}")
    return matches[0]


def insert_dict_entries(
    source: str,
    *,
    class_name: str | None,
    function_name: str,
    target_name: str,
    after_key: str,
    entries: list[str],
) -> str:
    tree = ast.parse(source)
    function = find_function(tree, class_name, function_name)
    dictionary = find_named_dict(function, target_name)
    matches = []
    for key, value in zip(dictionary.keys, dictionary.values):
        if isinstance(key, ast.Constant) and key.value == after_key:
            matches.append((key, value))
    require(len(matches) == 1, f"dictionary key is not unique: {after_key}")
    key, value = matches[0]
    require(value.end_lineno is not None, f"no end line for {after_key}")
    lines = source.splitlines(keepends=True)
    key_line = lines[key.lineno - 1]
    indentation = key_line[: len(key_line) - len(key_line.lstrip())]
    insertion = "".join(indentation + entry + "\n" for entry in entries)
    return "".join(lines[: value.end_lineno] + [insertion] + lines[value.end_lineno :])


ESTIMATE_METHOD = r'''
    def _estimate(
        self,
        z: np.ndarray,
        rx: np.ndarray,
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        had_valid_estimate = bool(
            getattr(self, "has_valid_estimate", False)
        )
        prior_state = self.x_previous if had_valid_estimate else None
        candidate, solved_reliably = self.metric.estimate(
            z,
            rx,
            gamma,
            prior_state=prior_state,
        )
        candidate_array = np.asarray(candidate, dtype=float).reshape(-1)
        candidate_finite = bool(
            candidate_array.shape == (self.n,)
            and np.all(np.isfinite(candidate_array))
        )
        previous_norm = float(np.linalg.norm(self.x_previous))
        candidate_norm = (
            float(np.linalg.norm(candidate_array))
            if candidate_finite
            else float("nan")
        )
        jump_norm = (
            float(np.linalg.norm(candidate_array - self.x_previous))
            if had_valid_estimate and candidate_finite
            else float("nan")
        )

        process_rms_increment = float(
            getattr(self.metric, "process_rms_increment", float("nan"))
        )
        if (
            not math.isfinite(process_rms_increment)
            or process_rms_increment <= 0.0
        ):
            raise RuntimeError("metric process RMS increment is invalid")
        process_prior_active = bool(
            getattr(self.metric, "last_process_prior_active", False)
        )
        process_increment_norm = float(
            getattr(self.metric, "last_process_increment_norm", float("nan"))
        )
        process_increment_mahalanobis = float(
            getattr(
                self.metric,
                "last_process_increment_mahalanobis",
                float("nan"),
            )
        )
        process_guard_limit = float(self.hold_factor)
        if not math.isfinite(process_guard_limit) or process_guard_limit <= 1.0:
            raise RuntimeError("process guard limit must be finite and exceed one")

        rx_array = np.asarray(rx, dtype=bool).reshape(-1)
        configured_telemetry = int(getattr(self, "n_telemetry", 0))
        external_total = min(configured_telemetry, len(rx_array))
        external_received = int(
            np.count_nonzero(rx_array[:external_total])
        )
        pseudo_received = int(
            np.count_nonzero(rx_array[external_total:])
        )
        if external_received > 0:
            external_support_state = "external_present"
        elif pseudo_received > 0:
            external_support_state = "pseudo_only"
        else:
            external_support_state = "no_received_measurements"

        if not solved_reliably:
            decision_reason = "solve_inexact"
            held = True
        elif not candidate_finite:
            decision_reason = "nonfinite_candidate"
            held = True
        elif not had_valid_estimate:
            decision_reason = "bootstrap_accept"
            held = False
        elif external_received == 0:
            decision_reason = "prior_only_hold"
            held = True
        elif (
            not process_prior_active
            or not math.isfinite(process_increment_norm)
            or not math.isfinite(process_increment_mahalanobis)
        ):
            decision_reason = "process_diagnostic_invalid"
            held = True
        elif process_increment_mahalanobis > process_guard_limit:
            decision_reason = "process_guard"
            held = True
        else:
            decision_reason = "accepted"
            held = False

        self.last_hold_reason = decision_reason
        self.last_candidate_norm = candidate_norm
        self.last_previous_norm = previous_norm
        self.last_jump_norm = jump_norm
        self.last_jump_limit = float("nan")
        self.last_jump_guard_policy = "q_process_mahalanobis"
        self.last_model_increment_scale = process_rms_increment
        self.last_candidate_finite = candidate_finite
        self.last_solved_exactly = bool(solved_reliably)
        self.last_estimator_reliable = bool(solved_reliably)
        self.last_estimator_solver = str(
            getattr(self.metric, "last_estimator_solver", "unavailable")
        )
        self.last_estimator_mode = str(
            getattr(self.metric, "last_estimator_mode", "unavailable")
        )
        self.last_estimator_rcond = float(
            getattr(self.metric, "last_estimator_rcond", float("nan"))
        )
        self.last_estimator_effective_rows = int(
            getattr(self.metric, "last_estimator_effective_rows", 0)
        )
        self.last_estimator_rank = int(
            getattr(self.metric, "last_estimator_rank", 0)
        )
        self.last_estimator_condition = float(
            getattr(self.metric, "last_estimator_condition", float("inf"))
        )
        self.last_estimator_singular_max = float(
            getattr(self.metric, "last_estimator_singular_max", float("nan"))
        )
        self.last_estimator_singular_min = float(
            getattr(self.metric, "last_estimator_singular_min", float("nan"))
        )
        self.last_estimator_residual_norm = float(
            getattr(self.metric, "last_estimator_residual_norm", float("nan"))
        )
        self.last_estimator_augmented_residual_norm = float(
            getattr(
                self.metric,
                "last_estimator_augmented_residual_norm",
                float("nan"),
            )
        )
        self.last_measurement_rank = int(
            getattr(self.metric, "last_measurement_rank", 0)
        )
        self.last_measurement_condition = float(
            getattr(self.metric, "last_measurement_condition", float("inf"))
        )
        self.last_measurement_singular_max = float(
            getattr(self.metric, "last_measurement_singular_max", float("nan"))
        )
        self.last_measurement_singular_min = float(
            getattr(self.metric, "last_measurement_singular_min", float("nan"))
        )
        self.last_measurement_full_rank = bool(
            getattr(self.metric, "last_measurement_full_rank", False)
        )
        self.last_posterior_rank = int(
            getattr(self.metric, "last_posterior_rank", 0)
        )
        self.last_posterior_condition = float(
            getattr(self.metric, "last_posterior_condition", float("inf"))
        )
        self.last_posterior_singular_max = float(
            getattr(self.metric, "last_posterior_singular_max", float("nan"))
        )
        self.last_posterior_singular_min = float(
            getattr(self.metric, "last_posterior_singular_min", float("nan"))
        )
        self.last_posterior_reliable = bool(
            getattr(self.metric, "last_posterior_reliable", False)
        )
        self.last_process_prior_active = process_prior_active
        self.last_process_rms_increment = process_rms_increment
        self.last_process_increment_norm = process_increment_norm
        self.last_process_increment_mahalanobis = process_increment_mahalanobis
        self.last_process_guard_statistic = process_increment_mahalanobis
        self.last_process_guard_limit = process_guard_limit
        self.last_process_guard_policy = "q_process_mahalanobis"
        self.last_external_received_count = external_received
        self.last_external_total = external_total
        self.last_pseudo_received_count = pseudo_received
        self.last_external_support_state = external_support_state
        self.last_pseudo_only = external_support_state == "pseudo_only"
        self.last_external_support_fraction = (
            float(external_received / external_total)
            if external_total > 0
            else 0.0
        )

        if held:
            return self.x_previous.copy(), True, solved_reliably
        self.x_previous = candidate_array.copy()
        self.has_valid_estimate = True
        return self.x_previous.copy(), False, solved_reliably
'''.strip("\n")


STEP_ESTIMATOR_ENTRIES = [
    '"estimator_augmented_residual_norm": self.last_estimator_augmented_residual_norm,',
    '"estimator_mode": self.last_estimator_mode,',
    '"measurement_rank": self.last_measurement_rank,',
    '"measurement_condition": self.last_measurement_condition,',
    '"measurement_singular_max": self.last_measurement_singular_max,',
    '"measurement_singular_min": self.last_measurement_singular_min,',
    '"measurement_full_rank": self.last_measurement_full_rank,',
    '"posterior_rank": self.last_posterior_rank,',
    '"posterior_condition": self.last_posterior_condition,',
    '"posterior_singular_max": self.last_posterior_singular_max,',
    '"posterior_singular_min": self.last_posterior_singular_min,',
    '"posterior_reliable": self.last_posterior_reliable,',
]

STEP_HOLD_ENTRIES = [
    '"prior_only_hold_step": self.last_hold_reason == "prior_only_hold",',
    '"process_diagnostic_invalid_hold_step": self.last_hold_reason == "process_diagnostic_invalid",',
    '"process_guard_hold_step": self.last_hold_reason == "process_guard",',
]

STEP_PROCESS_ENTRIES = [
    '"process_prior_active": self.last_process_prior_active,',
    '"process_rms_increment": self.last_process_rms_increment,',
    '"process_increment_norm": self.last_process_increment_norm,',
    '"process_increment_mahalanobis": self.last_process_increment_mahalanobis,',
    '"process_guard_statistic": self.last_process_guard_statistic,',
    '"process_guard_limit": self.last_process_guard_limit,',
    '"process_guard_policy": self.last_process_guard_policy,',
]

PUBLICATION_ESTIMATOR_ENTRIES = [
    '"estimator_augmented_residual_norm": row["estimator_augmented_residual_norm"],',
    '"estimator_mode": row["estimator_mode"],',
    '"measurement_rank": row["measurement_rank"],',
    '"measurement_condition": row["measurement_condition"],',
    '"measurement_singular_max": row["measurement_singular_max"],',
    '"measurement_singular_min": row["measurement_singular_min"],',
    '"measurement_full_rank": row["measurement_full_rank"],',
    '"posterior_rank": row["posterior_rank"],',
    '"posterior_condition": row["posterior_condition"],',
    '"posterior_singular_max": row["posterior_singular_max"],',
    '"posterior_singular_min": row["posterior_singular_min"],',
    '"posterior_reliable": row["posterior_reliable"],',
]

PUBLICATION_HOLD_ENTRIES = [
    '"prior_only_hold_step": row["prior_only_hold_step"],',
    '"process_diagnostic_invalid_hold_step": row["process_diagnostic_invalid_hold_step"],',
    '"process_guard_hold_step": row["process_guard_hold_step"],',
]

PUBLICATION_PROCESS_ENTRIES = [
    '"process_prior_active": row["process_prior_active"],',
    '"process_rms_increment": row["process_rms_increment"],',
    '"process_increment_norm": row["process_increment_norm"],',
    '"process_increment_mahalanobis": row["process_increment_mahalanobis"],',
    '"process_guard_statistic": row["process_guard_statistic"],',
    '"process_guard_limit": row["process_guard_limit"],',
    '"process_guard_policy": row["process_guard_policy"],',
]

EVENT_ESTIMATOR_ENTRIES = [
    '"estimator_augmented_residual_norm_max": float(pd.to_numeric(block["estimator_augmented_residual_norm"], errors="coerce").max()),',
    '"estimator_mode_first": str(first["estimator_mode"]),',
    '"estimator_mode_last": str(block["estimator_mode"].iloc[-1]),',
    '"measurement_rank_min": int(block["measurement_rank"].min()),',
    '"measurement_condition_max": float(pd.to_numeric(block["measurement_condition"], errors="coerce").max()),',
    '"measurement_full_rank_fraction": float(block["measurement_full_rank"].mean()),',
    '"posterior_rank_min": int(block["posterior_rank"].min()),',
    '"posterior_condition_max": float(pd.to_numeric(block["posterior_condition"], errors="coerce").max()),',
    '"posterior_reliable_fraction": float(block["posterior_reliable"].mean()),',
]

EVENT_HOLD_ENTRIES = [
    '"prior_only_hold_fraction": float(block["prior_only_hold_step"].mean()),',
    '"process_diagnostic_invalid_hold_fraction": float(block["process_diagnostic_invalid_hold_step"].mean()),',
    '"process_guard_hold_fraction": float(block["process_guard_hold_step"].mean()),',
]

EVENT_PROCESS_ENTRIES = [
    '"process_prior_active_fraction": float(block["process_prior_active"].mean()),',
    '"process_rms_increment": float(first["process_rms_increment"]),',
    '"process_increment_norm_max": float(pd.to_numeric(block["process_increment_norm"], errors="coerce").max()),',
    '"process_increment_mahalanobis_max": float(pd.to_numeric(block["process_increment_mahalanobis"], errors="coerce").max()),',
    '"process_guard_statistic_max": float(pd.to_numeric(block["process_guard_statistic"], errors="coerce").max()),',
    '"process_guard_limit": float(first["process_guard_limit"]),',
    '"process_guard_policy": str(first["process_guard_policy"]),',
]


def build_twin_candidate() -> str:
    require(sha256(LIVE_TWIN) == EXPECTED_LIVE_TWIN_SHA256, "live twin hash mismatch")
    source = LIVE_TWIN.read_text(encoding="utf-8-sig")
    source = replace_method(source, "ProductionTwin", "_estimate", ESTIMATE_METHOD)
    source = insert_dict_entries(
        source,
        class_name="ProductionTwin",
        function_name="update",
        target_name="row",
        after_key="estimator_residual_norm",
        entries=STEP_ESTIMATOR_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name="ProductionTwin",
        function_name="update",
        target_name="row",
        after_key="jump_guard_hold_step",
        entries=STEP_HOLD_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name="ProductionTwin",
        function_name="update",
        target_name="row",
        after_key="model_increment_scale",
        entries=STEP_PROCESS_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="aggregate_events",
        target_name="row",
        after_key="estimator_residual_norm_max",
        entries=EVENT_ESTIMATOR_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="aggregate_events",
        target_name="row",
        after_key="jump_guard_hold_fraction",
        entries=EVENT_HOLD_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="aggregate_events",
        target_name="row",
        after_key="model_increment_scale",
        entries=EVENT_PROCESS_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="publish_score",
        target_name="payload",
        after_key="estimator_residual_norm",
        entries=PUBLICATION_ESTIMATOR_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="publish_score",
        target_name="payload",
        after_key="jump_guard_hold_step",
        entries=PUBLICATION_HOLD_ENTRIES,
    )
    source = insert_dict_entries(
        source,
        class_name=None,
        function_name="publish_score",
        target_name="payload",
        after_key="model_increment_scale",
        entries=PUBLICATION_PROCESS_ENTRIES,
    )
    compile(source, str(TWIN_CANDIDATE), "exec")
    TWIN_CANDIDATE.write_text(source, encoding="utf-8", newline="\n")
    return sha256(TWIN_CANDIDATE)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dictionary_keys(source: str, class_name: str | None, function_name: str, target: str):
    tree = ast.parse(source)
    function = find_function(tree, class_name, function_name)
    dictionary = find_named_dict(function, target)
    return {
        key.value
        for key in dictionary.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def run_tests(trust_module, twin_module) -> list[dict]:
    with np.load(ROOT / "feeder.npz", allow_pickle=False) as feeder:
        h = np.asarray(feeder["H"], dtype=float)
        sigma2 = np.asarray(feeder["sigma2"], dtype=float)
        q = np.asarray(feeder["Q"], dtype=float)
        n_telemetry = int(feeder["n_telemetry"])

    metric = trust_module.TrustMetric(
        h,
        sigma2,
        q,
        trust_module.MetricConfig(n_telemetry=n_telemetry, ridge=0.0),
    )
    n = h.shape[1]
    m = h.shape[0]
    rng = np.random.default_rng(52003)
    tests = []

    def check(name: str, condition: bool, detail: dict) -> None:
        require(condition, f"{name} failed: {detail}")
        tests.append({"name": name, "status": "PASS", **detail})
        print(name, "PASS")

    def make_twin(initial_state: np.ndarray, active_metric=metric):
        twin = twin_module.ProductionTwin.__new__(twin_module.ProductionTwin)
        twin.metric = active_metric
        twin.x_previous = np.asarray(initial_state, dtype=float).copy()
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

    estimate, held, reliable = twin._estimate(exact_z, pseudo_rx, gamma)
    check(
        "pseudo_only_bootstrap_rejected",
        held
        and not reliable
        and twin.last_hold_reason == "solve_inexact"
        and not hasattr(twin, "has_valid_estimate"),
        {"hold_reason": twin.last_hold_reason},
    )

    estimate, held, reliable = twin._estimate(exact_z, all_rx, gamma)
    bootstrap_error = float(np.linalg.norm(estimate - state))
    check(
        "first_measurement_full_rank_bootstrap_accepted",
        not held
        and reliable
        and twin.last_hold_reason == "bootstrap_accept"
        and twin.has_valid_estimate
        and bootstrap_error < 1e-8,
        {"bootstrap_error": bootstrap_error, "measurement_rank": twin.last_measurement_rank},
    )

    state_before = twin.x_previous.copy()
    prior_z = h @ state_before
    estimate, held, reliable = twin._estimate(prior_z, pseudo_rx, gamma)
    check(
        "pseudo_only_postbootstrap_is_explicit_hold",
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
    estimate, held, reliable = twin._estimate(mixed_z, all_rx, gamma)
    check(
        "small_network_supported_q_update_accepted",
        not held
        and reliable
        and twin.last_hold_reason == "accepted"
        and twin.last_process_prior_active
        and twin.last_process_increment_mahalanobis < twin.last_process_guard_limit,
        {
            "process_increment_mahalanobis": twin.last_process_increment_mahalanobis,
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
            direction = np.zeros(n, dtype=float)
            direction[0] = 0.1
            return np.asarray(prior_state, dtype=float) + direction, True

    guarded_twin = make_twin(state_before, GuardStub())
    guarded_twin.has_valid_estimate = True
    estimate, held, reliable = guarded_twin._estimate(exact_z, all_rx, gamma)
    check(
        "catastrophic_process_coordinate_update_rejected",
        held
        and reliable
        and guarded_twin.last_hold_reason == "process_guard"
        and np.array_equal(estimate, state_before),
        {
            "process_guard_statistic": guarded_twin.last_process_guard_statistic,
            "process_guard_limit": guarded_twin.last_process_guard_limit,
        },
    )

    candidate_source = TWIN_CANDIDATE.read_text(encoding="utf-8")
    step_keys = dictionary_keys(candidate_source, "ProductionTwin", "update", "row")
    event_keys = dictionary_keys(candidate_source, None, "aggregate_events", "row")
    publication_keys = dictionary_keys(candidate_source, None, "publish_score", "payload")
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
        ast.parse(candidate_source),
        None,
        "publish_score",
    )
    publication_text = (
        ast.get_source_segment(candidate_source, publication_node) or ""
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

    tree = ast.parse(candidate_source)
    estimate_node = find_function(tree, "ProductionTwin", "_estimate")
    estimate_text = ast.get_source_segment(candidate_source, estimate_node) or ""
    check(
        "prior_and_process_guard_wiring_present",
        "prior_state=prior_state" in estimate_text
        and "process_increment_mahalanobis > process_guard_limit" in estimate_text
        and "math.sqrt(self.n) * omega" not in estimate_text
        and "jump_norm > jump_limit" not in estimate_text,
        {"guard_policy": "q_process_mahalanobis"},
    )
    return tests


WORKSPACE.mkdir(parents=True, exist_ok=True)
for path in (LIVE_TRUST, LIVE_TWIN, TRUST_CANDIDATE, ROOT / "feeder.npz"):
    require(path.is_file(), f"required input is missing: {path}")
require(sha256(LIVE_TRUST) == EXPECTED_LIVE_TRUST_SHA256, "live trust hash mismatch")
require(sha256(TRUST_CANDIDATE) == EXPECTED_TRUST_CANDIDATE_SHA256, "trust candidate hash mismatch")

live_trust_before = sha256(LIVE_TRUST)
live_twin_before = sha256(LIVE_TWIN)
trust_candidate_before = sha256(TRUST_CANDIDATE)
twin_candidate_hash = build_twin_candidate()
trust_module = load_module("trust_metric", TRUST_CANDIDATE)
twin_module = load_module("twin_fed_v5_2_candidate", TWIN_CANDIDATE)
tests = run_tests(trust_module, twin_module)

require(sha256(LIVE_TRUST) == live_trust_before, "live trust was modified")
require(sha256(LIVE_TWIN) == live_twin_before, "live twin was modified")
require(sha256(TRUST_CANDIDATE) == trust_candidate_before, "trust candidate was modified")

report = {
    "schema": "paper1.v5_2.twin_integration_tests.v1",
    "live_trust_sha256": live_trust_before,
    "live_twin_sha256": live_twin_before,
    "trust_candidate_sha256": trust_candidate_before,
    "twin_candidate_sha256": twin_candidate_hash,
    "estimator_mode": "q_prior_innovation_after_measurement_bootstrap",
    "guard_policy": "q_process_mahalanobis",
    "guard_limit_source": "existing_hold_factor_dimensionless",
    "legacy_euclidean_jump_guard_active": False,
    "tests_passed": len(tests),
    "tests_failed": 0,
    "tests": tests,
    "live_files_modified": False,
    "simulation_rerun": False,
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
}
REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_2_TWIN_INTEGRATION_TESTS_OK")
print("TESTS_PASSED=", len(tests))
print("TESTS_FAILED=0")
print("TRUST_CANDIDATE_SHA256=", trust_candidate_before)
print("TWIN_CANDIDATE_SHA256=", twin_candidate_hash)
print("REPORT_SHA256=", sha256(REPORT_PATH))
print("ESTIMATOR_MODE=q_prior_innovation_after_measurement_bootstrap")
print("PROCESS_GUARD_POLICY=q_process_mahalanobis")
print("PROCESS_GUARD_LIMIT_SOURCE=existing_hold_factor_dimensionless")
print("LEGACY_EUCLIDEAN_JUMP_GUARD_ACTIVE=False")
print("LIVE_FILES_MODIFIED=False")
print("SIMULATION_RERUN=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_2_INTEGRATED_CANDIDATE_READY_FOR_INSTALL_REVIEW")
