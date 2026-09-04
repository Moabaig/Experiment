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
WORKSPACE = ROOT / "paper1_v5_2_repair"
SOURCE_PATH = ROOT / "trust_metric.py"
CANDIDATE_PATH = WORKSPACE / "trust_metric_v5_2_candidate.py"
REPORT_PATH = WORKSPACE / "v5_2_estimator_candidate_tests.json"
EXPECTED_SOURCE_SHA256 = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)
RCOND = math.sqrt(np.finfo(float).eps)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_method(source: str, class_name: str, method_name: str, replacement: str) -> str:
    tree = ast.parse(source)
    matches = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            matches.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == method_name
            )
    require(len(matches) == 1, f"expected one {class_name}.{method_name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    require(node.end_lineno is not None, "AST end line is unavailable")
    return "".join(
        lines[: node.lineno - 1]
        + [replacement.rstrip() + "\n"]
        + lines[node.end_lineno :]
    )


Q_INITIALIZATION = """

        process_covariance = np.asarray(Q, dtype=float)
        if process_covariance.shape != (self.n, self.n):
            raise ValueError("Q must match the state dimension")
        if not np.all(np.isfinite(process_covariance)):
            raise ValueError("Q contains nonfinite values")
        process_covariance = 0.5 * (
            process_covariance + process_covariance.T
        )
        try:
            process_cholesky = np.linalg.cholesky(process_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Q must be symmetric positive definite for the process prior"
            ) from exc
        process_rms_increment = float(
            np.sqrt(np.trace(process_covariance))
        )
        if (
            not np.isfinite(process_rms_increment)
            or process_rms_increment <= 0.0
        ):
            raise ValueError("Q has an invalid RMS process increment")
        self.process_covariance = process_covariance
        self.process_cholesky = process_cholesky
        self.process_rms_increment = process_rms_increment
""".rstrip()


ESTIMATE_METHOD = r'''
    def estimate(self, z, rx, gamma, *, prior_state=None):
        """Solve the age-aware state estimate in a rank-revealing form.

        With no prior state, this is the V5.1 direct weighted SVD solve used
        only for bootstrap.  With a valid prior state, solve the process-noise
        coordinates ``x = prior_state + L_Q u`` by augmented least squares:

            min ||W(H L_Q u - (z - H prior_state))||_2^2 + ||u||_2^2.

        This is a Q-centred MAP/Tikhonov update.  It never forms normal
        equations and never anchors the absolute state to zero.  Measurement
        rank and posterior-system reliability are recorded separately.
        """
        z_array = np.asarray(z, dtype=float).reshape(-1)
        rx_array = np.asarray(rx, dtype=bool).reshape(-1)
        gamma_array = np.asarray(gamma, dtype=float).reshape(-1)

        if (
            len(z_array) != self.m
            or len(rx_array) != self.m
            or len(gamma_array) != self.m
        ):
            raise ValueError(
                "z, rx, and gamma must match the measurement dimension"
            )

        prior_active = prior_state is not None
        if prior_active:
            prior_array = np.asarray(prior_state, dtype=float).reshape(-1)
            if prior_array.shape != (self.n,):
                raise ValueError("prior_state must match the state dimension")
            if not np.all(np.isfinite(prior_array)):
                raise ValueError("prior_state contains nonfinite values")
        else:
            prior_array = None

        rcond = float(np.sqrt(np.finfo(float).eps))
        ridge = float(self.cfg.ridge)
        if not np.isfinite(ridge) or ridge != 0.0:
            raise ValueError(
                "legacy zero-centred ridge must remain zero; use the Q prior"
            )

        self.last_estimator_mode = (
            "q_prior_innovation" if prior_active else "measurement_bootstrap"
        )
        self.last_estimator_solver = (
            "q_prior_augmented_lstsq_svd"
            if prior_active
            else "weighted_lstsq_svd"
        )
        self.last_estimator_rcond = rcond
        self.last_estimator_effective_rows = 0
        self.last_estimator_rank = 0
        self.last_estimator_condition = float("inf")
        self.last_estimator_singular_max = float("nan")
        self.last_estimator_singular_min = float("nan")
        self.last_estimator_residual_norm = float("nan")
        self.last_estimator_augmented_residual_norm = float("nan")
        self.last_measurement_rank = 0
        self.last_measurement_condition = float("inf")
        self.last_measurement_singular_max = float("nan")
        self.last_measurement_singular_min = float("nan")
        self.last_measurement_full_rank = False
        self.last_posterior_rank = 0
        self.last_posterior_condition = float("inf")
        self.last_posterior_singular_max = float("nan")
        self.last_posterior_singular_min = float("nan")
        self.last_posterior_reliable = False
        self.last_process_prior_active = bool(prior_active)
        self.last_process_rms_increment = float(self.process_rms_increment)
        self.last_process_increment_norm = float("nan")
        self.last_process_increment_mahalanobis = float("nan")

        idx = np.flatnonzero(rx_array)
        raw_weights = gamma_array[idx] / self.s2[idx]
        valid = (
            np.isfinite(raw_weights)
            & (raw_weights > 0.0)
            & np.isfinite(z_array[idx])
        )
        idx = idx[valid]
        weights = raw_weights[valid]
        self.last_estimator_effective_rows = int(len(idx))

        if len(idx) == 0:
            weighted_h = np.empty((0, self.n), dtype=float)
            weighted_z = np.empty(0, dtype=float)
        else:
            square_root_weights = np.sqrt(weights)
            weighted_h = self.H[idx] * square_root_weights[:, None]
            weighted_z = z_array[idx] * square_root_weights

        if (
            not np.all(np.isfinite(weighted_h))
            or not np.all(np.isfinite(weighted_z))
        ):
            return np.full(self.n, np.nan), False

        if len(idx) > 0:
            try:
                measurement_singular = np.linalg.svd(
                    weighted_h,
                    compute_uv=False,
                )
            except np.linalg.LinAlgError:
                return np.full(self.n, np.nan), False
        else:
            measurement_singular = np.empty(0, dtype=float)

        if len(measurement_singular) > 0:
            measurement_singular_max = float(measurement_singular[0])
            measurement_threshold = rcond * measurement_singular_max
            measurement_rank = int(
                np.sum(measurement_singular > measurement_threshold)
            )
            measurement_singular_min = float(measurement_singular[-1])
            measurement_condition = (
                float(measurement_singular_max / measurement_singular_min)
                if measurement_rank == self.n
                and measurement_singular_min > 0.0
                else float("inf")
            )
        else:
            measurement_rank = 0
            measurement_singular_max = float("nan")
            measurement_singular_min = float("nan")
            measurement_condition = float("inf")

        measurement_full_rank = bool(
            measurement_rank == self.n
            and np.isfinite(measurement_condition)
            and measurement_condition <= 1.0 / rcond
        )
        self.last_measurement_rank = measurement_rank
        self.last_measurement_condition = measurement_condition
        self.last_measurement_singular_max = measurement_singular_max
        self.last_measurement_singular_min = measurement_singular_min
        self.last_measurement_full_rank = measurement_full_rank

        if prior_active:
            transformed_h = weighted_h @ self.process_cholesky
            innovation_rhs = weighted_z - weighted_h @ prior_array
            solve_h = np.vstack([transformed_h, np.eye(self.n)])
            solve_rhs = np.concatenate(
                [innovation_rhs, np.zeros(self.n, dtype=float)]
            )
        else:
            if len(idx) == 0:
                return np.full(self.n, np.nan), False
            solve_h = weighted_h
            solve_rhs = weighted_z

        try:
            solution, _, solve_rank, solve_singular = np.linalg.lstsq(
                solve_h,
                solve_rhs,
                rcond=rcond,
            )
        except np.linalg.LinAlgError:
            return np.full(self.n, np.nan), False

        solution = np.asarray(solution, dtype=float).reshape(-1)
        solve_rank = int(solve_rank)
        solve_singular = np.asarray(solve_singular, dtype=float)
        if len(solve_singular) > 0:
            solve_singular_max = float(solve_singular[0])
            solve_singular_min = float(solve_singular[-1])
            solve_condition = (
                float(solve_singular_max / solve_singular_min)
                if solve_singular_min > 0.0
                else float("inf")
            )
        else:
            solve_singular_max = float("nan")
            solve_singular_min = float("nan")
            solve_condition = float("inf")

        posterior_reliable = bool(
            solve_rank == self.n
            and np.isfinite(solve_condition)
            and solve_condition <= 1.0 / rcond
            and solution.shape == (self.n,)
            and np.all(np.isfinite(solution))
        )

        if prior_active:
            process_coordinates = solution
            process_increment = self.process_cholesky @ process_coordinates
            candidate = prior_array + process_increment
            self.last_process_increment_norm = float(
                np.linalg.norm(process_increment)
            )
            self.last_process_increment_mahalanobis = float(
                np.linalg.norm(process_coordinates)
            )
        else:
            candidate = solution

        candidate = np.asarray(candidate, dtype=float).reshape(-1)
        candidate_finite = bool(
            candidate.shape == (self.n,)
            and np.all(np.isfinite(candidate))
        )
        measurement_residual_norm = (
            float(np.linalg.norm(weighted_h @ candidate - weighted_z))
            if candidate_finite
            else float("nan")
        )
        augmented_residual_norm = (
            float(np.linalg.norm(solve_h @ solution - solve_rhs))
            if candidate_finite
            else float("nan")
        )

        self.last_estimator_rank = solve_rank
        self.last_estimator_condition = solve_condition
        self.last_estimator_singular_max = solve_singular_max
        self.last_estimator_singular_min = solve_singular_min
        self.last_estimator_residual_norm = measurement_residual_norm
        self.last_estimator_augmented_residual_norm = augmented_residual_norm
        self.last_posterior_rank = solve_rank
        self.last_posterior_condition = solve_condition
        self.last_posterior_singular_max = solve_singular_max
        self.last_posterior_singular_min = solve_singular_min
        self.last_posterior_reliable = posterior_reliable

        if prior_active:
            solved_reliably = bool(candidate_finite and posterior_reliable)
        else:
            solved_reliably = bool(candidate_finite and measurement_full_rank)

        return candidate, solved_reliably
'''.strip("\n")


def build_candidate() -> str:
    require(SOURCE_PATH.is_file(), f"missing source: {SOURCE_PATH}")
    observed = sha256(SOURCE_PATH)
    require(observed == EXPECTED_SOURCE_SHA256, f"unexpected source hash: {observed}")
    source = SOURCE_PATH.read_text(encoding="utf-8-sig")
    anchor = "        self.m, self.n = self.H.shape\n"
    require(source.count(anchor) == 1, "state-dimension anchor is not unique")
    source = source.replace(anchor, anchor + Q_INITIALIZATION + "\n", 1)
    source = replace_method(source, "TrustMetric", "estimate", ESTIMATE_METHOD)
    compile(source, str(CANDIDATE_PATH), "exec")
    CANDIDATE_PATH.write_text(source, encoding="utf-8", newline="\n")
    return sha256(CANDIDATE_PATH)


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "trust_metric_v5_2_candidate",
        CANDIDATE_PATH,
    )
    require(spec is not None and spec.loader is not None, "candidate import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_tests(module) -> list[dict]:
    feeder_path = ROOT / "feeder.npz"
    require(feeder_path.is_file(), f"missing feeder: {feeder_path}")
    with np.load(feeder_path, allow_pickle=False) as feeder:
        h = np.asarray(feeder["H"], dtype=float)
        sigma2 = np.asarray(feeder["sigma2"], dtype=float)
        q = np.asarray(feeder["Q"], dtype=float)
        n_telemetry = int(feeder["n_telemetry"])

    cfg = module.MetricConfig(n_telemetry=n_telemetry, ridge=0.0)
    metric = module.TrustMetric(h, sigma2, q, cfg)
    m, n = h.shape
    rng = np.random.default_rng(52002)
    all_rx = np.ones(m, dtype=bool)
    gamma_one = np.ones(m, dtype=float)
    pseudo_rx = np.ones(m, dtype=bool)
    pseudo_rx[:n_telemetry] = False
    tests = []

    def check(name: str, condition: bool, detail: dict) -> None:
        require(condition, f"{name} failed: {detail}")
        tests.append({"name": name, "status": "PASS", **detail})
        print(name, "PASS")

    candidate_source = CANDIDATE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(candidate_source)
    estimate_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrustMetric":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "estimate":
                    estimate_node = item
    require(estimate_node is not None, "candidate estimate method is missing")
    estimate_text = ast.get_source_segment(candidate_source, estimate_node) or ""
    check(
        "no_normal_equations_or_dense_solve",
        "np.linalg.solve(" not in estimate_text and ".T @" not in estimate_text,
        {"solver": "direct_augmented_lstsq_svd"},
    )

    state = rng.normal(scale=0.02, size=n)
    exact_z = h @ state
    estimate, reliable = metric.estimate(exact_z, all_rx, gamma_one)
    relative_error = float(
        np.linalg.norm(estimate - state) / max(np.linalg.norm(state), 1e-15)
    )
    check(
        "bootstrap_full_design_exact_recovery",
        reliable and relative_error < 1e-9 and metric.last_measurement_rank == n,
        {"relative_error": relative_error, "measurement_rank": metric.last_measurement_rank},
    )

    estimate, reliable = metric.estimate(exact_z, pseudo_rx, gamma_one)
    check(
        "bootstrap_pseudo_only_rejected",
        (not reliable) and metric.last_measurement_rank == 489,
        {"measurement_rank": metric.last_measurement_rank, "reliable": bool(reliable)},
    )

    prior = rng.normal(scale=0.02, size=n)
    prior_z = h @ prior
    estimate, reliable = metric.estimate(
        prior_z,
        pseudo_rx,
        gamma_one,
        prior_state=prior,
    )
    prior_hold_error = float(np.linalg.norm(estimate - prior))
    check(
        "prior_pseudo_only_holds_previous_state",
        reliable
        and prior_hold_error < 1e-12
        and metric.last_measurement_rank == 489
        and metric.last_posterior_rank == n,
        {
            "state_change_norm": prior_hold_error,
            "measurement_rank": metric.last_measurement_rank,
            "posterior_rank": metric.last_posterior_rank,
        },
    )

    process_coordinates = rng.normal(size=n)
    process_coordinates *= 0.5 / np.linalg.norm(process_coordinates)
    true_increment = metric.process_cholesky @ process_coordinates
    truth = prior + true_increment
    truth_z = h @ truth
    estimate, reliable = metric.estimate(
        truth_z,
        all_rx,
        gamma_one,
        prior_state=prior,
    )
    prior_error = float(np.linalg.norm(prior - truth))
    estimate_error = float(np.linalg.norm(estimate - truth))
    check(
        "q_prior_exact_data_moves_toward_truth",
        reliable
        and estimate_error < prior_error
        and metric.last_process_increment_mahalanobis <= 0.5 + 1e-10,
        {
            "prior_error": prior_error,
            "estimate_error": estimate_error,
            "process_mahalanobis": metric.last_process_increment_mahalanobis,
        },
    )

    noise_update_norms = []
    for _ in range(5):
        noisy_z = h @ prior + rng.normal(scale=np.sqrt(sigma2), size=m)
        estimate, reliable = metric.estimate(
            noisy_z,
            all_rx,
            gamma_one,
            prior_state=prior,
        )
        require(reliable, "noisy Q-prior solve was unreliable")
        noise_update_norms.append(float(np.linalg.norm(estimate - prior)))
    check(
        "production_noise_update_bounded_by_q_rms",
        max(noise_update_norms) < metric.process_rms_increment,
        {
            "maximum_update_norm": max(noise_update_norms),
            "q_rms_increment": metric.process_rms_increment,
        },
    )

    support_cases = []
    for telemetry_gamma in (1.0, 1e-3, 1e-6, 1e-12, 0.0):
        gamma = np.ones(m, dtype=float)
        gamma[:n_telemetry] = telemetry_gamma
        mixed_z = h @ prior
        mixed_z[:n_telemetry] = (h @ truth)[:n_telemetry]
        estimate, reliable = metric.estimate(
            mixed_z,
            all_rx,
            gamma,
            prior_state=prior,
        )
        support_cases.append(
            {
                "telemetry_gamma": telemetry_gamma,
                "reliable": bool(reliable),
                "measurement_rank": metric.last_measurement_rank,
                "posterior_rank": metric.last_posterior_rank,
                "update_norm": float(np.linalg.norm(estimate - prior)),
            }
        )
    check(
        "age_weight_stress_remains_posterior_reliable",
        all(case["reliable"] and case["posterior_rank"] == n for case in support_cases)
        and support_cases[-1]["measurement_rank"] == 489
        and support_cases[-1]["update_norm"] < 1e-12,
        {"cases": support_cases},
    )

    zero_anchor_prior = np.full(n, 0.25, dtype=float)
    zero_anchor_z = h @ zero_anchor_prior
    estimate, reliable = metric.estimate(
        zero_anchor_z,
        pseudo_rx,
        gamma_one,
        prior_state=zero_anchor_prior,
    )
    check(
        "prior_is_previous_state_not_zero",
        reliable
        and np.linalg.norm(estimate - zero_anchor_prior) < 1e-12
        and np.linalg.norm(estimate) > 1.0,
        {"candidate_norm": float(np.linalg.norm(estimate))},
    )

    check(
        "process_covariance_is_positive_definite",
        np.all(np.diag(metric.process_cholesky) > 0.0)
        and metric.process_rms_increment > 0.0,
        {"q_rms_increment": metric.process_rms_increment},
    )
    return tests


WORKSPACE.mkdir(parents=True, exist_ok=True)
source_hash_before = sha256(SOURCE_PATH)
candidate_hash = build_candidate()
module = load_candidate()
tests = run_tests(module)
source_hash_after = sha256(SOURCE_PATH)
require(source_hash_after == source_hash_before, "live trust_metric.py was modified")

report = {
    "schema": "paper1.v5_2.estimator_candidate_tests.v1",
    "source_sha256": source_hash_before,
    "candidate_sha256": candidate_hash,
    "candidate_path": str(CANDIDATE_PATH),
    "solver": "q_prior_augmented_lstsq_svd",
    "prior_center": "previous_state",
    "prior_covariance": "Q_dt_1_second",
    "normal_equations_used": False,
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

print("PAPER1_V5_2_ESTIMATOR_CANDIDATE_TESTS_OK")
print("TESTS_PASSED=", len(tests))
print("TESTS_FAILED=0")
print("SOURCE_SHA256=", source_hash_before)
print("CANDIDATE_SHA256=", candidate_hash)
print("REPORT_SHA256=", sha256(REPORT_PATH))
print("SOLVER=q_prior_augmented_lstsq_svd")
print("PRIOR_CENTER=previous_state")
print("PRIOR_COVARIANCE=Q_dt_1_second")
print("NORMAL_EQUATIONS_USED=False")
print("LIVE_FILES_MODIFIED=False")
print("SIMULATION_RERUN=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_2_ESTIMATOR_CANDIDATE_READY_FOR_TWIN_INTEGRATION")
