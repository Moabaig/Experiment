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
FEEDER_PATH = ROOT / "feeder.npz"
STEP_PATH = RUN_ROOT / "twin" / "scores.parquet"
REPORT_PATH = (
    ROOT
    / "paper1_v5_2_repair"
    / "v5_2_observability_ratcheting_diagnostic.json"
)

RCOND = math.sqrt(np.finfo(float).eps)
EXPECTED_STEPS = 13200
EXPECTED_STATES = 491
EXPECTED_TELEMETRY = 45
EXPECTED_PSEUDO_RANK = 489

MECHANICAL_COLUMNS = [
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
    "estimator_singular_max",
    "estimator_singular_min",
    "estimator_residual_norm",
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
    "jump_guard_policy",
    "model_increment_scale",
    "external_received_count",
    "external_total",
    "external_support_fraction",
    "pseudo_received_count",
    "pseudo_only_step",
    "external_support_present_step",
    "no_received_measurements_step",
    "mean_age_telemetry",
    "max_age_telemetry",
]

FORBIDDEN_COLUMNS = {
    "T",
    "alarm",
    "label",
    "is_nominal",
    "drift_family",
    "trajectory_id",
    "x_error_norm",
    "r",
    "chi2",
    "huber",
    "lnr",
    "s",
    "s_lmax",
    "s_trace",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def quantiles(values) -> dict:
    array = finite(values)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def svd_summary(matrix: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    threshold = RCOND * singular[0]
    rank = int(np.sum(singular > threshold))
    summary = {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "rank": rank,
        "nullity": int(matrix.shape[1] - rank),
        "rcond": RCOND,
        "rank_threshold": float(threshold),
        "singular_max": float(singular[0]),
        "singular_min": float(singular[-1]),
        "condition": float(singular[0] / singular[-1]),
        "five_smallest_singular_values": [
            float(value) for value in singular[-5:]
        ],
    }
    return summary, singular, vh


def q_coordinate_summary(weighted_h: np.ndarray, chol_q: np.ndarray) -> dict:
    transformed = weighted_h @ chol_q
    _, singular, _ = np.linalg.svd(transformed, full_matrices=False)
    augmented_condition = math.sqrt(
        (1.0 + float(singular[0]) ** 2)
        / (1.0 + float(singular[-1]) ** 2)
    )
    return {
        "singular_max": float(singular[0]),
        "singular_min": float(singular[-1]),
        "augmented_condition": float(augmented_condition),
    }


def first_crossing(values: np.ndarray, threshold: float):
    locations = np.flatnonzero(values >= threshold)
    return None if locations.size == 0 else int(locations[0])


def max_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


require(not (set(MECHANICAL_COLUMNS) & FORBIDDEN_COLUMNS), "forbidden read")
require(FEEDER_PATH.is_file(), f"missing feeder: {FEEDER_PATH}")
require(STEP_PATH.is_file(), f"missing twin steps: {STEP_PATH}")

with np.load(FEEDER_PATH, allow_pickle=False) as feeder:
    h = np.asarray(feeder["H"], dtype=float)
    h_telemetry = np.asarray(feeder["H_telemetry"], dtype=float)
    h_pseudo = np.asarray(feeder["H_pseudo"], dtype=float)
    sigma2 = np.asarray(feeder["sigma2"], dtype=float)
    sigma2_telemetry = np.asarray(feeder["sigma2_telemetry"], dtype=float)
    sigma2_pseudo = np.asarray(feeder["sigma2_pseudo"], dtype=float)
    q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])

require(h.shape == (583, EXPECTED_STATES), f"unexpected H shape: {h.shape}")
require(
    h_telemetry.shape == (EXPECTED_TELEMETRY, EXPECTED_STATES),
    f"unexpected telemetry shape: {h_telemetry.shape}",
)
require(n_telemetry == EXPECTED_TELEMETRY, "telemetry count mismatch")
require(np.all(np.isfinite(h)), "H contains nonfinite values")
require(np.all(sigma2 > 0.0), "sigma2 must be positive")

weighted_pseudo = h_pseudo / np.sqrt(sigma2_pseudo)[:, None]
weighted_telemetry = h_telemetry / np.sqrt(sigma2_telemetry)[:, None]
weighted_full = h / np.sqrt(sigma2)[:, None]

pseudo_summary, _, pseudo_vh = svd_summary(weighted_pseudo)
full_summary, _, _ = svd_summary(weighted_full)
require(pseudo_summary["rank"] == EXPECTED_PSEUDO_RANK, "pseudo rank mismatch")
require(pseudo_summary["nullity"] == 2, "pseudo nullity must be two")
require(full_summary["rank"] == EXPECTED_STATES, "full design is not full rank")

null_basis = pseudo_vh[pseudo_summary["rank"] :, :].T
telemetry_on_null = weighted_telemetry @ null_basis
null_singular = np.linalg.svd(telemetry_on_null, compute_uv=False)

q_symmetric = 0.5 * (q + q.T)
q_eigenvalues = np.linalg.eigvalsh(q_symmetric)
require(float(q_eigenvalues[0]) > 0.0, "Q is not positive definite")
chol_q = np.linalg.cholesky(q_symmetric)
q_on_null = null_basis.T @ q_symmetric @ null_basis
q_null_eigenvalues = np.linalg.eigvalsh(q_on_null)

q_rms_increment = math.sqrt(float(np.trace(q_symmetric)))
q_coordinate = {
    "pseudo": q_coordinate_summary(weighted_pseudo, chol_q),
    "full_gamma_one": q_coordinate_summary(weighted_full, chol_q),
}

steps = pd.read_parquet(STEP_PATH, columns=MECHANICAL_COLUMNS)
require(len(steps) == EXPECTED_STEPS, f"unexpected step count: {len(steps)}")
step_index = steps["step_index"].to_numpy(dtype=np.int64)
require(
    np.array_equal(step_index, np.arange(EXPECTED_STEPS, dtype=np.int64)),
    "step indices are not contiguous",
)

accepted = steps["state_update_accepted_step"].to_numpy(dtype=bool)
bootstrap = steps["bootstrap_accept_step"].to_numpy(dtype=bool)
reliable = steps["estimator_reliable"].to_numpy(dtype=bool)
held = steps["held"].to_numpy(dtype=bool)
candidate = steps["candidate_norm"].to_numpy(dtype=float)
previous = steps["previous_norm"].to_numpy(dtype=float)
jump = steps["jump_norm"].to_numpy(dtype=float)
jump_limit = steps["jump_limit"].to_numpy(dtype=float)

bootstrap_positions = np.flatnonzero(bootstrap)
reliable_positions = np.flatnonzero(reliable)
accepted_positions = np.flatnonzero(accepted)
require(bootstrap_positions.size == 1, "bootstrap must occur exactly once")
require(reliable_positions.size > 0, "no reliable solve was recorded")
require(accepted_positions.size > 0, "no state update was accepted")
bootstrap_position = int(bootstrap_positions[0])
require(
    bootstrap_position == int(reliable_positions[0]),
    "bootstrap is not the first reliable solve",
)
require(
    bootstrap_position == int(accepted_positions[0]),
    "bootstrap is not the first accepted solve",
)

expected_next_previous = np.where(accepted[:-1], candidate[:-1], previous[:-1])
transition_difference = np.abs(previous[1:] - expected_next_previous)
transition_tolerance = 1e-10 * np.maximum(1.0, np.abs(expected_next_previous))
transition_mismatch_count = int(
    np.sum(transition_difference > transition_tolerance)
)
require(transition_mismatch_count == 0, "state-anchor transition mismatch")

bootstrap_norm = float(candidate[bootstrap_position])
require(math.isfinite(bootstrap_norm) and bootstrap_norm > 0.0, "bad bootstrap norm")
accepted_delta = candidate[accepted] - previous[accepted]
accepted_before = previous[accepted]
accepted_after = candidate[accepted]
away_before = np.abs(accepted_before - bootstrap_norm)
away_after = np.abs(accepted_after - bootstrap_norm)

finite_reliable = reliable & np.isfinite(candidate)
candidate_ratio = candidate[finite_reliable] / bootstrap_norm
condition = steps["estimator_condition"].to_numpy(dtype=float)
correlation_mask = finite_reliable & np.isfinite(condition) & (condition > 0.0)
if int(np.sum(correlation_mask)) > 1:
    log_condition_candidate_correlation = float(
        np.corrcoef(
            np.log10(condition[correlation_mask]),
            candidate[correlation_mask] / bootstrap_norm,
        )[0, 1]
    )
else:
    log_condition_candidate_correlation = None

reason_summaries = {}
for reason, block in steps.groupby("hold_reason", dropna=False):
    key = str(reason)
    reason_summaries[key] = {
        "count": int(len(block)),
        "candidate_norm": quantiles(block["candidate_norm"]),
        "jump_norm": quantiles(block["jump_norm"]),
        "condition": quantiles(block["estimator_condition"]),
    }

fixed_jump_limit = float(np.median(jump_limit[np.isfinite(jump_limit)]))
model_increment_scale = float(
    np.median(finite(steps["model_increment_scale"]))
)

report = {
    "schema": "paper1.v5_2.observability_ratcheting_diagnostic.v1",
    "run_id": RUN_ID,
    "scope": {
        "simulation_rerun": False,
        "implementation_modified": False,
        "performance_outcome_columns_read": False,
        "performance_outcomes_inspected": False,
        "columns_read": MECHANICAL_COLUMNS,
    },
    "input_sha256": {
        "feeder_npz": sha256(FEEDER_PATH),
        "twin_scores_parquet": sha256(STEP_PATH),
    },
    "observability": {
        "weighted_pseudo": pseudo_summary,
        "weighted_full_gamma_one": full_summary,
        "pseudo_null_residual_frobenius": float(
            np.linalg.norm(weighted_pseudo @ null_basis)
        ),
        "telemetry_on_pseudo_null_singular_values": [
            float(value) for value in null_singular
        ],
        "telemetry_on_pseudo_null_condition": float(
            null_singular[0] / null_singular[-1]
        ),
    },
    "physical_process_prior": {
        "q_positive_definite": True,
        "q_eigenvalue_min": float(q_eigenvalues[0]),
        "q_eigenvalue_max": float(q_eigenvalues[-1]),
        "q_condition": float(q_eigenvalues[-1] / q_eigenvalues[0]),
        "q_rms_increment": q_rms_increment,
        "q_nullspace_eigenvalues": [
            float(value) for value in q_null_eigenvalues
        ],
        "q_coordinate_design": q_coordinate,
    },
    "run_mechanics": {
        "steps": int(len(steps)),
        "bootstrap_position": bootstrap_position,
        "bootstrap_step_index": int(step_index[bootstrap_position]),
        "bootstrap_norm": bootstrap_norm,
        "reliable_count": int(np.sum(reliable)),
        "accepted_count": int(np.sum(accepted)),
        "held_count": int(np.sum(held)),
        "hold_reason_counts": {
            str(key): int(value)
            for key, value in steps["hold_reason"].value_counts().items()
        },
        "transition_mismatch_count": transition_mismatch_count,
        "fixed_jump_limit": fixed_jump_limit,
        "model_increment_scale": model_increment_scale,
        "jump_limit_to_q_rms_ratio": fixed_jump_limit / q_rms_increment,
        "model_increment_scale_to_q_rms_ratio": (
            model_increment_scale / q_rms_increment
        ),
        "reliable_candidate_to_bootstrap_ratio": quantiles(candidate_ratio),
        "first_previous_norm_crossing_step": {
            str(multiplier): first_crossing(previous, multiplier * bootstrap_norm)
            for multiplier in (2, 5, 10, 20)
        },
        "accepted_positive_norm_increment_sum": float(
            np.sum(np.maximum(accepted_delta, 0.0))
        ),
        "accepted_negative_norm_increment_sum": float(
            np.sum(np.minimum(accepted_delta, 0.0))
        ),
        "accepted_moves_away_from_bootstrap_count": int(
            np.sum(away_after > away_before + 1e-12)
        ),
        "accepted_moves_toward_bootstrap_count": int(
            np.sum(away_after < away_before - 1e-12)
        ),
        "maximum_consecutive_accepted_steps": max_true_run(accepted),
        "maximum_consecutive_jump_guard_steps": max_true_run(
            steps["jump_guard_hold_step"].to_numpy(dtype=bool)
        ),
        "log_condition_candidate_ratio_correlation": (
            log_condition_candidate_correlation
        ),
        "by_hold_reason": reason_summaries,
    },
    "classification": {
        "pseudo_design_is_rank_deficient": True,
        "numerical_full_rank_is_not_physical_reliability": True,
        "accepted_updates_reanchor_pseudo_measurements": True,
        "fixed_jump_guard_is_not_a_process_prior": True,
        "mechanism": (
            "network-supported WLS updates weak state directions; accepted "
            "sub-limit updates become the next pseudo-measurement anchor and "
            "can accumulate as state-scale ratcheting"
        ),
    },
    "v5_2_design_decision": {
        "bootstrap": "first measurement-full-rank reliable unregularized solve",
        "post_bootstrap_parameterization": "x_candidate = x_previous + L_Q @ u",
        "post_bootstrap_objective": (
            "minimize ||W(H L_Q u - (z - H x_previous))||_2^2 + ||u||_2^2"
        ),
        "solver": "direct_augmented_lstsq_svd",
        "normal_equations": False,
        "prior_center": "x_previous",
        "prior_covariance": "Q at dt=1 second",
        "measurement_rank_logged_separately": True,
        "posterior_solver_reliability_logged_separately": True,
        "jump_guard_role": "secondary_nonfinite_and_catastrophic_safety_check",
        "tuning_from_performance_outcomes": False,
    },
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_2_OBSERVABILITY_RATCHETING_DIAGNOSTIC_OK")
print("PSEUDO_RANK=", pseudo_summary["rank"])
print("PSEUDO_NULLITY=", pseudo_summary["nullity"])
print("FULL_GAMMA_ONE_RANK=", full_summary["rank"])
print("Q_POSITIVE_DEFINITE=True")
print("Q_RMS_INCREMENT=", q_rms_increment)
print("Q_NULLSPACE_EIGENVALUES=", json.dumps(report["physical_process_prior"]["q_nullspace_eigenvalues"]))
print("Q_AUGMENTED_FULL_CONDITION=", q_coordinate["full_gamma_one"]["augmented_condition"])
print("BOOTSTRAP_STEP=", report["run_mechanics"]["bootstrap_step_index"])
print("ACCEPTED_COUNT=", report["run_mechanics"]["accepted_count"])
print("RELIABLE_COUNT=", report["run_mechanics"]["reliable_count"])
print("JUMP_LIMIT_TO_Q_RMS_RATIO=", report["run_mechanics"]["jump_limit_to_q_rms_ratio"])
print("CANDIDATE_TO_BOOTSTRAP_RATIO=", json.dumps(report["run_mechanics"]["reliable_candidate_to_bootstrap_ratio"], sort_keys=True))
print("ACCEPTED_MOVES_AWAY=", report["run_mechanics"]["accepted_moves_away_from_bootstrap_count"])
print("ACCEPTED_MOVES_TOWARD=", report["run_mechanics"]["accepted_moves_toward_bootstrap_count"])
print("TRANSITION_MISMATCH_COUNT=", transition_mismatch_count)
print("CLASSIFICATION=PSEUDO_NULLSPACE_PLUS_REANCHORING_RATCHET")
print("V5_2_ESTIMATOR=Q_CENTERED_AUGMENTED_INNOVATION_LSTSQ")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("REPORT=", REPORT_PATH)
print("REPORT_SHA256=", sha256(REPORT_PATH))
print("PAPER1_V5_2_ROOT_CAUSE_DIAGNOSTIC_COMPLETE")
