from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


ROOT = Path("/workspace")
RUN_ID = "paper1_v5_2mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID
CELL_EVIDENCE_ROOT = ROOT / "frozen" / "v52cell" / "C20260903_174122"

N = 491
M = 583
STEPS = 13200
EVENTS = 1100
STEPS_PER_EVENT = 12
EXTERNAL_TOTAL = 45
PSEUDO_TOTAL = 538
EXPECTED_RCOND = math.sqrt(np.finfo(float).eps)
EXPECTED_GUARD_LIMIT = 50.0
EXPECTED_BANDWIDTH_CAP = 1.0e12

EXPECTED_TRUST = "936dd373a2d8a2f0b905604ca4c3de61ec2cc889ba233aa150a24f44f2926fe6"
EXPECTED_TWIN = "9cd9ffaa32dcfe2f12ed161a8d62d2d97b2ab0b4d462fda0e97e7f46572043a4"
EXPECTED_FEEDER = "9df3426ea48c55f509e1f5f149e72f4e076de7d9099980d97e721efb94c8bd5d"
EXPECTED_ENV = "55a4fcb1acb19d86cbe2da4bcc4fe814170a14a5a637ec6cec97d9c94195d694"
EXPECTED_RUNNER = "5753de4e708206d0a1f8669adc52fa74ba5e4395318a24ccfbfad6c1fceb6629"
EXPECTED_CONTRACT = "11b20715bd970988a25429ba645373382671c8d5713cba5a836705f83b09256c"
EXPECTED_GATE_RECORD = "e48be42d0d1a8b735bc02718d3a485eb8d4c1f962116fef14533e0c64b22bfcd"
EXPECTED_GATE_MANIFEST = "cae5b6a07227ad1ad28b6aee8da96f8faca65adf01dbf998db599fba480e36be"

EXPECTED_PARQUET_HASHES = {
    "oracle/oracle_events.parquet": "b754cc1caff4e2d67604d89d53f8d76aa54681387b4188d7a070b835c2fd74d8",
    "oracle/oracle_scores.parquet": "80c2fbb446dcbf0f2fde6131ebd35fe1cd69aecd3e4c8b1bf79c1a7736de33c8",
    "twin/scores.parquet": "adad69f15d4478efc5698c01c34fb7ad7b2db4d078f87dbfec55519bcf50eee1",
    "twin/scores.partial.parquet": "adad69f15d4478efc5698c01c34fb7ad7b2db4d078f87dbfec55519bcf50eee1",
    "twin/scores_events.parquet": "8bd205aa1f59af3ceb19c83da260ef43366c77ecf30919337b92a398ccb14447",
    "twin/scores_events.partial.parquet": "8bd205aa1f59af3ceb19c83da260ef43366c77ecf30919337b92a398ccb14447",
}

STEP_COLUMNS = [
    "step_index", "event_id", "step_in_event", "bandwidth_level",
    "bandwidth_cap_bps", "held", "solve_exact", "estimator_reliable",
    "estimator_solver", "estimator_mode", "estimator_rcond",
    "estimator_effective_rows", "estimator_rank", "estimator_condition",
    "estimator_singular_max", "estimator_singular_min",
    "estimator_residual_norm", "estimator_augmented_residual_norm",
    "measurement_rank", "measurement_condition", "measurement_singular_max",
    "measurement_singular_min", "measurement_full_rank", "posterior_rank",
    "posterior_condition", "posterior_singular_max", "posterior_singular_min",
    "posterior_reliable", "hold_reason", "state_update_accepted_step",
    "bootstrap_accept_step", "solve_inexact_hold_step",
    "nonfinite_candidate_hold_step", "jump_guard_hold_step",
    "prior_only_hold_step", "process_diagnostic_invalid_hold_step",
    "process_guard_hold_step", "candidate_norm", "previous_norm", "jump_norm",
    "jump_limit", "jump_guard_policy", "model_increment_scale",
    "process_prior_active", "process_rms_increment", "process_increment_norm",
    "process_increment_mahalanobis", "process_guard_statistic",
    "process_guard_limit", "process_guard_policy", "external_received_count",
    "external_total", "external_support_fraction", "pseudo_received_count",
    "pseudo_only_step", "external_support_present_step",
    "no_received_measurements_step",
]

EVENT_COLUMNS = [
    "event_id", "bandwidth_level", "bandwidth_cap_bps", "held", "held_any",
    "solve_exact_fraction", "solve_exact_all", "estimator_reliable_fraction",
    "estimator_reliable_all", "estimator_solver", "estimator_rcond",
    "estimator_effective_rows_min", "estimator_effective_rows_max",
    "estimator_rank_min", "estimator_condition_max",
    "estimator_singular_max_max", "estimator_singular_min_min",
    "estimator_residual_norm_max", "estimator_augmented_residual_norm_max",
    "estimator_mode_first", "estimator_mode_last", "measurement_rank_min",
    "measurement_condition_max", "measurement_full_rank_fraction",
    "posterior_rank_min", "posterior_condition_max",
    "posterior_reliable_fraction", "state_update_accepted_fraction",
    "bootstrap_accept_fraction", "solve_inexact_hold_fraction",
    "nonfinite_candidate_hold_fraction", "jump_guard_hold_fraction",
    "prior_only_hold_fraction", "process_diagnostic_invalid_hold_fraction",
    "process_guard_hold_fraction", "pseudo_only_fraction",
    "external_support_present_fraction", "no_received_measurements_fraction",
    "candidate_norm_max", "previous_norm_max", "jump_norm_max",
    "jump_limit_min", "jump_guard_policy", "model_increment_scale",
    "process_prior_active_fraction", "process_rms_increment",
    "process_increment_norm_max", "process_increment_mahalanobis_max",
    "process_guard_statistic_max", "process_guard_limit",
    "process_guard_policy", "external_received_count_min", "external_total",
    "external_support_fraction_min", "pseudo_received_count_max", "steps",
]

FORBIDDEN_PERFORMANCE_COLUMNS = {
    "T", "alarm", "label", "is_nominal", "drift_family", "trajectory_id",
    "x_error_norm", "r", "chi2", "huber", "lnr", "s", "s_lmax",
    "s_trace", "sB1", "sB2", "s_gated_lmax", "s_gated_trace", "u_lmax",
    "u_trace", "u_lmax_mean", "u_trace_mean", "b1", "b2",
    "mean_age_telemetry", "max_age_telemetry", "floor_kind",
    "floor_deflation", "loss_quantile",
}


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    require(observed == expected, f"hash mismatch for {path}: {observed}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bools(series: pd.Series) -> np.ndarray:
    return series.astype(bool).to_numpy()


def numbers(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def same_float(left: float, right: float, *, rtol=1e-9, atol=1e-12) -> bool:
    left = float(left)
    right = float(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return bool(math.isclose(left, right, rel_tol=rtol, abs_tol=atol))


def same_array(left, right, *, rtol=1e-9, atol=1e-12) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )


def aggregate_numeric(series: pd.Series, operation: str) -> float:
    values = numbers(series)
    valid = ~np.isnan(values)
    if not np.any(valid):
        return float("nan")
    if operation == "max":
        return float(np.max(values[valid]))
    if operation == "min":
        return float(np.min(values[valid]))
    raise ValueError(operation)


def safe_run_manifest_path(raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    marker = f"/runs/{RUN_ID}/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    else:
        relative = normalized.lstrip("/")
    pure = PurePosixPath(relative)
    require(
        relative and not pure.is_absolute() and ".." not in pure.parts,
        f"unsafe cell-manifest path: {raw_path}",
    )
    resolved = (RUN_ROOT / Path(*pure.parts)).resolve()
    require(
        resolved == RUN_ROOT.resolve() or RUN_ROOT.resolve() in resolved.parents,
        f"cell-manifest path escapes run root: {raw_path}",
    )
    return resolved


def resolve_workspace_path(raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    if normalized.startswith("/workspace/"):
        relative = normalized[len("/workspace/") :]
    else:
        relative = normalized.lstrip("/")
    pure = PurePosixPath(relative)
    require(
        relative and not pure.is_absolute() and ".." not in pure.parts,
        f"unsafe workspace path: {raw_path}",
    )
    resolved = (ROOT / Path(*pure.parts)).resolve()
    require(
        resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents,
        f"path escapes workspace: {raw_path}",
    )
    return resolved


print("PAPER1_V5_2_ORACLE_MECHANICAL_AUDIT")
print(f"RUN_ID={RUN_ID}")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")

require(
    not ((set(STEP_COLUMNS) | set(EVENT_COLUMNS)) & FORBIDDEN_PERFORMANCE_COLUMNS),
    "audit column allowlist intersects the performance-outcome denylist",
)

trust_path = ROOT / "trust_metric.py"
twin_path = ROOT / "twin_fed.py"
feeder_path = ROOT / "feeder.npz"
env_path = ROOT / ".env"
runner_path = ROOT / "run_paper1_factor_campaign_v5_2_mechanical.ps1"
contract_path = ROOT / "paper1_v5_2_repair" / "paper1_v5_2_mechanical_validation_contract.json"
gate_record_path = CELL_EVIDENCE_ROOT / "paper1_v5_2_oracle_mechanical_cell_record.json"
gate_manifest_path = CELL_EVIDENCE_ROOT / "PAPER1_V5_2_ORACLE_CELL_GATE_SHA256.csv"
cell_record_path = RUN_ROOT / "cell_record.paper1.v5_2.mechanical.json"
cell_manifest_path = RUN_ROOT / "CELL_OUTPUT_SHA256SUMS.csv"
step_path = RUN_ROOT / "twin" / "scores.parquet"
event_path = RUN_ROOT / "twin" / "scores_events.parquet"

for path, expected in (
    (trust_path, EXPECTED_TRUST),
    (twin_path, EXPECTED_TWIN),
    (feeder_path, EXPECTED_FEEDER),
    (env_path, EXPECTED_ENV),
    (runner_path, EXPECTED_RUNNER),
    (contract_path, EXPECTED_CONTRACT),
    (gate_record_path, EXPECTED_GATE_RECORD),
    (gate_manifest_path, EXPECTED_GATE_MANIFEST),
):
    require_hash(path, expected)

require(RUN_ROOT.is_dir(), f"completed run directory is missing: {RUN_ROOT}")

run_files = sorted(path for path in RUN_ROOT.rglob("*") if path.is_file())
require(len(run_files) == 12, f"unexpected output-file count: {len(run_files)}")

with cell_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
    manifest_rows = list(csv.DictReader(handle))
require(len(manifest_rows) == 11, f"unexpected cell-manifest rows: {len(manifest_rows)}")
manifest_files: set[Path] = set()
for row in manifest_rows:
    require("Path" in row and "Hash" in row, "cell manifest lacks Path or Hash")
    path = safe_run_manifest_path(row["Path"])
    require(path not in manifest_files, f"duplicate cell-manifest path: {path}")
    manifest_files.add(path)
    require_hash(path, row["Hash"].lower())
require(
    manifest_files == {path.resolve() for path in run_files if path != cell_manifest_path},
    "cell manifest and run-directory files disagree",
)

for relative, expected in EXPECTED_PARQUET_HASHES.items():
    require_hash(RUN_ROOT / Path(*PurePosixPath(relative).parts), expected)

require(
    sha256(RUN_ROOT / "twin" / "scores.parquet")
    == sha256(RUN_ROOT / "twin" / "scores.partial.parquet"),
    "twin final and partial step files differ",
)
require(
    sha256(RUN_ROOT / "twin" / "scores_events.parquet")
    == sha256(RUN_ROOT / "twin" / "scores_events.partial.parquet"),
    "twin final and partial event files differ",
)

gate_record = read_json(gate_record_path)
require(gate_record["schema"] == "paper1.v5_2.oracle_mechanical_cell_gate.v1", "gate-record schema mismatch")
require(gate_record["run_id"] == RUN_ID, "gate-record run mismatch")
require(gate_record["simulation_completed"] is True, "gate record does not confirm completion")
require(gate_record["simulation_rerun"] is False, "gate record indicates a rerun")
require(gate_record["implementation_modified"] is False, "gate record indicates implementation modification")
require(gate_record["env_restored_exactly"] is True, "gate record does not confirm exact environment restoration")
require(gate_record["performance_outcome_columns_read"] is False, "gate record indicates performance columns were read")
require(gate_record["performance_outcomes_inspected"] is False, "gate record indicates performance inspection")

contract = read_json(contract_path)
require(contract["schema"] == "paper1.v5_2.mechanical_validation_contract.v1", "contract schema mismatch")
require(contract["candidate_version"] == "paper1_v5_2", "contract version mismatch")
require(contract["estimator"]["mode"] == "q_prior_innovation_after_measurement_bootstrap", "contract estimator mismatch")
require(contract["estimator"]["prior_center"] == "previous_state", "contract prior center mismatch")
require(contract["estimator"]["prior_covariance"] == "Q_dt_1_second", "contract prior covariance mismatch")
require(contract["estimator"]["normal_equations_used"] is False, "contract permits normal equations")
require(contract["guard"]["policy"] == "q_process_mahalanobis", "contract guard mismatch")
require(contract["guard"]["legacy_euclidean_jump_guard_active"] is False, "contract enables legacy Euclidean guard")
authorization = contract["authorization"]
require(authorization["mechanical_validation_only"] is True, "contract is not mechanical-only")
require(authorization["authorized_cell_count"] == 1, "contract authorizes more than one cell")
require(authorization["calibration_authorized"] is False, "contract authorizes calibration")
require(authorization["full_campaign_authorized"] is False, "contract authorizes full campaign")
require(authorization["performance_outcome_columns_may_be_read"] is False, "contract permits performance-column reads")
require(authorization["performance_outcomes_may_be_inspected"] is False, "contract permits performance inspection")

record = read_json(cell_record_path)
require(record["schema"] == "twin.factor.cell.record.paper1.v5_2.mechanical", "cell schema mismatch")
require(record["run_id"] == RUN_ID, "cell run ID mismatch")
require(record["status"] == "complete", "cell status is not complete")
require(record["estimator_version"] == "paper1_v5_2", "cell estimator version mismatch")
require(record["estimator_solver"] == "q_prior_innovation_after_measurement_bootstrap", "cell estimator descriptor mismatch")
require(record["jump_guard_policy"] == "q_process_mahalanobis", "cell guard policy mismatch")
require(record["normal_equations_used"] is False, "cell record says normal equations were used")
require(record["mechanical_validation_only"] is True, "cell record is not mechanical-only")
require(record["performance_outcomes_inspected"] is False, "cell record says performance was inspected")
require(record["trust_metric_sha256"].lower() == EXPECTED_TRUST, "cell trust hash mismatch")
require(record["twin_fed_sha256"].lower() == EXPECTED_TWIN, "cell twin hash mismatch")
require(record["bandwidth_level"] == "bw04_oracle", "cell bandwidth level mismatch")
require(float(record["bandwidth_cap_bps"]) == EXPECTED_BANDWIDTH_CAP, "cell bandwidth cap mismatch")

truth_path = resolve_workspace_path(record["truth_file"])
require_hash(truth_path, record["truth_sha256"].lower())

power_meta = read_json(RUN_ROOT / "power" / "meta.json")
net_meta = read_json(RUN_ROOT / "net" / "meta.json")
twin_meta = read_json(RUN_ROOT / "twin" / "meta.json")
oracle_meta = read_json(RUN_ROOT / "oracle" / "meta.json")

for name, meta, schema in (
    ("power", power_meta, "power.run.meta.v1"),
    ("net", net_meta, "net.run.meta.v1"),
    ("twin", twin_meta, "twin.run.meta.v1"),
    ("oracle", oracle_meta, "oracle.run.meta.v1"),
):
    require(meta["schema"] == schema, f"{name} meta schema mismatch")
    require(meta["status"] == "complete", f"{name} meta status is not complete")

require(int(power_meta["dimensions"]["steps"]) == STEPS, "power step count mismatch")
require(int(net_meta["events"]) == EVENTS, "network event count mismatch")
require(int(twin_meta["runtime_counts"]["steps"]) == STEPS, "twin step count mismatch")
require(int(twin_meta["runtime_counts"]["events"]) == EVENTS, "twin event count mismatch")
require(int(oracle_meta["runtime_counts"]["events"]) == EVENTS, "oracle event count mismatch")
require(power_meta["inputs"]["truth"]["sha256"].lower() == record["truth_sha256"].lower(), "power truth hash mismatch")
require(oracle_meta["inputs"]["truth"]["sha256"].lower() == record["truth_sha256"].lower(), "oracle truth hash mismatch")
require(net_meta["bandwidth_level"] == "bw04_oracle", "network bandwidth mismatch")
require(float(net_meta["bandwidth_cap_bps"]) == EXPECTED_BANDWIDTH_CAP, "network cap mismatch")
require(twin_meta["factor_design"]["bandwidth_level"] == "bw04_oracle", "twin bandwidth mismatch")
require(float(twin_meta["factor_design"]["bandwidth_cap_bps"]) == EXPECTED_BANDWIDTH_CAP, "twin cap mismatch")

with np.load(feeder_path, allow_pickle=False) as feeder:
    h = np.asarray(feeder["H"], dtype=float)
    sigma2 = np.asarray(feeder["sigma2"], dtype=float)
    q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])

require(h.shape == (M, N), f"unexpected feeder H shape: {h.shape}")
require(sigma2.shape == (M,), f"unexpected feeder sigma2 shape: {sigma2.shape}")
require(q.shape == (N, N), f"unexpected feeder Q shape: {q.shape}")
require(n_telemetry == EXTERNAL_TOTAL, "feeder telemetry count mismatch")
require(bool(np.all(np.isfinite(q))), "Q contains nonfinite values")
q_symmetric = 0.5 * (q + q.T)
require(bool(np.allclose(q, q_symmetric, rtol=0.0, atol=1e-14)), "Q is not symmetric")
q_eigenvalues = np.linalg.eigvalsh(q_symmetric)
require(float(q_eigenvalues[0]) > 0.0, "Q is not positive definite")
process_rms_expected = math.sqrt(float(np.trace(q_symmetric)))

# These calls read only the explicit mechanical/provenance allowlists.
step = pd.read_parquet(step_path, columns=STEP_COLUMNS)
event = pd.read_parquet(event_path, columns=EVENT_COLUMNS)

require(len(step) == STEPS, "step Parquet row count mismatch")
require(len(event) == EVENTS, "event Parquet row count mismatch")
step_index = step["step_index"].to_numpy(dtype=int)
require(np.array_equal(step_index, np.arange(STEPS)), "step_index sequence mismatch")
require(np.array_equal(step["event_id"].to_numpy(dtype=int), np.repeat(np.arange(EVENTS), STEPS_PER_EVENT)), "step event sequence mismatch")
require(np.array_equal(step["step_in_event"].to_numpy(dtype=int), np.tile(np.arange(STEPS_PER_EVENT), EVENTS)), "step_in_event sequence mismatch")
require(np.array_equal(event["event_id"].to_numpy(dtype=int), np.arange(EVENTS)), "event_id sequence mismatch")
require(bool((step["bandwidth_level"] == "bw04_oracle").all()), "step bandwidth mismatch")
require(bool((event["bandwidth_level"] == "bw04_oracle").all()), "event bandwidth mismatch")
require(bool(np.all(numbers(step["bandwidth_cap_bps"]) == EXPECTED_BANDWIDTH_CAP)), "step bandwidth cap mismatch")
require(bool(np.all(numbers(event["bandwidth_cap_bps"]) == EXPECTED_BANDWIDTH_CAP)), "event bandwidth cap mismatch")

held = bools(step["held"])
accepted = bools(step["state_update_accepted_step"])
reliable = bools(step["estimator_reliable"])
solve_exact = bools(step["solve_exact"])
posterior_reliable = bools(step["posterior_reliable"])
measurement_full_rank = bools(step["measurement_full_rank"])
prior_active = bools(step["process_prior_active"])
reasons = step["hold_reason"].astype(str).to_numpy()
candidate_norm = numbers(step["candidate_norm"])
previous_norm = numbers(step["previous_norm"])
jump_norm = numbers(step["jump_norm"])
jump_limit = numbers(step["jump_limit"])
rcond = numbers(step["estimator_rcond"])
effective_rows = step["estimator_effective_rows"].to_numpy(dtype=int)
rank = step["estimator_rank"].to_numpy(dtype=int)
condition = numbers(step["estimator_condition"])
measurement_rank = step["measurement_rank"].to_numpy(dtype=int)
measurement_condition = numbers(step["measurement_condition"])
posterior_rank = step["posterior_rank"].to_numpy(dtype=int)
posterior_condition = numbers(step["posterior_condition"])
process_rms = numbers(step["process_rms_increment"])
process_increment_norm = numbers(step["process_increment_norm"])
process_mahalanobis = numbers(step["process_increment_mahalanobis"])
process_guard_statistic = numbers(step["process_guard_statistic"])
process_guard_limit = numbers(step["process_guard_limit"])
external_received = step["external_received_count"].to_numpy(dtype=int)
external_total = step["external_total"].to_numpy(dtype=int)
pseudo_received = step["pseudo_received_count"].to_numpy(dtype=int)
support_fraction = numbers(step["external_support_fraction"])

allowed_reasons = {
    "accepted", "bootstrap_accept", "solve_inexact", "nonfinite_candidate",
    "prior_only_hold", "process_diagnostic_invalid", "process_guard",
}
require(set(reasons).issubset(allowed_reasons), "unexpected hold reason")
require(np.array_equal(accepted, ~held), "accepted flag does not equal not-held")
require(np.array_equal(reliable, solve_exact), "solve_exact and estimator_reliable disagree")
require(np.array_equal(accepted, np.isin(reasons, ["accepted", "bootstrap_accept"])), "decision reason and accepted flag disagree")

indicator_pairs = {
    "bootstrap_accept_step": "bootstrap_accept",
    "solve_inexact_hold_step": "solve_inexact",
    "nonfinite_candidate_hold_step": "nonfinite_candidate",
    "prior_only_hold_step": "prior_only_hold",
    "process_diagnostic_invalid_hold_step": "process_diagnostic_invalid",
    "process_guard_hold_step": "process_guard",
}
for column, reason in indicator_pairs.items():
    require(np.array_equal(bools(step[column]), reasons == reason), f"{column} indicator mismatch")
require(not bool(np.any(bools(step["jump_guard_hold_step"]))), "legacy jump guard fired")
require(not bool(np.any(reasons == "jump_guard")), "legacy jump-guard reason is present")
require(bool(np.all(np.isnan(jump_limit))), "legacy Euclidean jump limit is active")

bootstrap_positions = np.flatnonzero(reasons == "bootstrap_accept")
require(len(bootstrap_positions) == 1, "bootstrap acceptance must occur exactly once")
bootstrap_index = int(bootstrap_positions[0])
reliable_positions = np.flatnonzero(reliable)
accepted_positions = np.flatnonzero(accepted)
require(len(reliable_positions) > 0, "no reliable estimator step exists")
require(len(accepted_positions) > 0, "no accepted estimator step exists")
require(bootstrap_index == int(reliable_positions[0]) == int(accepted_positions[0]), "bootstrap is not the first reliable and accepted estimate")
require(not bool(np.any(prior_active[: bootstrap_index + 1])), "process prior was active before or during bootstrap")
require(bool(np.all(prior_active[bootstrap_index + 1 :])), "process prior was not active after bootstrap")
require(bool(np.all(step.loc[~prior_active, "estimator_mode"] == "measurement_bootstrap")), "bootstrap estimator mode mismatch")
require(bool(np.all(step.loc[~prior_active, "estimator_solver"] == "weighted_lstsq_svd")), "bootstrap solver mismatch")
require(bool(np.all(step.loc[prior_active, "estimator_mode"] == "q_prior_innovation")), "post-bootstrap estimator mode mismatch")
require(bool(np.all(step.loc[prior_active, "estimator_solver"] == "q_prior_augmented_lstsq_svd")), "post-bootstrap solver mismatch")

require(bool(np.allclose(rcond, EXPECTED_RCOND, rtol=0.0, atol=1e-18)), "estimator rcond mismatch")
require(bool(np.all((rank >= 0) & (rank <= N))), "estimator rank is outside range")
require(bool(np.all((measurement_rank >= 0) & (measurement_rank <= N))), "measurement rank is outside range")
require(bool(np.all((posterior_rank >= 0) & (posterior_rank <= N))), "posterior rank is outside range")
require(bool(np.all(effective_rows >= measurement_rank)), "effective rows are smaller than measurement rank")
require(bool(np.all(effective_rows == external_received + pseudo_received)), "effective-row accounting mismatch")
require(np.array_equal(rank, posterior_rank), "estimator and posterior rank disagree")
require(same_array(condition, posterior_condition), "estimator and posterior condition disagree")
require(same_array(numbers(step["estimator_singular_max"]), numbers(step["posterior_singular_max"])), "estimator and posterior singular maximum disagree")
require(same_array(numbers(step["estimator_singular_min"]), numbers(step["posterior_singular_min"])), "estimator and posterior singular minimum disagree")

expected_measurement_full_rank = (
    (measurement_rank == N)
    & np.isfinite(measurement_condition)
    & (measurement_condition <= 1.0 / rcond)
)
expected_posterior_reliable = (
    (posterior_rank == N)
    & np.isfinite(posterior_condition)
    & (posterior_condition <= 1.0 / rcond)
    & np.isfinite(candidate_norm)
)
require(np.array_equal(measurement_full_rank, expected_measurement_full_rank), "measurement full-rank flag mismatch")
require(np.array_equal(posterior_reliable, expected_posterior_reliable), "posterior reliable flag mismatch")
require(np.array_equal(reliable, posterior_reliable), "estimator and posterior reliability disagree")
require(bool(np.all(measurement_full_rank[~prior_active & reliable])), "a reliable bootstrap lacks measurement full rank")
require(bool(np.all(posterior_rank[prior_active] == N)), "a Q-prior posterior solve lacks full rank")
require(bool(np.all(posterior_reliable[prior_active])), "a Q-prior posterior solve is unreliable")

require(bool(np.all(external_total == EXTERNAL_TOTAL)), "external-total field mismatch")
require(bool(np.all((external_received >= 0) & (external_received <= EXTERNAL_TOTAL))), "external received count invalid")
require(bool(np.all((pseudo_received >= 0) & (pseudo_received <= PSEUDO_TOTAL))), "pseudo received count invalid")
require(bool(np.allclose(support_fraction, external_received / EXTERNAL_TOTAL, rtol=0.0, atol=1e-15)), "external support fraction mismatch")
external_present = external_received > 0
pseudo_only = (external_received == 0) & (pseudo_received > 0)
none_received = (external_received == 0) & (pseudo_received == 0)
require(np.array_equal(bools(step["external_support_present_step"]), external_present), "external-present flag mismatch")
require(np.array_equal(bools(step["pseudo_only_step"]), pseudo_only), "pseudo-only flag mismatch")
require(np.array_equal(bools(step["no_received_measurements_step"]), none_received), "no-received flag mismatch")
require(bool(np.all(external_present.astype(int) + pseudo_only.astype(int) + none_received.astype(int) == 1)), "support states are not exclusive")

require(bool(np.allclose(process_rms, process_rms_expected, rtol=1e-12, atol=1e-15)), "process RMS does not equal sqrt(trace(Q))")
require(bool(np.allclose(numbers(step["model_increment_scale"]), process_rms_expected, rtol=1e-12, atol=1e-15)), "model increment scale does not equal process RMS")
require(bool((step["process_guard_policy"] == "q_process_mahalanobis").all()), "process guard policy mismatch")
require(bool((step["jump_guard_policy"] == "q_process_mahalanobis").all()), "legacy guard-policy field mismatch")
require(bool(np.allclose(process_guard_limit, EXPECTED_GUARD_LIMIT, rtol=0.0, atol=1e-15)), "process guard limit mismatch")
require(same_array(process_guard_statistic, process_mahalanobis), "process guard statistic is not the Q Mahalanobis norm")

finite_prior_process = prior_active & np.isfinite(process_increment_norm) & np.isfinite(process_mahalanobis)
require(bool(np.all(finite_prior_process[reliable & prior_active])), "a reliable prior update lacks finite process diagnostics")
require(bool(np.all(process_increment_norm[finite_prior_process] >= 0.0)), "negative process increment norm")
require(bool(np.all(process_mahalanobis[finite_prior_process] >= 0.0)), "negative process Mahalanobis norm")
require(same_array(jump_norm[finite_prior_process], process_increment_norm[finite_prior_process], rtol=1e-9, atol=1e-10), "state jump norm does not equal Q-prior increment norm")

q_min_scale = math.sqrt(float(q_eigenvalues[0]))
q_max_scale = math.sqrt(float(q_eigenvalues[-1]))
mahal = process_mahalanobis[finite_prior_process]
increments = process_increment_norm[finite_prior_process]
require(bool(np.all(increments + 1e-12 >= q_min_scale * mahal)), "process increment violates the lower Q spectral bound")
require(bool(np.all(increments <= q_max_scale * mahal + 1e-12)), "process increment violates the upper Q spectral bound")

ordinary_accept = reasons == "accepted"
process_guard = reasons == "process_guard"
prior_only_hold = reasons == "prior_only_hold"
diagnostic_invalid = reasons == "process_diagnostic_invalid"
require(bool(np.all(prior_active[ordinary_accept] & external_present[ordinary_accept] & reliable[ordinary_accept])), "ordinary acceptance lacks a reliable network-supported prior")
require(bool(np.all(np.isfinite(process_mahalanobis[ordinary_accept]))), "ordinary acceptance has invalid Mahalanobis statistic")
require(bool(np.all(process_mahalanobis[ordinary_accept] <= process_guard_limit[ordinary_accept] + 1e-12)), "ordinary acceptance exceeded the process guard")
require(bool(np.all(prior_active[process_guard] & external_present[process_guard] & reliable[process_guard])), "process-guard hold lacks a reliable network-supported prior")
require(bool(np.all(process_mahalanobis[process_guard] > process_guard_limit[process_guard])), "process guard fired at or below its limit")
require(bool(np.all(prior_active[prior_only_hold] & pseudo_only[prior_only_hold] & reliable[prior_only_hold])), "prior-only hold logic mismatch")
if bool(np.any(diagnostic_invalid)):
    invalid_diagnostic = (
        ~np.isfinite(process_increment_norm)
        | ~np.isfinite(process_mahalanobis)
        | ~prior_active
    )
    require(bool(np.all(external_present[diagnostic_invalid] & reliable[diagnostic_invalid] & invalid_diagnostic[diagnostic_invalid])), "process-diagnostic-invalid hold logic mismatch")
require(bool(np.all(reasons[~reliable] == "solve_inexact")), "unreliable solve reason mismatch")

for index in range(STEPS - 1):
    expected_next = candidate_norm[index] if accepted[index] else previous_norm[index]
    require(
        same_float(previous_norm[index + 1], expected_next, rtol=1e-9, atol=1e-9),
        f"state-norm transition mismatch after step {index}",
    )

with np.load(truth_path, allow_pickle=False) as truth:
    x_true = np.asarray(truth["x_true"], dtype=float)
require(x_true.shape == (STEPS, N), "truth state shape mismatch")
truth_norm = np.linalg.norm(x_true, axis=1)
require(bool(np.all(np.isfinite(truth_norm) & (truth_norm > 0.0))), "truth norms are invalid")
reliable_ratio = candidate_norm[reliable] / truth_norm[reliable]
accepted_ratio = candidate_norm[accepted] / truth_norm[accepted]
bootstrap_ratio = float(candidate_norm[bootstrap_index] / truth_norm[bootstrap_index])

reason_counts = {
    str(reason): int(np.sum(reasons == reason))
    for reason in sorted(set(reasons))
}
summary = {
    "steps": STEPS,
    "events": EVENTS,
    "bootstrap_step_index": int(step_index[bootstrap_index]),
    "reliable_count": int(np.sum(reliable)),
    "reliable_fraction": float(np.mean(reliable)),
    "accepted_count": int(np.sum(accepted)),
    "accepted_fraction": float(np.mean(accepted)),
    "held_count": int(np.sum(held)),
    "hold_reason_counts": reason_counts,
    "measurement_full_rank_fraction": float(np.mean(measurement_full_rank)),
    "posterior_reliable_fraction": float(np.mean(posterior_reliable)),
    "process_prior_active_fraction": float(np.mean(prior_active)),
    "process_guard_count": int(np.sum(process_guard)),
    "prior_only_hold_count": int(np.sum(prior_only_hold)),
    "process_diagnostic_invalid_count": int(np.sum(diagnostic_invalid)),
    "process_rms_increment": process_rms_expected,
    "process_increment_mahalanobis_p99": float(np.quantile(process_mahalanobis[finite_prior_process], 0.99)),
    "process_increment_mahalanobis_max": float(np.max(process_mahalanobis[finite_prior_process])),
    "bootstrap_norm_ratio": bootstrap_ratio,
    "accepted_norm_ratio_median": float(np.median(accepted_ratio)),
    "accepted_norm_ratio_p99": float(np.quantile(accepted_ratio, 0.99)),
    "accepted_norm_ratio_max": float(np.max(accepted_ratio)),
    "reliable_norm_ratio_max": float(np.max(reliable_ratio)),
    "q_eigenvalue_min": float(q_eigenvalues[0]),
    "q_eigenvalue_max": float(q_eigenvalues[-1]),
    "external_support_present_fraction": float(np.mean(external_present)),
    "pseudo_only_fraction": float(np.mean(pseudo_only)),
}
print("MECHANICAL_SUMMARY=" + json.dumps(summary, sort_keys=True, separators=(",", ":")))

# Broad state-scale gates were fixed before reading this V5.2 cell.
require(float(np.mean(reliable)) > 0.5, "fewer than half of oracle steps are solver-reliable")
require(float(np.mean(accepted)) > 0.5, "fewer than half of oracle steps update state")
require(0.25 <= bootstrap_ratio <= 4.0, "bootstrap state norm is physically scale-inconsistent")
require(0.5 <= float(np.median(accepted_ratio)) <= 2.0, "median accepted state norm is scale-inconsistent")
require(float(np.quantile(accepted_ratio, 0.99)) <= 5.0, "accepted state-norm p99 exceeds broad mechanical bound")
require(float(np.max(accepted_ratio)) <= 10.0, "accepted state norm exceeds broad mechanical bound")
require(float(np.max(reliable_ratio)) <= 10.0, "solver-reliable candidate norm exceeds broad mechanical bound")

# Reconstruct the mechanical event aggregation; outcome columns remain excluded.
for event_id, block in step.groupby("event_id", sort=True):
    row = event.iloc[int(event_id)]
    first = block.iloc[0]
    require(int(row["event_id"]) == int(event_id), f"event row mismatch {event_id}")
    require(int(row["steps"]) == len(block) == STEPS_PER_EVENT, f"event step count mismatch {event_id}")
    pairs = {
        "held": float(bools(block["held"]).mean()),
        "solve_exact_fraction": float(bools(block["solve_exact"]).mean()),
        "estimator_reliable_fraction": float(bools(block["estimator_reliable"]).mean()),
        "estimator_rcond": float(numbers(block["estimator_rcond"])[0]),
        "estimator_effective_rows_min": float(np.min(numbers(block["estimator_effective_rows"]))),
        "estimator_effective_rows_max": float(np.max(numbers(block["estimator_effective_rows"]))),
        "estimator_rank_min": float(np.min(numbers(block["estimator_rank"]))),
        "estimator_condition_max": aggregate_numeric(block["estimator_condition"], "max"),
        "estimator_singular_max_max": aggregate_numeric(block["estimator_singular_max"], "max"),
        "estimator_singular_min_min": aggregate_numeric(block["estimator_singular_min"], "min"),
        "estimator_residual_norm_max": aggregate_numeric(block["estimator_residual_norm"], "max"),
        "estimator_augmented_residual_norm_max": aggregate_numeric(block["estimator_augmented_residual_norm"], "max"),
        "measurement_rank_min": float(np.min(numbers(block["measurement_rank"]))),
        "measurement_condition_max": aggregate_numeric(block["measurement_condition"], "max"),
        "measurement_full_rank_fraction": float(bools(block["measurement_full_rank"]).mean()),
        "posterior_rank_min": float(np.min(numbers(block["posterior_rank"]))),
        "posterior_condition_max": aggregate_numeric(block["posterior_condition"], "max"),
        "posterior_reliable_fraction": float(bools(block["posterior_reliable"]).mean()),
        "state_update_accepted_fraction": float(bools(block["state_update_accepted_step"]).mean()),
        "bootstrap_accept_fraction": float(bools(block["bootstrap_accept_step"]).mean()),
        "solve_inexact_hold_fraction": float(bools(block["solve_inexact_hold_step"]).mean()),
        "nonfinite_candidate_hold_fraction": float(bools(block["nonfinite_candidate_hold_step"]).mean()),
        "jump_guard_hold_fraction": float(bools(block["jump_guard_hold_step"]).mean()),
        "prior_only_hold_fraction": float(bools(block["prior_only_hold_step"]).mean()),
        "process_diagnostic_invalid_hold_fraction": float(bools(block["process_diagnostic_invalid_hold_step"]).mean()),
        "process_guard_hold_fraction": float(bools(block["process_guard_hold_step"]).mean()),
        "pseudo_only_fraction": float(bools(block["pseudo_only_step"]).mean()),
        "external_support_present_fraction": float(bools(block["external_support_present_step"]).mean()),
        "no_received_measurements_fraction": float(bools(block["no_received_measurements_step"]).mean()),
        "candidate_norm_max": aggregate_numeric(block["candidate_norm"], "max"),
        "previous_norm_max": aggregate_numeric(block["previous_norm"], "max"),
        "jump_norm_max": aggregate_numeric(block["jump_norm"], "max"),
        "jump_limit_min": aggregate_numeric(block["jump_limit"], "min"),
        "model_increment_scale": float(numbers(block["model_increment_scale"])[0]),
        "process_prior_active_fraction": float(bools(block["process_prior_active"]).mean()),
        "process_rms_increment": float(numbers(block["process_rms_increment"])[0]),
        "process_increment_norm_max": aggregate_numeric(block["process_increment_norm"], "max"),
        "process_increment_mahalanobis_max": aggregate_numeric(block["process_increment_mahalanobis"], "max"),
        "process_guard_statistic_max": aggregate_numeric(block["process_guard_statistic"], "max"),
        "process_guard_limit": float(numbers(block["process_guard_limit"])[0]),
        "external_received_count_min": float(np.min(numbers(block["external_received_count"]))),
        "external_total": float(numbers(block["external_total"])[0]),
        "external_support_fraction_min": float(np.min(numbers(block["external_support_fraction"]))),
        "pseudo_received_count_max": float(np.max(numbers(block["pseudo_received_count"]))),
    }
    for column, expected in pairs.items():
        require(same_float(row[column], expected), f"event aggregation mismatch event={event_id} column={column}")
    require(bool(row["held_any"]) == bool(bools(block["held"]).any()), f"held_any mismatch {event_id}")
    require(bool(row["solve_exact_all"]) == bool(bools(block["solve_exact"]).all()), f"solve_exact_all mismatch {event_id}")
    require(bool(row["estimator_reliable_all"]) == bool(bools(block["estimator_reliable"]).all()), f"estimator_reliable_all mismatch {event_id}")
    require(str(row["estimator_solver"]) == str(first["estimator_solver"]), f"event solver mismatch {event_id}")
    require(str(row["estimator_mode_first"]) == str(first["estimator_mode"]), f"event first mode mismatch {event_id}")
    require(str(row["estimator_mode_last"]) == str(block["estimator_mode"].iloc[-1]), f"event last mode mismatch {event_id}")
    require(str(row["jump_guard_policy"]) == str(first["jump_guard_policy"]), f"event legacy guard-policy field mismatch {event_id}")
    require(str(row["process_guard_policy"]) == str(first["process_guard_policy"]), f"event process guard-policy mismatch {event_id}")

report = {
    "schema": "paper1.v5_2.oracle.mechanical.audit.v1",
    "run_id": RUN_ID,
    "status": "pass",
    "mechanical_columns_read": {"step": STEP_COLUMNS, "event": EVENT_COLUMNS},
    "mechanical_summary": summary,
    "gates": {
        "frozen_provenance": True,
        "cell_manifest_integrity": True,
        "metadata_consistency": True,
        "step_and_event_sequence": True,
        "measurement_vs_posterior_rank_logic": True,
        "q_prior_activation": True,
        "q_process_spectral_bounds": True,
        "mahalanobis_guard_logic": True,
        "legacy_euclidean_guard_inactive": True,
        "state_norm_transition": True,
        "physical_state_scale": True,
        "external_pseudo_support_exclusivity": True,
        "diagnostic_event_aggregation": True,
    },
    "simulation_rerun": False,
    "implementation_modified": False,
    "full_campaign_authorized": False,
    "calibration_authorized": False,
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
}

raw_report_path = os.environ.get(
    "PAPER1_V5_2_AUDIT_REPORT",
    "/workspace/paper1_v5_2_oracle_mechanical_audit.json",
)
report_path = resolve_workspace_path(raw_report_path)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
if report_path.exists():
    require(report_path.read_bytes() == report_bytes, f"existing audit report conflicts: {report_path}")
else:
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    require(not temporary_path.exists(), f"temporary audit report already exists: {temporary_path}")
    temporary_path.write_bytes(report_bytes)
    temporary_path.replace(report_path)

print(f"REPORT={report_path}")
print(f"REPORT_SHA256={sha256(report_path).upper()}")
print("MECHANICAL_GATES_PASSED=13")
print("SIMULATION_RERUN=False")
print("IMPLEMENTATION_MODIFIED=False")
print("FULL_CAMPAIGN_AUTHORIZED=False")
print("CALIBRATION_AUTHORIZED=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_2_ORACLE_MECHANICAL_AUDIT_OK")
