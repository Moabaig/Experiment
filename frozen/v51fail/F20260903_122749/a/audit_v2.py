from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


ROOT = Path("/workspace")
RUN_ID = "paper1_v5_1mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID

N = 491
STEPS = 13200
EVENTS = 1100
STEPS_PER_EVENT = 12
EXTERNAL_TOTAL = 45
HOLD_FACTOR = 50.0
MODEL_INCREMENT_SCALE = 0.02
EXPECTED_JUMP_LIMIT = HOLD_FACTOR * math.sqrt(N) * MODEL_INCREMENT_SCALE
EXPECTED_RCOND = math.sqrt(np.finfo(float).eps)
EXPECTED_SOLVER = "weighted_lstsq_svd"
EXPECTED_GUARD_POLICY = "fixed_model_increment"
EXPECTED_TRUST = "0a2627bdaacad03e582bb039eeb2fb3ac73d33d20b77e96881ebceec64aae437"
EXPECTED_TWIN = "39e6729af233032ab9c58851c9682252f02d36eed739eb2ec769e165659da34c"

STEP_COLUMNS = [
    "step_index", "event_id", "step_in_event", "bandwidth_level",
    "bandwidth_cap_bps", "held", "solve_exact", "estimator_reliable",
    "estimator_solver", "estimator_rcond", "estimator_effective_rows",
    "estimator_rank", "estimator_condition", "estimator_singular_max",
    "estimator_singular_min", "estimator_residual_norm", "hold_reason",
    "state_update_accepted_step", "bootstrap_accept_step",
    "solve_inexact_hold_step", "nonfinite_candidate_hold_step",
    "jump_guard_hold_step", "candidate_norm", "previous_norm", "jump_norm",
    "jump_limit", "jump_guard_policy", "model_increment_scale",
    "external_received_count", "external_total", "external_support_fraction",
    "pseudo_received_count", "pseudo_only_step",
    "external_support_present_step", "no_received_measurements_step",
]

EVENT_COLUMNS = [
    "event_id", "bandwidth_level", "bandwidth_cap_bps", "held", "held_any",
    "solve_exact_fraction", "solve_exact_all", "estimator_reliable_fraction",
    "estimator_reliable_all", "estimator_solver", "estimator_rcond",
    "estimator_effective_rows_min", "estimator_effective_rows_max",
    "estimator_rank_min", "estimator_condition_max",
    "estimator_singular_max_max", "estimator_singular_min_min",
    "estimator_residual_norm_max", "state_update_accepted_fraction",
    "bootstrap_accept_fraction", "solve_inexact_hold_fraction",
    "nonfinite_candidate_hold_fraction", "jump_guard_hold_fraction",
    "pseudo_only_fraction", "external_support_present_fraction",
    "no_received_measurements_fraction", "candidate_norm_max",
    "previous_norm_max", "jump_norm_max", "jump_limit_min",
    "jump_guard_policy", "model_increment_scale", "external_received_count_min",
    "external_total", "external_support_fraction_min",
    "pseudo_received_count_max", "steps",
]


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bools(series: pd.Series) -> np.ndarray:
    return series.astype(bool).to_numpy()


def numbers(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def same_float(left: float, right: float, *, rtol=1e-10, atol=1e-12) -> bool:
    left = float(left)
    right = float(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return bool(math.isclose(left, right, rel_tol=rtol, abs_tol=atol))


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


print("PAPER1_V5_1_ORACLE_MECHANICAL_AUDIT")
print("RUN_ID=", RUN_ID)
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")

trust_path = ROOT / "trust_metric.py"
twin_path = ROOT / "twin_fed.py"
record_path = RUN_ROOT / "cell_record.paper1.v5_1.mechanical.json"
step_path = RUN_ROOT / "twin" / "scores.parquet"
event_path = RUN_ROOT / "twin" / "scores_events.parquet"

require(sha256(trust_path) == EXPECTED_TRUST, "trust_metric.py hash mismatch")
require(sha256(twin_path) == EXPECTED_TWIN, "twin_fed.py hash mismatch")

record = read_json(record_path)
require(record["schema"] == "twin.factor.cell.record.paper1.v5_1.mechanical", "cell schema mismatch")
require(record["run_id"] == RUN_ID, "cell run ID mismatch")
require(record["status"] == "complete", "cell status is not complete")
require(record["estimator_version"] == "paper1_v5_1", "estimator version mismatch")
require(record["estimator_solver"] == EXPECTED_SOLVER, "record solver mismatch")
require(record["jump_guard_policy"] == EXPECTED_GUARD_POLICY, "record guard policy mismatch")
require(record["normal_equations_used"] is False, "record says normal equations were used")
require(record["mechanical_validation_only"] is True, "cell is not mechanical-only")
require(record["performance_outcomes_inspected"] is False, "record says performance was inspected")
require(record["trust_metric_sha256"] == EXPECTED_TRUST, "record trust hash mismatch")
require(record["twin_fed_sha256"] == EXPECTED_TWIN, "record twin hash mismatch")
require(record["bandwidth_level"] == "bw04_oracle", "record bandwidth level mismatch")
require(float(record["bandwidth_cap_bps"]) == 1.0e12, "record bandwidth cap mismatch")

truth_path = ROOT / record["truth_file"]
require(truth_path.is_file(), "truth file is missing")
require(sha256(truth_path) == record["truth_sha256"], "truth hash mismatch")

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
require(power_meta["inputs"]["truth"]["sha256"] == record["truth_sha256"], "power truth hash mismatch")
require(oracle_meta["inputs"]["truth"]["sha256"] == record["truth_sha256"], "oracle truth hash mismatch")
require(net_meta["bandwidth_level"] == "bw04_oracle", "network bandwidth mismatch")
require(float(net_meta["bandwidth_cap_bps"]) == 1.0e12, "network cap mismatch")
require(twin_meta["factor_design"]["bandwidth_level"] == "bw04_oracle", "twin bandwidth mismatch")
require(float(twin_meta["factor_design"]["bandwidth_cap_bps"]) == 1.0e12, "twin cap mismatch")

# Only the declared mechanical/provenance columns are read.
step = pd.read_parquet(step_path, columns=STEP_COLUMNS)
event = pd.read_parquet(event_path, columns=EVENT_COLUMNS)

require(len(step) == STEPS, "step parquet row count mismatch")
require(len(event) == EVENTS, "event parquet row count mismatch")
require(np.array_equal(step["step_index"].to_numpy(int), np.arange(STEPS)), "step_index sequence mismatch")
require(np.array_equal(step["event_id"].to_numpy(int), np.repeat(np.arange(EVENTS), STEPS_PER_EVENT)), "step event sequence mismatch")
require(np.array_equal(step["step_in_event"].to_numpy(int), np.tile(np.arange(STEPS_PER_EVENT), EVENTS)), "step_in_event sequence mismatch")
require(np.array_equal(event["event_id"].to_numpy(int), np.arange(EVENTS)), "event_id sequence mismatch")
require(bool((step["bandwidth_level"] == "bw04_oracle").all()), "step bandwidth level mismatch")
require(bool((event["bandwidth_level"] == "bw04_oracle").all()), "event bandwidth level mismatch")
require(bool(np.all(numbers(step["bandwidth_cap_bps"]) == 1.0e12)), "step bandwidth cap mismatch")
require(bool(np.all(numbers(event["bandwidth_cap_bps"]) == 1.0e12)), "event bandwidth cap mismatch")

held = bools(step["held"])
accepted = bools(step["state_update_accepted_step"])
reliable = bools(step["estimator_reliable"])
solve_exact = bools(step["solve_exact"])
reasons = step["hold_reason"].astype(str).to_numpy()
candidate_norm = numbers(step["candidate_norm"])
previous_norm = numbers(step["previous_norm"])
jump_norm = numbers(step["jump_norm"])
jump_limit = numbers(step["jump_limit"])
rcond = numbers(step["estimator_rcond"])
rank = step["estimator_rank"].to_numpy(int)
condition = numbers(step["estimator_condition"])
singular_max = numbers(step["estimator_singular_max"])
singular_min = numbers(step["estimator_singular_min"])
residual_norm = numbers(step["estimator_residual_norm"])
effective_rows = step["estimator_effective_rows"].to_numpy(int)
external_received = step["external_received_count"].to_numpy(int)
external_total = step["external_total"].to_numpy(int)
pseudo_received = step["pseudo_received_count"].to_numpy(int)
support_fraction = numbers(step["external_support_fraction"])

require(np.array_equal(accepted, ~held), "accepted flag does not equal not-held")
require(np.array_equal(reliable, solve_exact), "solve_exact and estimator_reliable disagree")
require(set(reasons).issubset({"accepted", "bootstrap_accept", "solve_inexact", "nonfinite_candidate", "jump_guard"}), "unexpected hold reason")
require(np.array_equal(bools(step["bootstrap_accept_step"]), reasons == "bootstrap_accept"), "bootstrap indicator mismatch")
require(np.array_equal(bools(step["solve_inexact_hold_step"]), reasons == "solve_inexact"), "solve-inexact indicator mismatch")
require(np.array_equal(bools(step["nonfinite_candidate_hold_step"]), reasons == "nonfinite_candidate"), "nonfinite indicator mismatch")
require(np.array_equal(bools(step["jump_guard_hold_step"]), reasons == "jump_guard"), "jump-guard indicator mismatch")
require(np.array_equal(accepted, np.isin(reasons, ["accepted", "bootstrap_accept"])), "decision reason and accepted flag disagree")
require(bool(np.all(held[~reliable])), "an unreliable estimate was accepted")
require(bool(np.all(reasons[~reliable] == "solve_inexact")), "unreliable solve reason mismatch")

bootstrap = np.flatnonzero(reasons == "bootstrap_accept")
require(len(bootstrap) == 1, "bootstrap acceptance must occur exactly once")
bootstrap_index = int(bootstrap[0])
reliable_positions = np.flatnonzero(reliable)
accepted_positions = np.flatnonzero(accepted)
require(len(reliable_positions) > 0, "no reliable estimator step exists")
require(len(accepted_positions) > 0, "no accepted estimator step exists")
first_reliable = int(reliable_positions[0])
first_accepted = int(accepted_positions[0])
require(
    bootstrap_index == first_reliable == first_accepted,
    "bootstrap must coincide with the first reliable and first accepted estimate",
)
require(
    not bool(np.any(reliable[:bootstrap_index]))
    and not bool(np.any(accepted[:bootstrap_index])),
    "a reliable or accepted estimate precedes bootstrap",
)
require(
    reliable[bootstrap_index] and accepted[bootstrap_index] and not held[bootstrap_index],
    "bootstrap estimate was not reliable and accepted",
)
require(
    same_float(previous_norm[bootstrap_index], 0.0),
    "bootstrap previous norm was not zero",
)

require(bool((step["estimator_solver"] == EXPECTED_SOLVER).all()), "solver name is not constant")
require(bool((step["jump_guard_policy"] == EXPECTED_GUARD_POLICY).all()), "guard policy is not constant")
require(bool(np.allclose(rcond, EXPECTED_RCOND, rtol=0.0, atol=1e-18)), "estimator rcond mismatch")
require(bool(np.allclose(numbers(step["model_increment_scale"]), MODEL_INCREMENT_SCALE, rtol=0.0, atol=1e-15)), "model increment scale mismatch")
require(bool(np.allclose(jump_limit, EXPECTED_JUMP_LIMIT, rtol=1e-12, atol=1e-12)), "fixed jump limit mismatch")

require(bool(np.all((rank >= 0) & (rank <= N))), "estimator rank is outside valid range")
require(bool(np.all(effective_rows >= rank)), "effective rows are smaller than rank")
require(bool(np.all(effective_rows == external_received + pseudo_received)), "effective-row accounting mismatch")
require(bool(np.all(rank[reliable] == N)), "a reliable solve lacks full numerical rank")
require(bool(np.all(np.isfinite(condition[reliable]))), "reliable condition number is nonfinite")
require(bool(np.all(condition[reliable] <= 1.0 / rcond[reliable])), "reliable condition exceeds solver threshold")
require(bool(np.all(np.isfinite(candidate_norm[reliable]))), "reliable candidate norm is nonfinite")
require(bool(np.all(np.isfinite(singular_max[reliable]) & (singular_max[reliable] > 0.0))), "reliable singular maximum invalid")
require(bool(np.all(np.isfinite(singular_min[reliable]) & (singular_min[reliable] > 0.0))), "reliable singular minimum invalid")
require(bool(np.all(np.isfinite(residual_norm[reliable]) & (residual_norm[reliable] >= 0.0))), "reliable residual norm invalid")

require(bool(np.all(external_total == EXTERNAL_TOTAL)), "external-total field mismatch")
require(bool(np.all((external_received >= 0) & (external_received <= EXTERNAL_TOTAL))), "external received count invalid")
require(bool(np.all(pseudo_received >= 0)), "pseudo received count invalid")
require(bool(np.allclose(support_fraction, external_received / EXTERNAL_TOTAL, rtol=0.0, atol=1e-15)), "external support fraction mismatch")

expected_external_present = external_received > 0
expected_pseudo_only = (external_received == 0) & (pseudo_received > 0)
expected_none = (external_received == 0) & (pseudo_received == 0)
require(np.array_equal(bools(step["external_support_present_step"]), expected_external_present), "external-present flag mismatch")
require(np.array_equal(bools(step["pseudo_only_step"]), expected_pseudo_only), "pseudo-only flag mismatch")
require(np.array_equal(bools(step["no_received_measurements_step"]), expected_none), "no-received flag mismatch")
require(bool(np.all(expected_external_present.astype(int) + expected_pseudo_only.astype(int) + expected_none.astype(int) == 1)), "support states are not exclusive")

jump_guard = reasons == "jump_guard"
ordinary_accept = reasons == "accepted"
require(bool(np.all(np.isfinite(jump_norm[jump_guard]))), "jump-guard norm is nonfinite")
require(bool(np.all(jump_norm[jump_guard] > jump_limit[jump_guard])), "jump guard fired below its limit")
require(bool(np.all(jump_norm[ordinary_accept] <= jump_limit[ordinary_accept] + 1e-10)), "ordinary acceptance exceeded jump limit")

for index in range(STEPS - 1):
    expected_next = candidate_norm[index] if accepted[index] else previous_norm[index]
    require(same_float(previous_norm[index + 1], expected_next, rtol=1e-9, atol=1e-9), f"state-norm transition mismatch after step {index}")

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
    "reliable_count": int(np.sum(reliable)),
    "reliable_fraction": float(np.mean(reliable)),
    "accepted_count": int(np.sum(accepted)),
    "accepted_fraction": float(np.mean(accepted)),
    "held_count": int(np.sum(held)),
    "hold_reason_counts": reason_counts,
    "bootstrap_position": bootstrap_index,
    "bootstrap_step_index": int(step_index[bootstrap_index]),
    "prebootstrap_solve_inexact_count": int(np.sum(reasons[:bootstrap_index] == "solve_inexact")),
    "bootstrap_candidate_norm": float(candidate_norm[bootstrap_index]),
    "bootstrap_truth_norm": float(truth_norm[bootstrap_index]),
    "bootstrap_norm_ratio": bootstrap_ratio,
    "truth_norm_min": float(np.min(truth_norm)),
    "truth_norm_median": float(np.median(truth_norm)),
    "truth_norm_max": float(np.max(truth_norm)),
    "reliable_candidate_norm_min": float(np.min(candidate_norm[reliable])),
    "reliable_candidate_norm_median": float(np.median(candidate_norm[reliable])),
    "reliable_candidate_norm_max": float(np.max(candidate_norm[reliable])),
    "accepted_norm_ratio_median": float(np.median(accepted_ratio)),
    "accepted_norm_ratio_p99": float(np.quantile(accepted_ratio, 0.99)),
    "accepted_norm_ratio_max": float(np.max(accepted_ratio)),
    "reliable_norm_ratio_max": float(np.max(reliable_ratio)),
    "rank_min": int(np.min(rank)),
    "rank_max": int(np.max(rank)),
    "reliable_rank_min": int(np.min(rank[reliable])),
    "reliable_condition_max": float(np.max(condition[reliable])),
    "fixed_jump_limit": float(jump_limit[0]),
    "jump_guard_count": int(np.sum(jump_guard)),
    "external_support_present_fraction": float(np.mean(expected_external_present)),
    "pseudo_only_fraction": float(np.mean(expected_pseudo_only)),
    "no_received_measurements_fraction": float(np.mean(expected_none)),
}

print("MECHANICAL_SUMMARY=", json.dumps(summary, sort_keys=True, separators=(",", ":")))

# Broad mechanical scale gates, fixed before inspecting these values.
require(float(np.mean(reliable)) > 0.5, "fewer than half of oracle steps are solver-reliable")
require(float(np.mean(accepted)) > 0.5, "fewer than half of oracle steps update state")
require(0.25 <= bootstrap_ratio <= 4.0, "bootstrap state norm is physically scale-inconsistent")
require(0.5 <= float(np.median(accepted_ratio)) <= 2.0, "median accepted state norm is scale-inconsistent")
require(float(np.quantile(accepted_ratio, 0.99)) <= 5.0, "accepted state-norm p99 exceeds broad mechanical bound")
require(float(np.max(accepted_ratio)) <= 10.0, "accepted state norm exceeds broad mechanical bound")
require(float(np.max(reliable_ratio)) <= 10.0, "solver-reliable candidate norm exceeds broad mechanical bound")

# Reconstruct diagnostic event aggregation exactly; outcome columns are excluded.
for event_id, block in step.groupby("event_id", sort=True):
    row = event.iloc[int(event_id)]
    require(int(row["event_id"]) == int(event_id), f"event row mismatch {event_id}")
    require(int(row["steps"]) == len(block) == STEPS_PER_EVENT, f"event step count mismatch {event_id}")
    pairs = {
        "held": float(bools(block["held"]).mean()),
        "solve_exact_fraction": float(bools(block["solve_exact"]).mean()),
        "estimator_reliable_fraction": float(bools(block["estimator_reliable"]).mean()),
        "state_update_accepted_fraction": float(bools(block["state_update_accepted_step"]).mean()),
        "bootstrap_accept_fraction": float(bools(block["bootstrap_accept_step"]).mean()),
        "solve_inexact_hold_fraction": float(bools(block["solve_inexact_hold_step"]).mean()),
        "nonfinite_candidate_hold_fraction": float(bools(block["nonfinite_candidate_hold_step"]).mean()),
        "jump_guard_hold_fraction": float(bools(block["jump_guard_hold_step"]).mean()),
        "pseudo_only_fraction": float(bools(block["pseudo_only_step"]).mean()),
        "external_support_present_fraction": float(bools(block["external_support_present_step"]).mean()),
        "no_received_measurements_fraction": float(bools(block["no_received_measurements_step"]).mean()),
        "estimator_rcond": float(numbers(block["estimator_rcond"])[0]),
        "estimator_effective_rows_min": float(np.min(numbers(block["estimator_effective_rows"]))),
        "estimator_effective_rows_max": float(np.max(numbers(block["estimator_effective_rows"]))),
        "estimator_rank_min": float(np.min(numbers(block["estimator_rank"]))),
        "estimator_condition_max": aggregate_numeric(block["estimator_condition"], "max"),
        "estimator_singular_max_max": aggregate_numeric(block["estimator_singular_max"], "max"),
        "estimator_singular_min_min": aggregate_numeric(block["estimator_singular_min"], "min"),
        "estimator_residual_norm_max": aggregate_numeric(block["estimator_residual_norm"], "max"),
        "candidate_norm_max": aggregate_numeric(block["candidate_norm"], "max"),
        "previous_norm_max": aggregate_numeric(block["previous_norm"], "max"),
        "jump_norm_max": aggregate_numeric(block["jump_norm"], "max"),
        "jump_limit_min": aggregate_numeric(block["jump_limit"], "min"),
        "model_increment_scale": float(numbers(block["model_increment_scale"])[0]),
        "external_received_count_min": float(np.min(numbers(block["external_received_count"]))),
        "external_total": float(numbers(block["external_total"])[0]),
        "external_support_fraction_min": float(np.min(numbers(block["external_support_fraction"]))),
        "pseudo_received_count_max": float(np.max(numbers(block["pseudo_received_count"]))),
    }
    for column, expected in pairs.items():
        require(same_float(row[column], expected, rtol=1e-9, atol=1e-12), f"event aggregation mismatch event={event_id} column={column}")
    require(bool(row["held_any"]) == bool(bools(block["held"]).any()), f"held_any mismatch {event_id}")
    require(bool(row["solve_exact_all"]) == bool(bools(block["solve_exact"]).all()), f"solve_exact_all mismatch {event_id}")
    require(bool(row["estimator_reliable_all"]) == bool(bools(block["estimator_reliable"]).all()), f"estimator_reliable_all mismatch {event_id}")
    require(str(row["estimator_solver"]) == EXPECTED_SOLVER, f"event solver mismatch {event_id}")
    require(str(row["jump_guard_policy"]) == EXPECTED_GUARD_POLICY, f"event guard policy mismatch {event_id}")

report = {
    "schema": "paper1.v5_1.oracle.mechanical.audit.v1",
    "run_id": RUN_ID,
    "status": "pass",
    "mechanical_summary": summary,
    "gates": {
        "provenance": True,
        "metadata": True,
        "step_sequence": True,
        "solver_reliability_logic": True,
        "fixed_guard_logic": True,
        "state_norm_transition": True,
        "physical_state_scale": True,
        "external_pseudo_support_exclusivity": True,
        "diagnostic_event_aggregation": True,
    },
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
    "full_campaign_authorized": False,
    "calibration_authorized": False,
}

report_path = ROOT / "paper1_v5_1_oracle_mechanical_audit.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("REPORT=", report_path)
print("REPORT_SHA256=", sha256(report_path))
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("FULL_CAMPAIGN_AUTHORIZED=False")
print("CALIBRATION_AUTHORIZED=False")
print("PAPER1_V5_1_ORACLE_MECHANICAL_AUDIT_OK")
