from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/workspace")
RUN_ID = "paper1_v5_2mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID
MECHANICAL_EVIDENCE_ROOT = ROOT / "frozen" / "v52audit" / "A20260903_223859"

EXPECTED_EVENTS = 1100
EXPECTED_STEPS = 13200
EXPECTED_STEPS_PER_EVENT = 12
EXPECTED_GUARD_STEPS = 3184
EXPECTED_ACCEPTED_STEPS = 10004
EXPECTED_POSTERIOR_RELIABLE_STEPS = 13188
EXPECTED_MEASUREMENT_FULL_RANK_STEPS = 10236
EXPECTED_GUARD_LIMIT = 50.0

EXPECTED_CONTRACT = "bd6489ea5b14d78b8825d48f129faf440f7fa3dfc46db118d74a5b5999ea43ae"
EXPECTED_TRUST = "936dd373a2d8a2f0b905604ca4c3de61ec2cc889ba233aa150a24f44f2926fe6"
EXPECTED_TWIN = "9cd9ffaa32dcfe2f12ed161a8d62d2d97b2ab0b4d462fda0e97e7f46572043a4"
EXPECTED_ENV = "55a4fcb1acb19d86cbe2da4bcc4fe814170a14a5a637ec6cec97d9c94195d694"
EXPECTED_TWIN_EVENTS = "8bd205aa1f59af3ceb19c83da260ef43366c77ecf30919337b92a398ccb14447"
EXPECTED_ORACLE_EVENTS = "b754cc1caff4e2d67604d89d53f8d76aa54681387b4188d7a070b835c2fd74d8"
EXPECTED_MECHANICAL_REPORT = "bdb0b93bf7312b10549456e7646812c24ff6e6b00ab9fbd459145b306a8e1f24"
EXPECTED_MECHANICAL_LOG = "51792e77faa1952a4d3eb1815aa76374fce073db5ea551bf4c6dabb9df77ad4c"
EXPECTED_MECHANICAL_GATE_RECORD = "15e1126efd9094a9de57d75c24c282085f49587ee3dcbabb2eafa3ab961cb313"
EXPECTED_MECHANICAL_MANIFEST = "ab06c8bde10825edb362f4e90ba44bed838ab13030f3ae249f8eb4732399b677"

TWIN_EVENT_COLUMNS = [
    "event_id", "arm", "regime", "stratum", "alarm", "s", "chi2",
    "sB1", "sB2", "b1", "b2", "held", "held_any",
    "residual_available", "n_rx_telemetry",
    "state_update_accepted_fraction", "measurement_full_rank_fraction",
    "posterior_reliable_fraction", "process_guard_hold_fraction",
    "process_guard_limit", "process_increment_mahalanobis_max", "steps",
]

ORACLE_EVENT_COLUMNS = [
    "event_id", "label", "is_nominal", "drift_family", "d", "steps",
]

AUTHORIZED_METRICS = ["s", "chi2", "sB1", "sB2"]


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def empirical_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=float)
    finite = np.isfinite(values)
    require(bool(np.all(finite)), "AUC score contains nonfinite values")
    positives = int(np.sum(y))
    negatives = int(np.sum(~y))
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    numerator = float(np.sum(ranks[y]) - positives * (positives + 1) / 2.0)
    return numerator / (positives * negatives)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)


def finite_quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
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


def normalized_group_value(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return "NA"
    return str(value)


def summarize_block(block: pd.DataFrame) -> dict[str, Any]:
    labels = block["label"].astype(bool).to_numpy()
    nominal = block["is_nominal"].astype(bool).to_numpy()
    alarm = block["alarm"].astype(bool).to_numpy()
    guard_fraction = numeric(block["process_guard_hold_fraction"])
    accepted_fraction = numeric(block["state_update_accepted_fraction"])
    guard_any = guard_fraction > 0.0
    steps = numeric(block["steps"])
    guard_steps = float(np.sum(guard_fraction * steps))
    return {
        "events": int(len(block)),
        "oracle_positive_events": int(np.sum(labels)),
        "label_negative_events": int(np.sum(~labels)),
        "nominal_events": int(np.sum(nominal)),
        "subthreshold_non_nominal_events": int(np.sum(~labels & ~nominal)),
        "guard_any_events": int(np.sum(guard_any)),
        "guard_any_event_rate": float(np.mean(guard_any)),
        "guard_step_equivalents": guard_steps,
        "guard_step_fraction": float(guard_steps / np.sum(steps)),
        "mean_state_update_accepted_fraction": float(np.mean(accepted_fraction)),
        "mean_held_fraction": float(np.mean(numeric(block["held"]))),
        "mean_measurement_full_rank_fraction": float(
            np.mean(numeric(block["measurement_full_rank_fraction"]))
        ),
        "mean_posterior_reliable_fraction": float(
            np.mean(numeric(block["posterior_reliable_fraction"]))
        ),
        "active_alarm_rate": float(np.mean(alarm)),
        "active_alarm_recall": (
            float(np.mean(alarm[labels])) if bool(np.any(labels)) else None
        ),
        "active_alarm_label_negative_rate": (
            float(np.mean(alarm[~labels])) if bool(np.any(~labels)) else None
        ),
        "active_alarm_nominal_rate": (
            float(np.mean(alarm[nominal])) if bool(np.any(nominal)) else None
        ),
        "mean_b1": float(np.mean(numeric(block["b1"]))),
        "mean_b2": float(np.mean(numeric(block["b2"]))),
        "mahalanobis_event_max": finite_quantiles(
            numeric(block["process_increment_mahalanobis_max"])
        ),
    }


def grouped_summaries(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        tagged = frame.assign(_group=frame[column].map(normalized_group_value))
        for value, block in tagged.groupby("_group", sort=True, dropna=False):
            rows.append(
                {
                    "dimension": column,
                    "value": str(value),
                    **summarize_block(block),
                }
            )
    tagged = frame.assign(
        _group=[
            f"arm={normalized_group_value(arm)}|regime={normalized_group_value(regime)}"
            for arm, regime in zip(frame["arm"], frame["regime"])
        ]
    )
    for value, block in tagged.groupby("_group", sort=True, dropna=False):
        rows.append(
            {
                "dimension": "arm_x_regime",
                "value": str(value),
                **summarize_block(block),
            }
        )
    return rows


def auc_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    for column in ("arm", "regime", "drift_family"):
        tagged = frame.assign(_group=frame[column].map(normalized_group_value))
        for value, block in tagged.groupby("_group", sort=True, dropna=False):
            group_specs.append((column, str(value), block))
    for dimension, value, block in group_specs:
        labels = block["label"].astype(bool).to_numpy()
        positives = int(np.sum(labels))
        negatives = int(np.sum(~labels))
        for metric in AUTHORIZED_METRICS:
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "metric": metric,
                    "events": int(len(block)),
                    "positives": positives,
                    "negatives": negatives,
                    "empirical_auc": empirical_auc(labels, numeric(block[metric])),
                    "hypothesis_test_performed": False,
                }
            )
    return rows


print("PAPER1_V5_2_ORACLE_GUARD_IMPACT_SENTINEL")
print(f"RUN_ID={RUN_ID}")
print("SINGLE_CELL_EXPLORATORY_ONLY=True")
print("EVENT_LEVEL_ONLY=True")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=True")
print("PERFORMANCE_OUTCOMES_INSPECTED=True")
print("CALIBRATION_AUTHORIZED=False")
print("PARAMETER_TUNING_AUTHORIZED=False")
print("FULL_CAMPAIGN_AUTHORIZED=False")

contract_path = ROOT / "paper1_v5_2_oracle_guard_impact_sentinel_contract.json"
trust_path = ROOT / "trust_metric.py"
twin_path = ROOT / "twin_fed.py"
env_path = ROOT / ".env"
twin_event_path = RUN_ROOT / "twin" / "scores_events.parquet"
oracle_event_path = RUN_ROOT / "oracle" / "oracle_events.parquet"
mechanical_report_path = MECHANICAL_EVIDENCE_ROOT / "paper1_v5_2_oracle_mechanical_audit.json"
mechanical_log_path = MECHANICAL_EVIDENCE_ROOT / "paper1_v5_2_oracle_mechanical_audit.log"
mechanical_gate_record_path = MECHANICAL_EVIDENCE_ROOT / "paper1_v5_2_oracle_mechanical_audit_gate.json"
mechanical_manifest_path = MECHANICAL_EVIDENCE_ROOT / "PAPER1_V5_2_MECHANICAL_AUDIT_SHA256.csv"

for path, expected in (
    (contract_path, EXPECTED_CONTRACT),
    (trust_path, EXPECTED_TRUST),
    (twin_path, EXPECTED_TWIN),
    (env_path, EXPECTED_ENV),
    (twin_event_path, EXPECTED_TWIN_EVENTS),
    (oracle_event_path, EXPECTED_ORACLE_EVENTS),
    (mechanical_report_path, EXPECTED_MECHANICAL_REPORT),
    (mechanical_log_path, EXPECTED_MECHANICAL_LOG),
    (mechanical_gate_record_path, EXPECTED_MECHANICAL_GATE_RECORD),
    (mechanical_manifest_path, EXPECTED_MECHANICAL_MANIFEST),
):
    require_hash(path, expected)

contract = read_json(contract_path)
require(
    contract["schema"]
    == "paper1.v5_2.oracle_guard_impact_sentinel_contract.v1",
    "sentinel contract schema mismatch",
)
require(contract["authorized_run_id"] == RUN_ID, "sentinel contract run mismatch")
require(contract["authorized_cell_count"] == 1, "sentinel contract cell count mismatch")
authorized_inputs = contract["authorized_inputs"]
require(
    authorized_inputs["twin_event_file"]["columns"] == TWIN_EVENT_COLUMNS,
    "twin event allowlist conflicts with contract",
)
require(
    authorized_inputs["oracle_event_file"]["columns"] == ORACLE_EVENT_COLUMNS,
    "oracle event allowlist conflicts with contract",
)
authorization = contract["authorization"]
require(authorization["existing_outputs_only"] is True, "contract permits new outputs")
require(authorization["event_level_only"] is True, "contract is not event-only")
require(authorization["performance_outcome_columns_may_be_read"] is True, "contract does not permit sentinel outcome reads")
require(authorization["performance_outcomes_may_be_inspected"] is True, "contract does not permit sentinel outcome inspection")
require(authorization["single_cell_exploratory_only"] is True, "contract is not exploratory-only")
for forbidden_permission in (
    "simulation_rerun_authorized",
    "step_level_outcome_read_authorized",
    "calibration_authorized",
    "parameter_tuning_authorized",
    "guard_limit_change_authorized",
    "implementation_modification_authorized",
    "confirmatory_inference_authorized",
    "full_campaign_authorized",
):
    require(authorization[forbidden_permission] is False, f"contract enables {forbidden_permission}")

mechanical_report = read_json(mechanical_report_path)
require(mechanical_report["schema"] == "paper1.v5_2.oracle.mechanical.audit.v1", "mechanical report schema mismatch")
require(mechanical_report["run_id"] == RUN_ID, "mechanical report run mismatch")
require(mechanical_report["status"] == "pass", "mechanical audit did not pass")
mechanical_summary = mechanical_report["mechanical_summary"]
require(int(mechanical_summary["process_guard_count"]) == EXPECTED_GUARD_STEPS, "mechanical guard count mismatch")
require(int(mechanical_summary["accepted_count"]) == EXPECTED_ACCEPTED_STEPS, "mechanical accepted count mismatch")
require(int(mechanical_summary["reliable_count"]) == EXPECTED_POSTERIOR_RELIABLE_STEPS, "mechanical reliable count mismatch")

# Only the two contract-declared event-level column allowlists are read.
twin_events = pd.read_parquet(twin_event_path, columns=TWIN_EVENT_COLUMNS)
oracle_events = pd.read_parquet(oracle_event_path, columns=ORACLE_EVENT_COLUMNS)

require(len(twin_events) == EXPECTED_EVENTS, "twin event count mismatch")
require(len(oracle_events) == EXPECTED_EVENTS, "oracle event count mismatch")
expected_event_ids = np.arange(EXPECTED_EVENTS)
require(np.array_equal(twin_events["event_id"].to_numpy(dtype=int), expected_event_ids), "twin event IDs are not contiguous")
require(np.array_equal(oracle_events["event_id"].to_numpy(dtype=int), expected_event_ids), "oracle event IDs are not contiguous")
require(bool(np.all(twin_events["steps"].to_numpy(dtype=int) == EXPECTED_STEPS_PER_EVENT)), "twin event step counts mismatch")
require(bool(np.all(oracle_events["steps"].to_numpy(dtype=int) == EXPECTED_STEPS_PER_EVENT)), "oracle event step counts mismatch")

merged = twin_events.merge(
    oracle_events,
    on="event_id",
    how="inner",
    validate="one_to_one",
    suffixes=("", "_oracle"),
)
require(len(merged) == EXPECTED_EVENTS, "event merge cardinality mismatch")
require(bool(np.all(merged["steps"] == merged["steps_oracle"])), "twin/oracle event step counts disagree")

labels = merged["label"].astype(bool).to_numpy()
nominal = merged["is_nominal"].astype(bool).to_numpy()
require(not bool(np.any(labels & nominal)), "an event is both nominal and oracle-positive")

bounded_fraction_columns = [
    "held", "state_update_accepted_fraction", "measurement_full_rank_fraction",
    "posterior_reliable_fraction", "process_guard_hold_fraction", "b1",
]
for column in bounded_fraction_columns:
    values = numeric(merged[column])
    require(bool(np.all(np.isfinite(values))), f"{column} contains nonfinite values")
    require(bool(np.all((values >= 0.0) & (values <= 1.0))), f"{column} is outside [0,1]")

held_fraction = numeric(merged["held"])
accepted_fraction = numeric(merged["state_update_accepted_fraction"])
guard_fraction = numeric(merged["process_guard_hold_fraction"])
guard_limit = numeric(merged["process_guard_limit"])
mahalanobis_max = numeric(merged["process_increment_mahalanobis_max"])
require(bool(np.allclose(held_fraction + accepted_fraction, 1.0, rtol=0.0, atol=1e-12)), "event held and accepted fractions disagree")
require(bool(np.all(guard_fraction <= held_fraction + 1e-12)), "process-guard fraction exceeds held fraction")
require(bool(np.allclose(guard_limit, EXPECTED_GUARD_LIMIT, rtol=0.0, atol=1e-12)), "process guard limit mismatch")
require(bool(np.all((guard_fraction > 0.0) == (mahalanobis_max > guard_limit))), "event guard flag and maximum Mahalanobis statistic disagree")

event_steps = numeric(merged["steps"])
guard_steps = float(np.sum(guard_fraction * event_steps))
accepted_steps = float(np.sum(accepted_fraction * event_steps))
posterior_reliable_steps = float(
    np.sum(numeric(merged["posterior_reliable_fraction"]) * event_steps)
)
measurement_full_rank_steps = float(
    np.sum(numeric(merged["measurement_full_rank_fraction"]) * event_steps)
)
require(math.isclose(guard_steps, EXPECTED_GUARD_STEPS, rel_tol=0.0, abs_tol=1e-9), "event guard aggregation conflicts with mechanical audit")
require(math.isclose(accepted_steps, EXPECTED_ACCEPTED_STEPS, rel_tol=0.0, abs_tol=1e-9), "event accepted aggregation conflicts with mechanical audit")
require(math.isclose(posterior_reliable_steps, EXPECTED_POSTERIOR_RELIABLE_STEPS, rel_tol=0.0, abs_tol=1e-9), "event posterior reliability conflicts with mechanical audit")
require(math.isclose(measurement_full_rank_steps, EXPECTED_MEASUREMENT_FULL_RANK_STEPS, rel_tol=0.0, abs_tol=1e-9), "event measurement rank conflicts with mechanical audit")

merged["outcome_class"] = np.select(
    [nominal, labels],
    ["nominal", "oracle_positive"],
    default="subthreshold_non_nominal",
)
merged["guard_any"] = guard_fraction > 0.0
merged["guard_step_equivalents"] = guard_fraction * event_steps

overall = summarize_block(merged)
outcome_groups = []
for outcome_class, block in merged.groupby("outcome_class", sort=True):
    outcome_groups.append(
        {
            "outcome_class": str(outcome_class),
            **summarize_block(block),
        }
    )

grouped = grouped_summaries(
    merged,
    ["arm", "regime", "drift_family", "outcome_class"],
)
aucs = auc_rows(merged)
auc_overall = {
    row["metric"]: row["empirical_auc"]
    for row in aucs
    if row["dimension"] == "overall"
}

guard_steps_positive = float(np.sum(merged.loc[labels, "guard_step_equivalents"]))
guard_steps_nominal = float(np.sum(merged.loc[nominal, "guard_step_equivalents"]))
subthreshold = ~labels & ~nominal
guard_steps_subthreshold = float(
    np.sum(merged.loc[subthreshold, "guard_step_equivalents"])
)
guard_any = merged["guard_any"].to_numpy(dtype=bool)
positive_guard_any_rate = float(np.mean(guard_any[labels])) if bool(np.any(labels)) else None
nominal_guard_any_rate = float(np.mean(guard_any[nominal])) if bool(np.any(nominal)) else None
positive_guard_step_fraction = (
    float(guard_steps_positive / np.sum(event_steps[labels]))
    if bool(np.any(labels))
    else None
)
nominal_guard_step_fraction = (
    float(guard_steps_nominal / np.sum(event_steps[nominal]))
    if bool(np.any(nominal))
    else None
)

alarm = merged["alarm"].astype(bool).to_numpy()
active_alarm = {
    "recall_on_oracle_positive": (
        float(np.mean(alarm[labels])) if bool(np.any(labels)) else None
    ),
    "false_positive_rate_on_all_label_negative": (
        float(np.mean(alarm[~labels])) if bool(np.any(~labels)) else None
    ),
    "alarm_rate_on_strictly_nominal": (
        float(np.mean(alarm[nominal])) if bool(np.any(nominal)) else None
    ),
    "alarm_rate_on_subthreshold_non_nominal": (
        float(np.mean(alarm[subthreshold])) if bool(np.any(subthreshold)) else None
    ),
}

guard_concentration = {
    "total_guard_step_equivalents": guard_steps,
    "guard_steps_in_oracle_positive": guard_steps_positive,
    "guard_steps_in_nominal": guard_steps_nominal,
    "guard_steps_in_subthreshold_non_nominal": guard_steps_subthreshold,
    "fraction_of_guard_steps_in_oracle_positive": safe_ratio(
        guard_steps_positive, guard_steps
    ),
    "fraction_of_guard_steps_in_nominal": safe_ratio(
        guard_steps_nominal, guard_steps
    ),
    "fraction_of_guard_steps_in_subthreshold_non_nominal": safe_ratio(
        guard_steps_subthreshold, guard_steps
    ),
    "positive_guard_any_event_rate": positive_guard_any_rate,
    "nominal_guard_any_event_rate": nominal_guard_any_rate,
    "positive_guard_step_fraction": positive_guard_step_fraction,
    "nominal_guard_step_fraction": nominal_guard_step_fraction,
    "positive_to_nominal_guard_any_enrichment": (
        safe_ratio(positive_guard_any_rate, nominal_guard_any_rate)
        if positive_guard_any_rate is not None
        and nominal_guard_any_rate is not None
        else None
    ),
    "positive_to_nominal_guard_step_enrichment": (
        safe_ratio(positive_guard_step_fraction, nominal_guard_step_fraction)
        if positive_guard_step_fraction is not None
        and nominal_guard_step_fraction is not None
        else None
    ),
}

report = {
    "schema": "paper1.v5_2.oracle.guard_impact_sentinel.v1",
    "run_id": RUN_ID,
    "status": "complete_pending_review",
    "scope": {
        "single_cell_exploratory_only": True,
        "event_level_only": True,
        "columns_read": {
            "twin_event": TWIN_EVENT_COLUMNS,
            "oracle_event": ORACLE_EVENT_COLUMNS,
        },
        "simulation_rerun": False,
        "step_level_outcome_columns_read": False,
        "calibration_performed": False,
        "parameter_tuning_performed": False,
        "guard_limit_changed": False,
        "implementation_modified": False,
        "confirmatory_inference_performed": False,
        "full_campaign_authorized": False,
        "performance_outcome_columns_read": True,
        "performance_outcomes_inspected": True,
    },
    "input_sha256": {
        "authorization_contract": EXPECTED_CONTRACT,
        "twin_events": EXPECTED_TWIN_EVENTS,
        "oracle_events": EXPECTED_ORACLE_EVENTS,
        "mechanical_audit_report": EXPECTED_MECHANICAL_REPORT,
    },
    "overall": overall,
    "outcome_class_summaries": outcome_groups,
    "guard_concentration": guard_concentration,
    "active_alarm_at_existing_threshold": active_alarm,
    "overall_empirical_auc": auc_overall,
    "empirical_auc_rows": aucs,
    "grouped_summaries": grouped,
    "interpretation_constraints": {
        "single_seed_supports_inference": False,
        "hypothesis_tests_performed": False,
        "confidence_intervals_computed": False,
        "may_tune_from_this_cell": False,
        "may_establish_F1_F2_or_F3": False,
        "requires_manual_review_before_next_authorization": True,
    },
}

raw_report_path = os.environ.get(
    "PAPER1_V5_2_SENTINEL_REPORT",
    "/workspace/paper1_v5_2_oracle_guard_impact_sentinel.json",
)
report_path = resolve_workspace_path(raw_report_path)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
if report_path.exists():
    require(report_path.read_bytes() == report_bytes, f"existing sentinel report conflicts: {report_path}")
else:
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    require(not temporary_path.exists(), f"temporary sentinel report exists: {temporary_path}")
    temporary_path.write_bytes(report_bytes)
    temporary_path.replace(report_path)

print(f"EVENTS={len(merged)}")
print(f"ORACLE_POSITIVE_EVENTS={int(np.sum(labels))}")
print(f"LABEL_NEGATIVE_EVENTS={int(np.sum(~labels))}")
print(f"NOMINAL_EVENTS={int(np.sum(nominal))}")
print(f"SUBTHRESHOLD_NON_NOMINAL_EVENTS={int(np.sum(subthreshold))}")
print(f"GUARD_ANY_EVENTS={int(np.sum(guard_any))}")
print(f"GUARD_STEP_FRACTION={guard_steps / EXPECTED_STEPS:.12g}")
print(f"POSITIVE_GUARD_ANY_EVENT_RATE={positive_guard_any_rate}")
print(f"NOMINAL_GUARD_ANY_EVENT_RATE={nominal_guard_any_rate}")
print(f"POSITIVE_GUARD_STEP_FRACTION={positive_guard_step_fraction}")
print(f"NOMINAL_GUARD_STEP_FRACTION={nominal_guard_step_fraction}")
print(f"GUARD_STEPS_IN_POSITIVE_FRACTION={safe_ratio(guard_steps_positive, guard_steps)}")
print(f"GUARD_STEPS_IN_NOMINAL_FRACTION={safe_ratio(guard_steps_nominal, guard_steps)}")
print(f"AUC_S={auc_overall.get('s')}")
print(f"AUC_CHI2={auc_overall.get('chi2')}")
auc_s = auc_overall.get("s")
auc_chi2 = auc_overall.get("chi2")
print(
    "AUC_S_MINUS_CHI2="
    + str(None if auc_s is None or auc_chi2 is None else auc_s - auc_chi2)
)
print(f"ACTIVE_ALARM_RECALL={active_alarm['recall_on_oracle_positive']}")
print(f"ACTIVE_ALARM_LABEL_NEGATIVE_RATE={active_alarm['false_positive_rate_on_all_label_negative']}")
print(f"ACTIVE_ALARM_NOMINAL_RATE={active_alarm['alarm_rate_on_strictly_nominal']}")
print(f"REPORT={report_path}")
print(f"REPORT_SHA256={sha256(report_path).upper()}")
print("SINGLE_CELL_EXPLORATORY_ONLY=True")
print("EVENT_LEVEL_ONLY=True")
print("SIMULATION_RERUN=False")
print("STEP_LEVEL_OUTCOME_COLUMNS_READ=False")
print("CALIBRATION_PERFORMED=False")
print("PARAMETER_TUNING_PERFORMED=False")
print("GUARD_LIMIT_CHANGED=False")
print("IMPLEMENTATION_MODIFIED=False")
print("CONFIRMATORY_INFERENCE_PERFORMED=False")
print("FULL_CAMPAIGN_AUTHORIZED=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=True")
print("PERFORMANCE_OUTCOMES_INSPECTED=True")
print("PAPER1_V5_2_ORACLE_GUARD_IMPACT_SENTINEL_OK")
