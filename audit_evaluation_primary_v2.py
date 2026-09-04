import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

run_id = os.environ["TARGET_RUN"]
root = Path("/workspace")
run_root = root / "runs" / run_id

calibration = json.loads(
    (root / "calibration.v2.json").read_text(
        encoding="utf-8"
    )
)

score_threshold = float(
    calibration["threshold"]["score_threshold"]
)
T_th = float(
    calibration["threshold"]["T_th"]
)

assert calibration["schema"] == "twin.calibration.v2"
assert calibration["threshold"]["event_score_construction"] == (
    "max_step(r/r0 + u/u0)"
)

threshold_from_T = float(np.log(1.0 / T_th))
assert np.isclose(
    score_threshold,
    threshold_from_T,
    rtol=0.0,
    atol=1e-12,
)

twin_steps = pd.read_parquet(
    run_root / "twin/scores.parquet"
)
twin_events = pd.read_parquet(
    run_root / "twin/scores_events.parquet"
)
oracle_steps = pd.read_parquet(
    run_root / "oracle/oracle_scores.parquet"
)
oracle_events = pd.read_parquet(
    run_root / "oracle/oracle_events.parquet"
)

assert len(twin_steps) == 13200
assert len(oracle_steps) == 13200
assert len(twin_events) == 1100
assert len(oracle_events) == 1100

assert twin_steps["step_index"].is_unique
assert oracle_steps["step_index"].is_unique
assert twin_events["event_id"].is_unique
assert oracle_events["event_id"].is_unique

assert (twin_steps.groupby("event_id").size() == 12).all()
assert (oracle_steps.groupby("event_id").size() == 12).all()

# Verify Oracle event aggregation and frozen labels.
oracle_step_d_max = (
    oracle_steps.groupby("event_id")["d"]
    .max()
    .sort_index()
)
oracle_step_label_any = (
    oracle_steps.groupby("event_id")["label"]
    .any()
    .sort_index()
)

oracle_event = (
    oracle_events.set_index("event_id")
    .sort_index()
)

d_error = float(
    np.max(
        np.abs(
            oracle_event["d"].to_numpy(float)
            - oracle_step_d_max.to_numpy(float)
        )
    )
)

assert d_error <= 1e-12
assert np.array_equal(
    oracle_event["label"].astype(bool).to_numpy(),
    oracle_step_label_any.astype(bool).to_numpy(),
)

gamma = 30.894480493264037
assert np.array_equal(
    oracle_steps["label"].astype(bool).to_numpy(),
    oracle_steps["d"].to_numpy(float) > gamma,
)

# Verify Twin score, trust and alarm aggregation.
twin_step_s_max = (
    twin_steps.groupby("event_id")["s"]
    .max()
    .sort_index()
)
twin_step_T_min = (
    twin_steps.groupby("event_id")["T"]
    .min()
    .sort_index()
)
twin_step_alarm_any = (
    twin_steps.groupby("event_id")["alarm"]
    .any()
    .sort_index()
)

twin_event = (
    twin_events.set_index("event_id")
    .sort_index()
)

s_error = float(
    np.max(
        np.abs(
            twin_event["s"].to_numpy(float)
            - twin_step_s_max.to_numpy(float)
        )
    )
)
T_error = float(
    np.max(
        np.abs(
            twin_event["T"].to_numpy(float)
            - twin_step_T_min.to_numpy(float)
        )
    )
)

assert s_error <= 1e-12
assert T_error <= 1e-12

event_alarm = twin_event["alarm"].astype(bool).to_numpy()
expected_alarm = (
    twin_event["s"].to_numpy(float)
    > score_threshold
)

assert np.array_equal(
    event_alarm,
    twin_step_alarm_any.astype(bool).to_numpy(),
)
assert np.array_equal(event_alarm, expected_alarm)
assert np.array_equal(
    event_alarm,
    twin_event["T"].to_numpy(float) < T_th,
)

# Align Oracle labels and Twin alarms.
assert np.array_equal(
    twin_event.index.to_numpy(int),
    oracle_event.index.to_numpy(int),
)

y = oracle_event["label"].astype(bool).to_numpy()
alarm = event_alarm

assert np.unique(y).size == 2

tp = int(np.sum(alarm & y))
fp = int(np.sum(alarm & ~y))
tn = int(np.sum(~alarm & ~y))
fn = int(np.sum(~alarm & y))

def divide(a, b):
    return float(a / b) if b else float("nan")

recall = divide(tp, tp + fn)
specificity = divide(tn, tn + fp)
precision = divide(tp, tp + fp)
fpr = divide(fp, fp + tn)
accuracy = divide(tp + tn, len(y))
balanced_accuracy = (recall + specificity) / 2.0
f1 = divide(2 * precision * recall, precision + recall)

def auc(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    n1 = int(y.sum())
    n0 = int((~y).sum())
    ranks = pd.Series(values).rank(
        method="average"
    ).to_numpy(float)
    return float(
        (
            ranks[y].sum()
            - n1 * (n1 + 1) / 2.0
        )
        / (n1 * n0)
    )

candidate_metrics = (
    "s",
    "s_lmax",
    "chi2",
    "huber",
    "sB1",
    "sB2",
    "s_gated_lmax",
)

auc_results = {}

for column in candidate_metrics:
    if column not in twin_event.columns:
        continue

    values = pd.to_numeric(
        twin_event[column],
        errors="coerce",
    ).to_numpy(float)

    if np.isfinite(values).all():
        auc_results[column] = auc(values)

# Frozen nominal/ample FAR population.
nominal = oracle_event["is_nominal"].astype(bool).to_numpy()
regime = oracle_event["regime"].astype(str).to_numpy()

nominal_ample = nominal & (regime == "ample")
nominal_moderate = nominal & (regime == "moderate")
nominal_severe = nominal & (regime == "severe")

nominal_ample_alarm_rate = divide(
    int(alarm[nominal_ample].sum()),
    int(nominal_ample.sum()),
)

# Create grouped descriptive results.
result_frame = pd.DataFrame({
    "event_id": oracle_event.index.to_numpy(int),
    "arm": oracle_event["arm"].fillna("NA").astype(str).to_numpy(),
    "regime": oracle_event["regime"].fillna("NA").astype(str).to_numpy(),
    "drift_family": oracle_event["drift_family"].astype(str).to_numpy(),
    "is_nominal": nominal,
    "label": y,
    "alarm": alarm,
})

grouped = (
    result_frame
    .groupby(
        ["arm", "regime", "drift_family"],
        dropna=False,
    )
    .agg(
        events=("event_id", "size"),
        positives=("label", "sum"),
        alarms=("alarm", "sum"),
    )
    .reset_index()
)

grouped["negatives"] = (
    grouped["events"] - grouped["positives"]
)

grouped["true_positives"] = (
    result_frame.assign(
        true_positive=(
            result_frame["label"]
            & result_frame["alarm"]
        )
    )
    .groupby(
        ["arm", "regime", "drift_family"],
        dropna=False,
    )["true_positive"]
    .sum()
    .to_numpy()
)

grouped["false_positives"] = (
    grouped["alarms"]
    - grouped["true_positives"]
)

grouped["label_rate"] = (
    grouped["positives"] / grouped["events"]
)
grouped["alarm_rate"] = (
    grouped["alarms"] / grouped["events"]
)
grouped["recall"] = (
    grouped["true_positives"]
    / grouped["positives"].replace(0, np.nan)
)
grouped["false_positive_rate"] = (
    grouped["false_positives"]
    / grouped["negatives"].replace(0, np.nan)
)

family_counts = (
    result_frame["drift_family"]
    .value_counts()
    .sort_index()
    .to_dict()
)

assert family_counts == {
    "load_ramp": 198,
    "nominal": 605,
    "parameter_change": 165,
    "topology_change": 132,
}

print("EVALUATION_PRIMARY_AUDIT_OK")
print("RUN_ID=", run_id)
print("SCORE_THRESHOLD=", score_threshold)
print("T_THRESHOLD=", T_th)
print("ORACLE_POSITIVES=", int(y.sum()))
print("ORACLE_NEGATIVES=", int((~y).sum()))
print("ALARMS=", int(alarm.sum()))
print("TP=", tp, "FP=", fp, "TN=", tn, "FN=", fn)
print("RECALL=", recall)
print("SPECIFICITY=", specificity)
print("PRECISION=", precision)
print("FALSE_POSITIVE_RATE=", fpr)
print("ACCURACY=", accuracy)
print("BALANCED_ACCURACY=", balanced_accuracy)
print("F1=", f1)
print("AUC_BY_METRIC=", auc_results)
print("NOMINAL_AMPLE_EVENTS=", int(nominal_ample.sum()))
print("NOMINAL_AMPLE_ALARM_RATE=", nominal_ample_alarm_rate)
print("NOMINAL_MODERATE_EVENTS=", int(nominal_moderate.sum()))
print("NOMINAL_SEVERE_EVENTS=", int(nominal_severe.sum()))
print("ORACLE_D_AGGREGATION_ERROR=", d_error)
print("TWIN_S_AGGREGATION_ERROR=", s_error)
print("TWIN_T_AGGREGATION_ERROR=", T_error)
print()
print("GROUPED_DESCRIPTIVE_RESULTS")
print(grouped.to_string(index=False))
