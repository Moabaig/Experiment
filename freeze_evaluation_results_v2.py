import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/workspace")
run_id = os.environ["TARGET_RUN"]
run_root = root / "runs" / run_id

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ratio(a, b):
    return float(a / b) if b else float("nan")

def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(centre - half), float(centre + half)]

calibration_path = root / "calibration.v2.json"
gamma_path = root / "gamma.frozen.v2.txt"
truth_path = root / "truth.eval.v2.seed001.npz"
plan_path = root / f"evaluation_run_plan.{run_id}.txt"

twin_step_path = run_root / "twin/scores.parquet"
twin_event_path = run_root / "twin/scores_events.parquet"
oracle_step_path = run_root / "oracle/oracle_scores.parquet"
oracle_event_path = run_root / "oracle/oracle_events.parquet"

calibration = json.loads(
    calibration_path.read_text(encoding="utf-8")
)

threshold = float(
    calibration["threshold"]["score_threshold"]
)
T_th = float(calibration["threshold"]["T_th"])
gamma = float(gamma_path.read_text().strip())

twin = (
    pd.read_parquet(twin_event_path)
    .set_index("event_id")
    .sort_index()
)
oracle = (
    pd.read_parquet(oracle_event_path)
    .set_index("event_id")
    .sort_index()
)

assert len(twin) == 1100
assert len(oracle) == 1100
assert np.array_equal(twin.index, oracle.index)

y = oracle["label"].astype(bool).to_numpy()
alarm = twin["alarm"].astype(bool).to_numpy()
score = twin["s"].to_numpy(float)

assert np.array_equal(alarm, score > threshold)

tp = int(np.sum(alarm & y))
fp = int(np.sum(alarm & ~y))
tn = int(np.sum(~alarm & ~y))
fn = int(np.sum(~alarm & y))

recall = ratio(tp, tp + fn)
specificity = ratio(tn, tn + fp)
precision = ratio(tp, tp + fp)
fpr = ratio(fp, fp + tn)
accuracy = ratio(tp + tn, len(y))
balanced_accuracy = (recall + specificity) / 2.0
f1 = ratio(2.0 * precision * recall, precision + recall)
prevalence = float(y.mean())
alarm_rate = float(alarm.mean())

def auc(values):
    values = np.asarray(values, dtype=float)
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

auc_results = {}

for column in (
    "s",
    "s_lmax",
    "chi2",
    "huber",
    "sB1",
    "sB2",
    "s_gated_lmax",
):
    if column in twin.columns:
        values = pd.to_numeric(
            twin[column],
            errors="coerce",
        ).to_numpy(float)
        if np.isfinite(values).all():
            auc_results[column] = auc(values)

nominal = oracle["is_nominal"].astype(bool).to_numpy()
regime = oracle["regime"].astype(str).to_numpy()
nominal_ample = nominal & (regime == "ample")

nominal_ample_events = int(nominal_ample.sum())
nominal_ample_alarms = int(alarm[nominal_ample].sum())
nominal_ample_rate = ratio(
    nominal_ample_alarms,
    nominal_ample_events,
)

frame = pd.DataFrame({
    "event_id": oracle.index.to_numpy(int),
    "arm": oracle["arm"].fillna("NA").astype(str).to_numpy(),
    "regime": oracle["regime"].fillna("NA").astype(str).to_numpy(),
    "drift_family": oracle["drift_family"].astype(str).to_numpy(),
    "label": y,
    "alarm": alarm,
})

frame["true_positive"] = frame["label"] & frame["alarm"]
frame["false_positive"] = ~frame["label"] & frame["alarm"]

grouped = (
    frame.groupby(
        ["arm", "regime", "drift_family"],
        dropna=False,
    )
    .agg(
        events=("event_id", "size"),
        positives=("label", "sum"),
        alarms=("alarm", "sum"),
        true_positives=("true_positive", "sum"),
        false_positives=("false_positive", "sum"),
    )
    .reset_index()
)

grouped["negatives"] = (
    grouped["events"] - grouped["positives"]
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

group_output = root / "evaluation_group_results.v2.csv"
grouped.to_csv(group_output, index=False)

results = {
    "schema": "evaluation.primary-results.v2",
    "created_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "run_id": run_id,
    "status": "frozen_primary_evaluation",
    "evaluation_data_used_for_tuning": False,
    "primary_endpoint": (
        "event alarm from max-step frozen lmax score "
        "versus event Oracle label"
    ),
    "label_definition": {
        "formula": "event maximum Oracle d > frozen gamma",
        "gamma": gamma,
    },
    "alarm_definition": {
        "formula": "event maximum s > frozen score_threshold",
        "score_threshold": threshold,
        "T_th": T_th,
    },
    "counts": {
        "events": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((~y).sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    },
    "metrics": {
        "prevalence": prevalence,
        "alarm_rate": alarm_rate,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "always_negative_accuracy": float((~y).mean()),
        "auc_by_metric": auc_results,
    },
    "wilson_95_percent_intervals": {
        "recall": wilson(tp, tp + fn),
        "specificity": wilson(tn, tn + fp),
        "precision": wilson(tp, tp + fp),
        "false_positive_rate": wilson(fp, fp + tn),
        "nominal_ample_alarm_rate": wilson(
            nominal_ample_alarms,
            nominal_ample_events,
        ),
    },
    "nominal_ample_control": {
        "events": nominal_ample_events,
        "alarms": nominal_ample_alarms,
        "alarm_rate": nominal_ample_rate,
        "target_far": 0.01,
    },
    "interpretation_flags": {
        "primary_score_auc_below_half": (
            auc_results["s"] < 0.5
        ),
        "balanced_accuracy_below_half": (
            balanced_accuracy < 0.5
        ),
        "accuracy_below_always_negative": (
            accuracy < float((~y).mean())
        ),
        "group_positive_counts_sparse": True,
    },
    "provenance": {
        "truth_sha256": sha256(truth_path),
        "calibration_sha256": sha256(calibration_path),
        "gamma_sha256": sha256(gamma_path),
        "evaluation_plan_sha256": sha256(plan_path),
        "twin_steps_sha256": sha256(twin_step_path),
        "twin_events_sha256": sha256(twin_event_path),
        "oracle_steps_sha256": sha256(oracle_step_path),
        "oracle_events_sha256": sha256(oracle_event_path),
        "group_results_sha256": sha256(group_output),
    },
}

result_output = root / "evaluation_primary_results.v2.json"
result_output.write_text(
    json.dumps(
        results,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("EVALUATION_PRIMARY_RESULTS_FROZEN_OK")
print("RESULT_JSON_SHA256=", sha256(result_output))
print("GROUP_CSV_SHA256=", sha256(group_output))
print("PRIMARY_AUC=", auc_results["s"])
print("BALANCED_ACCURACY=", balanced_accuracy)
print("FALSE_POSITIVE_RATE=", fpr)
print("RECALL=", recall)
