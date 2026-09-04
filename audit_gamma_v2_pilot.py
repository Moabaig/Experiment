import os
from pathlib import Path

import numpy as np
import pandas as pd

run_id = os.environ["TARGET_RUN"]
root = Path("/workspace/runs") / run_id

twin_steps = pd.read_parquet(
    root / "twin/scores.parquet"
)
twin_events = pd.read_parquet(
    root / "twin/scores_events.parquet"
)
oracle_steps = pd.read_parquet(
    root / "oracle/oracle_scores.parquet"
)
oracle_events = pd.read_parquet(
    root / "oracle/oracle_events.parquet"
)

assert len(twin_steps) == 13200, len(twin_steps)
assert len(twin_events) == 1100, len(twin_events)
assert len(oracle_steps) == 13200, len(oracle_steps)
assert len(oracle_events) == 1100, len(oracle_events)

assert twin_steps["event_id"].nunique() == 1100
assert oracle_steps["event_id"].nunique() == 1100
assert twin_events["event_id"].is_unique
assert oracle_events["event_id"].is_unique

expected_ids = set(range(1100))
assert set(twin_events["event_id"].astype(int)) == expected_ids
assert set(oracle_events["event_id"].astype(int)) == expected_ids

step_counts = oracle_steps.groupby("event_id").size()
assert (step_counts == 12).all(), step_counts.value_counts().to_dict()

step_d_max = (
    oracle_steps.groupby("event_id")["d"]
    .max()
    .sort_index()
)

event_table = (
    oracle_events.set_index("event_id")
    .sort_index()
)

event_d = pd.to_numeric(
    event_table["d"],
    errors="raise",
).to_numpy(float)

d_aggregation_error = float(
    np.max(
        np.abs(
            event_d
            - step_d_max.to_numpy(float)
        )
    )
)

assert d_aggregation_error <= 1e-12, d_aggregation_error
assert np.isfinite(event_d).all()

pilot_labels = event_table["label"].astype(bool).to_numpy()
expected_pilot_labels = event_d > 1.0
assert np.array_equal(
    pilot_labels,
    expected_pilot_labels,
), "Pilot labels do not implement d > 1.0"

eligible_mask = (
    event_table["is_nominal"].astype(bool).to_numpy()
    & event_table["regime"].astype(str).eq("ample").to_numpy()
)

eligible_d = event_d[eligible_mask]

assert len(eligible_d) >= 20, len(eligible_d)
assert np.isfinite(eligible_d).all()

print("GAMMA_V2_PILOT_AUDIT_OK")
print("RUN_ID=", run_id)
print("TWIN_STEPS=", len(twin_steps))
print("TWIN_EVENTS=", len(twin_events))
print("ORACLE_STEPS=", len(oracle_steps))
print("ORACLE_EVENTS=", len(oracle_events))
print("EVENT_D_MAX_AGGREGATION_ERROR=", d_aggregation_error)
print("PILOT_POSITIVE_EVENTS=", int(pilot_labels.sum()))
print("PILOT_NEGATIVE_EVENTS=", int((~pilot_labels).sum()))
print("ELIGIBLE_NOMINAL_AMPLE_EVENTS=", len(eligible_d))
print("ELIGIBLE_D_MIN=", float(eligible_d.min()))
print("ELIGIBLE_D_MEDIAN=", float(np.median(eligible_d)))
print(
    "ELIGIBLE_D_P90=",
    float(np.quantile(eligible_d, 0.90, method="higher")),
)
print(
    "ELIGIBLE_D_P99_HIGHER=",
    float(np.quantile(eligible_d, 0.99, method="higher")),
)
print("ELIGIBLE_D_MAX=", float(eligible_d.max()))
