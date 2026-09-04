import os
from pathlib import Path

import numpy as np
import pandas as pd

run_id = os.environ["TARGET_RUN"]
gamma = 30.894480493264037

root = Path("/workspace/runs") / run_id

steps = pd.read_parquet(
    root / "oracle/oracle_scores.parquet"
)
events = pd.read_parquet(
    root / "oracle/oracle_events.parquet"
)
twin_steps = pd.read_parquet(
    root / "twin/scores.parquet"
)
twin_events = pd.read_parquet(
    root / "twin/scores_events.parquet"
)

assert len(steps) == 13200
assert len(events) == 1100
assert len(twin_steps) == 13200
assert len(twin_events) == 1100

assert steps["step_index"].is_unique
assert events["event_id"].is_unique
assert steps["event_id"].nunique() == 1100

step_counts = steps.groupby("event_id").size()
assert (step_counts == 12).all()

step_d = pd.to_numeric(
    steps["d"],
    errors="raise",
).to_numpy(float)

assert np.isfinite(step_d).all()

step_labels = steps["label"].astype(bool).to_numpy()
assert np.array_equal(step_labels, step_d > gamma)

grouped = steps.groupby("event_id", sort=True)

expected_event_d = grouped["d"].max()
expected_event_label = grouped["label"].any()
expected_event_nominal = grouped["is_nominal"].all()

event_table = (
    events.set_index("event_id")
    .sort_index()
)

event_d = pd.to_numeric(
    event_table["d"],
    errors="raise",
).to_numpy(float)

aggregation_error = float(
    np.max(
        np.abs(
            event_d
            - expected_event_d.to_numpy(float)
        )
    )
)

assert aggregation_error <= 1e-12
assert np.array_equal(
    event_table["label"].astype(bool).to_numpy(),
    expected_event_label.astype(bool).to_numpy(),
)
assert np.array_equal(
    event_table["is_nominal"].astype(bool).to_numpy(),
    expected_event_nominal.astype(bool).to_numpy(),
)

event_labels = event_table["label"].astype(bool).to_numpy()

assert int(event_labels.sum()) == 47
assert int((~event_labels).sum()) == 1053

eligible = (
    event_table["is_nominal"].astype(bool).to_numpy()
    & event_table["regime"].astype(str)
      .eq("ample").to_numpy()
)

moderate = (
    event_table["is_nominal"].astype(bool).to_numpy()
    & event_table["regime"].astype(str)
      .eq("moderate").to_numpy()
)

assert int(eligible.sum()) == 117
assert int(moderate.sum()) >= 20

eligible_d = event_d[eligible]
eligible_exceedances = int(
    np.count_nonzero(eligible_d > gamma)
)
eligible_far = float(
    np.mean(eligible_d > gamma)
)

assert eligible_exceedances == 1
assert np.isclose(
    eligible_far,
    1.0 / 117.0,
    rtol=0.0,
    atol=1e-15,
)

print("FINAL_CALIBRATION_TABLES_OK")
print("RUN_ID=", run_id)
print("STEP_ROWS=", len(steps))
print("EVENT_ROWS=", len(events))
print("STEP_POSITIVES=", int(step_labels.sum()))
print("EVENT_POSITIVES=", int(event_labels.sum()))
print("EVENT_NEGATIVES=", int((~event_labels).sum()))
print("NOMINAL_AMPLE_EVENTS=", int(eligible.sum()))
print("NOMINAL_MODERATE_EVENTS=", int(moderate.sum()))
print("NOMINAL_AMPLE_EXCEEDANCES=", eligible_exceedances)
print("NOMINAL_AMPLE_FAR=", eligible_far)
print("EVENT_D_AGGREGATION_MAX_ERROR=", aggregation_error)
