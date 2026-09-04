import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

run = os.environ["TARGET_RUN"]
root = Path("/workspace/runs") / run

step = pd.read_parquet(root / "twin/scores.parquet")
event = pd.read_parquet(root / "oracle/oracle_events.parquet")
cal = json.loads(Path("/workspace/calibration.json").read_text())

r0 = float(cal["normalizers"]["r0"])
u0 = float(cal["normalizers"]["u0_lmax"])
score_threshold = float(cal["threshold"]["score_threshold"])
T_th = float(cal["threshold"]["T_th"])

label_column = "label" if "label" in event.columns else "oracle_label"

step_s = pd.to_numeric(step["s"], errors="raise").to_numpy(float)
step_T = pd.to_numeric(step["T"], errors="raise").to_numpy(float)

step_identity_error = np.max(np.abs(step_T - np.exp(-step_s)))

smax_stored = step.groupby("event_id")["s"].max()
Tmin_stored = step.groupby("event_id")["T"].min()
Tmax_stored = step.groupby("event_id")["T"].max()

event_indexed = event.set_index("event_id")

event_s_error = np.max(
    np.abs(event_indexed.loc[smax_stored.index, "s"].to_numpy(float)
           - smax_stored.to_numpy(float))
)

event_T_to_min_error = np.max(
    np.abs(event_indexed.loc[Tmin_stored.index, "T"].to_numpy(float)
           - Tmin_stored.to_numpy(float))
)

event_T_to_max_error = np.max(
    np.abs(event_indexed.loc[Tmax_stored.index, "T"].to_numpy(float)
           - Tmax_stored.to_numpy(float))
)

frozen_step_s = (
    pd.to_numeric(step["r"], errors="raise").to_numpy(float) / r0
    + pd.to_numeric(step["u_lmax"], errors="raise").to_numpy(float) / u0
)

frozen_event_s = (
    pd.DataFrame({
        "event_id": step["event_id"].to_numpy(),
        "frozen_s": frozen_step_s
    })
    .groupby("event_id")["frozen_s"]
    .max()
)

aligned = event_indexed.loc[frozen_event_s.index]
y = aligned[label_column].astype(bool).to_numpy()
alarm = frozen_event_s.to_numpy(float) > score_threshold

n1 = int(y.sum())
n0 = int((~y).sum())

ranks = pd.Series(frozen_event_s.to_numpy(float)).rank(
    method="average"
).to_numpy()

auc = (
    ranks[y].sum() - n1 * (n1 + 1) / 2
) / (n1 * n0)

print("=" * 72)
print("RUN=", run)
print("STEP_T_EXP_MINUS_S_MAX_ERROR=", step_identity_error)
print("EVENT_S_VS_STEP_MAX_ERROR=", event_s_error)
print("EVENT_T_VS_STEP_MIN_ERROR=", event_T_to_min_error)
print("EVENT_T_VS_STEP_MAX_ERROR=", event_T_to_max_error)
print("FROZEN_SCORE_THRESHOLD=", score_threshold)
print("FROZEN_T_THRESHOLD=", T_th)
print("FROZEN_SCORE_AUC=", auc)
print("FROZEN_SCORE_ALARMS=", int(alarm.sum()))
print("FROZEN_SCORE_RECALL=", float(alarm[y].mean()))
print("FROZEN_SCORE_FAR=", float(alarm[~y].mean()))
