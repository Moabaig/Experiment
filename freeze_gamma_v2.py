import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/workspace")
run_id = os.environ["TARGET_RUN"]

event_path = (
    root / "runs" / run_id /
    "oracle" / "oracle_events.parquet"
)
step_path = (
    root / "runs" / run_id /
    "oracle" / "oracle_scores.parquet"
)
oracle_meta_path = (
    root / "runs" / run_id /
    "oracle" / "meta.json"
)
truth_path = root / "truth.calibration.v2.npz"
weight_path = root / "W.frozen.v2.npy"
rule_path = root / f"gamma_rule_{run_id}.txt"

gamma_path = root / "gamma.frozen.v2.txt"
selection_path = root / "gamma_selection.v2.json"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path in (
    event_path,
    step_path,
    oracle_meta_path,
    truth_path,
    weight_path,
    rule_path,
):
    assert path.is_file(), f"Missing required input: {path}"

assert not gamma_path.exists(), gamma_path
assert not selection_path.exists(), selection_path

expected_truth_sha = (
    "f83ca336d4ab69756214fc7649d18a6a"
    "5aff8555e3df1ca34db6230c56b1bd8a"
)
expected_weight_sha = (
    "451d751207e27b194e4fc42e4c23c862"
    "ba6399caa83e2fd8cbc5db78dcaa728f"
)

assert sha256(truth_path) == expected_truth_sha
assert sha256(weight_path) == expected_weight_sha

rule_text = rule_path.read_text(encoding="utf-8")

required_rule_lines = (
    "schema=oracle.gamma.selection.v2",
    f"run_id={run_id}",
    "source_split=calibration_only",
    "quantile=0.99",
    "quantile_method=higher",
    "evaluation_data_used=false",
    "pilot_labels_used=false",
)

for line in required_rule_lines:
    assert line in rule_text, (
        f"Predeclared rule is missing: {line}"
    )

oracle_meta = json.loads(
    oracle_meta_path.read_text(encoding="utf-8")
)

assert oracle_meta["status"] == "complete"
assert float(
    oracle_meta["label_definition"]["gamma"]
) == 1.0
assert (
    oracle_meta["event_aggregation"]["high_is_bad"]
    == "max"
)

events = pd.read_parquet(event_path)

assert len(events) == 1100
assert events["event_id"].is_unique

d = pd.to_numeric(
    events["d"],
    errors="raise",
).to_numpy(float)

assert np.isfinite(d).all()

eligible = (
    events["is_nominal"].astype(bool).to_numpy()
    & events["regime"].astype(str).eq("ample").to_numpy()
)

eligible_d = d[eligible]

assert len(eligible_d) == 117, len(eligible_d)
assert np.isfinite(eligible_d).all()

gamma = float(
    np.quantile(
        eligible_d,
        0.99,
        method="higher",
    )
)

expected_gamma = 30.894480493264037

assert np.isfinite(gamma)
assert gamma > 0.0
assert gamma == expected_gamma, (gamma, expected_gamma)

frozen_labels = d > gamma
eligible_exceedances = int(
    np.count_nonzero(eligible_d > gamma)
)
realized_far = float(
    np.mean(eligible_d > gamma)
)

assert eligible_exceedances == 1
assert np.isclose(
    realized_far,
    1.0 / 117.0,
    rtol=0.0,
    atol=1e-15,
)

pilot_labels = events["label"].astype(bool).to_numpy()

selection = {
    "schema": "oracle.gamma.selection.v2",
    "created_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "gamma": gamma,
    "rule": (
        "99th percentile of nominal/ample event-level "
        "maximum Oracle d"
    ),
    "quantile": 0.99,
    "quantile_method": "higher",
    "selection_mask": (
        "is_nominal == true and regime == ample"
    ),
    "statistic": (
        "event-level d = maximum step-level Oracle d"
    ),
    "source_split": "calibration_only",
    "source_run_id": run_id,
    "source_event_table": str(event_path),
    "source_event_sha256": sha256(event_path),
    "source_step_table": str(step_path),
    "source_step_sha256": sha256(step_path),
    "source_oracle_meta_sha256": sha256(
        oracle_meta_path
    ),
    "truth_file": str(truth_path),
    "truth_sha256": sha256(truth_path),
    "weight_definition": (
        "identity_matrix_491_by_491"
    ),
    "weight_file": str(weight_path),
    "weight_sha256": sha256(weight_path),
    "predeclared_rule_file": str(rule_path),
    "predeclared_rule_sha256": sha256(rule_path),
    "eligible_events": int(len(eligible_d)),
    "eligible_exceedances": eligible_exceedances,
    "target_nominal_ample_event_far": 0.01,
    "realized_nominal_ample_event_far": realized_far,
    "positive_events_all_calibration": int(
        frozen_labels.sum()
    ),
    "negative_events_all_calibration": int(
        (~frozen_labels).sum()
    ),
    "events_equal_to_gamma": int(
        np.count_nonzero(d == gamma)
    ),
    "pilot_gamma": float(
        oracle_meta["label_definition"]["gamma"]
    ),
    "pilot_positive_events": int(
        pilot_labels.sum()
    ),
    "pilot_labels_used": False,
    "evaluation_data_used": False,
}

gamma_path.write_text(
    format(gamma, ".17g") + "\n",
    encoding="ascii",
)

selection_path.write_text(
    json.dumps(
        selection,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print(
    "GAMMA_V2_FREEZE_OK",
    "gamma=", format(gamma, ".17g"),
    "eligible=", len(eligible_d),
    "eligible_exceedances=", eligible_exceedances,
    "realized_far=", realized_far,
    "positive=", int(frozen_labels.sum()),
    "negative=", int((~frozen_labels).sum()),
)
