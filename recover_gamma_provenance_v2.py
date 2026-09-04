import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/workspace")
run_id = os.environ["TARGET_RUN"]

selection_path = root / "gamma_selection.v2.json"
gamma_path = root / "gamma.frozen.v2.txt"

run_root = root / "runs" / run_id
step_path = run_root / "oracle/oracle_scores.parquet"
event_path = run_root / "oracle/oracle_events.parquet"
meta_path = run_root / "oracle/meta.json"

output_path = root / "gamma_provenance_recovery.v2.json"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_selection_sha = (
    "268153d15cdbbf2cee97d40aaf884126"
    "c79081471d86d3ad58af60c056d23b9c"
)
expected_gamma_sha = (
    "964e2e4d8229aef1533e894c6b724d91"
    "f9a70e4beb5cfa367e27dfecd63b23d0"
)

assert sha256(selection_path) == expected_selection_sha
assert sha256(gamma_path) == expected_gamma_sha

selection = json.loads(
    selection_path.read_text(encoding="utf-8")
)
oracle_meta = json.loads(
    meta_path.read_text(encoding="utf-8")
)

current_step_sha = sha256(step_path)
current_event_sha = sha256(event_path)
current_meta_sha = sha256(meta_path)

assert (
    current_step_sha
    == selection["source_step_sha256"]
), "The preserved step-level source no longer matches"

steps = pd.read_parquet(step_path)
events = pd.read_parquet(event_path)

assert len(steps) == 13200
assert len(events) == 1100
assert steps["event_id"].nunique() == 1100
assert events["event_id"].is_unique

step_counts = steps.groupby("event_id").size()
assert (step_counts == 12).all()

grouped = steps.groupby("event_id", sort=True)

reconstructed_d = grouped["d"].max()
reconstructed_nominal = grouped["is_nominal"].all()

regime_counts = grouped["regime"].nunique(dropna=False)
assert (regime_counts == 1).all()

reconstructed_regime = grouped["regime"].first()

current_events = (
    events.set_index("event_id")
    .sort_index()
)

assert np.array_equal(
    current_events.index.to_numpy(int),
    reconstructed_d.index.to_numpy(int),
)

current_d = pd.to_numeric(
    current_events["d"],
    errors="raise",
).to_numpy(float)

reconstructed_d_array = pd.to_numeric(
    reconstructed_d,
    errors="raise",
).to_numpy(float)

aggregation_error = float(
    np.max(
        np.abs(
            current_d - reconstructed_d_array
        )
    )
)

assert aggregation_error <= 1e-12, aggregation_error

assert np.array_equal(
    current_events["is_nominal"].astype(bool).to_numpy(),
    reconstructed_nominal.astype(bool).to_numpy(),
)

current_regime = (
    current_events["regime"]
    .fillna("<NA>")
    .astype(str)
    .to_numpy()
)

reconstructed_regime_array = (
    reconstructed_regime
    .fillna("<NA>")
    .astype(str)
    .to_numpy()
)

assert np.array_equal(
    current_regime,
    reconstructed_regime_array,
)

eligible = (
    reconstructed_nominal.astype(bool).to_numpy()
    & reconstructed_regime.astype(str)
      .eq("ample").to_numpy()
)

eligible_d = reconstructed_d_array[eligible]

assert len(eligible_d) == 117
assert np.isfinite(eligible_d).all()

recomputed_gamma = float(
    np.quantile(
        eligible_d,
        0.99,
        method="higher",
    )
)

frozen_gamma = float(
    gamma_path.read_text(encoding="ascii").strip()
)

assert recomputed_gamma == frozen_gamma
assert recomputed_gamma == selection["gamma"]

eligible_exceedances = int(
    np.count_nonzero(eligible_d > recomputed_gamma)
)
realized_far = float(
    np.mean(eligible_d > recomputed_gamma)
)

assert eligible_exceedances == 1
assert np.isclose(
    realized_far,
    1.0 / 117.0,
    rtol=0.0,
    atol=1e-15,
)

assert float(
    oracle_meta["label_definition"]["gamma"]
) == 1.0

recovery = {
    "schema": "oracle.gamma.provenance-recovery.v2",
    "created_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "reason": (
        "The pilot run identifier was accidentally reused, "
        "replacing the byte-serialized event and metadata files."
    ),
    "scientific_effect": (
        "None on gamma selection: the step-level Oracle table "
        "is byte-for-byte identical to the originally recorded "
        "source and deterministically reconstructs the event-level "
        "selection statistic."
    ),
    "source_run_id": run_id,
    "original_selection_file": str(selection_path),
    "original_selection_sha256": sha256(selection_path),
    "frozen_gamma_file": str(gamma_path),
    "frozen_gamma_sha256": sha256(gamma_path),
    "original_recorded_step_sha256": (
        selection["source_step_sha256"]
    ),
    "current_step_sha256": current_step_sha,
    "step_source_byte_identical": True,
    "original_recorded_event_sha256": (
        selection["source_event_sha256"]
    ),
    "current_event_sha256": current_event_sha,
    "event_source_byte_identical": (
        current_event_sha
        == selection["source_event_sha256"]
    ),
    "original_recorded_oracle_meta_sha256": (
        selection["source_oracle_meta_sha256"]
    ),
    "current_oracle_meta_sha256": current_meta_sha,
    "event_rows": int(len(events)),
    "step_rows": int(len(steps)),
    "steps_per_event": 12,
    "event_d_reconstruction": (
        "maximum step-level d grouped by event_id"
    ),
    "event_d_reconstruction_max_error": aggregation_error,
    "eligible_events": int(len(eligible_d)),
    "quantile": 0.99,
    "quantile_method": "higher",
    "recomputed_gamma": recomputed_gamma,
    "frozen_gamma": frozen_gamma,
    "eligible_exceedances": eligible_exceedances,
    "realized_nominal_ample_event_far": realized_far,
    "pilot_labels_used": False,
    "evaluation_data_used": False,
}

output_path.write_text(
    json.dumps(
        recovery,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("GAMMA_PROVENANCE_RECOVERY_OK")
print("STEP_SOURCE_BYTE_IDENTICAL=True")
print("EVENT_D_RECONSTRUCTION_MAX_ERROR=", aggregation_error)
print("ELIGIBLE_EVENTS=", len(eligible_d))
print("RECOMPUTED_GAMMA=", format(recomputed_gamma, ".17g"))
print("REALIZED_FAR=", realized_far)
