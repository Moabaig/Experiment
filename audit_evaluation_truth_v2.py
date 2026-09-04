import hashlib
import json
from pathlib import Path

import numpy as np

root = Path("/workspace")

calibration_path = root / "truth.calibration.v2.npz"
evaluation_path = root / "truth.eval.v2.seed001.npz"
weight_path = root / "W.frozen.v2.npy"
design_path = root / "physical_design.production.v1.json"
master_path = root / "opendss/123Bus/IEEE123Master.dss"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_calibration_sha = (
    "f83ca336d4ab69756214fc7649d18a6a"
    "5aff8555e3df1ca34db6230c56b1bd8a"
)
expected_evaluation_sha = (
    "7a47cabb4822523334bee49b9868af054"
    "c80b6a3b245be38f03411ba20aba32d"
)
expected_weight_sha = (
    "451d751207e27b194e4fc42e4c23c862"
    "ba6399caa83e2fd8cbc5db78dcaa728f"
)
expected_design_sha = (
    "ac8a9688135d34ed4abed453acdf9af7"
    "a7ed04c7e0c9d67dcc0bfbbce88d85b5"
)
expected_master_sha = (
    "c92a69d9b218b1b2646ec79117838262"
    "29309038e72f16b848304c0457c0a54d"
)
expected_allocation = {
    "nominal": 605,
    "load_ramp": 198,
    "parameter_change": 165,
    "topology_change": 132,
}

assert sha256(calibration_path) == expected_calibration_sha
assert sha256(evaluation_path) == expected_evaluation_sha
assert sha256(weight_path) == expected_weight_sha
assert sha256(design_path) == expected_design_sha
assert sha256(master_path) == expected_master_sha

with (
    np.load(calibration_path, allow_pickle=False) as cal,
    np.load(evaluation_path, allow_pickle=False) as eva,
):
    cal_meta = json.loads(cal["meta"].item())
    eva_meta = json.loads(eva["meta"].item())

    assert cal_meta["role"] == "calibration.v2"
    assert cal_meta["physical_seed"] == 51031

    assert eva_meta["role"] == "eval.v2.seed001"
    assert eva_meta["physical_seed"] == 81001

    assert eva_meta["label_independence"] == (
        "contains no detector-derived labels"
    )

    assert eva_meta["design_sha256"] == expected_design_sha
    assert eva_meta["master_sha256"] == expected_master_sha
    assert eva_meta["weight_source_sha256"] == expected_weight_sha
    assert eva_meta["event_counts"] == expected_allocation

    assert cal["x_true"].shape == (13200, 491)
    assert eva["x_true"].shape == (13200, 491)
    assert cal["z_true"].shape == (13200, 45)
    assert eva["z_true"].shape == (13200, 45)
    assert eva["z_physical"].shape == (13200, 45)

    assert np.isfinite(eva["x_true"]).all()
    assert np.isfinite(eva["z_true"]).all()
    assert np.isfinite(eva["z_physical"]).all()

    assert int(eva["physical_seed"]) == 81001
    assert np.array_equal(eva["W"], cal["W"])

    forbidden = {"label", "oracle_label", "alarm"}
    assert forbidden.isdisjoint(eva.files)

    cal_trajectories = set(
        cal["trajectory_id"].astype(str)
    )
    eva_trajectories = set(
        eva["trajectory_id"].astype(str)
    )

    assert len(cal_trajectories) == 1100
    assert len(eva_trajectories) == 1100
    assert cal_trajectories.isdisjoint(eva_trajectories)

    state_difference = float(
        np.max(
            np.abs(
                eva["x_true"].astype(float)
                - cal["x_true"].astype(float)
            )
        )
    )

    telemetry_difference = float(
        np.max(
            np.abs(
                eva["z_true"].astype(float)
                - cal["z_true"].astype(float)
            )
        )
    )

    assert state_difference > 0.0
    assert telemetry_difference > 0.0

    affine_error = float(
        np.max(
            np.abs(
                eva["z_true"].astype(float)
                - eva["z_physical"].astype(float)
                - eva["telemetry_offset"].astype(float)
            )
        )
    )

    assert affine_error <= 2e-6, affine_error

print("EVALUATION_TRUTH_V2_AUDIT_OK")
print("EVALUATION_SHA256=", sha256(evaluation_path))
print("ROLE=eval.v2.seed001")
print("PHYSICAL_SEED=81001")
print("EVENT_ALLOCATION=", expected_allocation)
print("TRAJECTORIES_DISJOINT=True")
print("STATE_MAX_DIFFERENCE_FROM_CALIBRATION=", state_difference)
print("TELEMETRY_MAX_DIFFERENCE_FROM_CALIBRATION=", telemetry_difference)
print("AFFINE_MAX_ERROR=", affine_error)
print("LABELS_PRESENT=False")
