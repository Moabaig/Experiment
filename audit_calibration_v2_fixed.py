import hashlib
import json
from pathlib import Path

import numpy as np

root = Path("/workspace")
truth_path = root / "truth.calibration.v2.npz"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected_output = (
    "f83ca336d4ab69756214fc7649d18a6a"
    "5aff8555e3df1ca34db6230c56b1bd8a"
)
expected_master = (
    "c92a69d9b218b1b2646ec79117838262"
    "29309038e72f16b848304c0457c0a54d"
)
expected_design = (
    "ac8a9688135d34ed4abed453acdf9af7"
    "a7ed04c7e0c9d67dcc0bfbbce88d85b5"
)
expected_allocation = {
    "nominal": 605,
    "load_ramp": 198,
    "parameter_change": 165,
    "topology_change": 132,
}

actual_output = sha256(truth_path)
assert actual_output == expected_output, actual_output

with np.load(truth_path, allow_pickle=False) as data:
    meta = json.loads(data["meta"].item())

    families, counts = np.unique(
        data["drift_family"].astype(str),
        return_counts=True,
    )
    allocation = dict(zip(families.tolist(), counts.tolist()))

    assert allocation == expected_allocation
    assert data["x_true"].shape == (13200, 491)
    assert data["z_true"].shape == (13200, 45)
    assert data["z_physical"].shape == (13200, 45)
    assert len(set(data["trajectory_id"].astype(str))) == 1100
    assert meta["role"] == "calibration.v2"
    assert meta["physical_seed"] == 51031
    assert meta["master_sha256"] == expected_master
    assert meta["design_sha256"] == expected_design

    assert sha256(
        root / "opendss/123Bus/IEEE123Master.dss"
    ) == meta["master_sha256"]

    assert sha256(
        root / "physical_design.production.v1.json"
    ) == meta["design_sha256"]

    assert sha256(
        root / "feeder.npz"
    ) == meta["feeder_sha256"]

    assert sha256(
        root / "export_opendss_truth.py"
    ) == meta["exporter_sha256"]

    assert sha256(
        root / "W.frozen.v2.npy"
    ) == meta["weight_source_sha256"]

    affine_error = float(
        np.max(
            np.abs(
                data["z_true"].astype(float)
                - data["z_physical"].astype(float)
                - data["telemetry_offset"].astype(float)
            )
        )
    )
    assert affine_error <= 2e-6, affine_error

print("CALIBRATION_V2_AUDIT_OK")
print("OUTPUT_SHA256=", actual_output)
print("MASTER_SHA256=", expected_master)
print("EVENT_ALLOCATION=", allocation)
print("UNIQUE_TRAJECTORIES=1100")
print("AFFINE_MAX_ERROR=", affine_error)
