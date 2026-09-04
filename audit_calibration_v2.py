import json
from pathlib import Path
import numpy as np

path = Path("/workspace/truth.calibration.v2.npz")

with np.load(path, allow_pickle=False) as data:
    meta = json.loads(data["meta"].item())

    families, counts = np.unique(
        data["drift_family"].astype(str),
        return_counts=True,
    )
    allocation = dict(zip(families.tolist(), counts.tolist()))

    expected = {
        "nominal": 605,
        "load_ramp": 198,
        "parameter_change": 165,
        "topology_change": 132,
    }

    assert allocation == expected, (allocation, expected)
    assert data["x_true"].shape == (13200, 491)
    assert data["z_true"].shape == (13200, 45)
    assert data["z_physical"].shape == (13200, 45)
    assert len(set(data["trajectory_id"].astype(str))) == 1100
    assert meta["role"] == "calibration.v2"
    assert meta["physical_seed"] == 51031
    assert meta["design_sha256"] == (
        "ac8a9688135d34ed4abed453acdf9af7"
        "a7ed04c7e0c9d67dcc0bfbbce88d85b5"
    )
    assert meta["master_sha256"] == (
        "c92a69d9b218b1b2646ec7911783826"
        "2293038e72f16b848304c0457c0a54d"
    )

    affine_error = float(
        np.max(
            np.abs(
                data["z_true"].astype(float)
                - data["z_physical"].astype(float)
                - data["telemetry_offset"].astype(float)
            )
        )
    )

    print("CALIBRATION_V2_AUDIT_OK")
    print("X_SHAPE=", data["x_true"].shape)
    print("Z_SHAPE=", data["z_true"].shape)
    print("EVENT_ALLOCATION=", allocation)
    print("UNIQUE_TRAJECTORIES=", len(set(data["trajectory_id"].astype(str))))
    print("AFFINE_MAX_ERROR=", affine_error)
    print("EXPORTER_SHA256=", meta["exporter_sha256"])
    print("DESIGN_SHA256=", meta["design_sha256"])
    print("MODEL_TREE_SHA256=", meta["model_tree_sha256"])
