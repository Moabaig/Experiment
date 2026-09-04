from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/workspace")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


design_path = ROOT / "factor_design.production.v3.json"
design = json.loads(design_path.read_text(encoding="utf-8"))

assert design["schema"] == "twin.factor.design.v3"
assert design["campaign_cells"] == 150
assert design["seed_policy"]["count"] == 30
assert design["seed_policy"]["seed_indices"] == list(range(1, 31))

levels = design["bandwidth_levels"]
assert [row["index"] for row in levels] == list(range(5))
assert [row["id"] for row in levels] == [
    "bw00_floor",
    "bw01_10kbps",
    "bw02_100kbps",
    "bw03_1mbps",
    "bw04_oracle",
]
assert [float(row["bandwidth_cap_bps"]) for row in levels] == [
    0.5,
    10_000.0,
    100_000.0,
    1_000_000.0,
    1_000_000_000_000.0,
]

for name, expected in (
    ("calibration.v2.json", design["frozen_inputs"]["calibration_sha256"]),
    ("gamma.frozen.v2.txt", design["frozen_inputs"]["gamma_sha256"]),
    ("W.frozen.v2.npy", design["frozen_inputs"]["weight_sha256"]),
    (
        "physical_design.production.v1.json",
        design["frozen_inputs"]["physical_design_sha256"],
    ),
):
    assert sha256(ROOT / name) == expected, name

with np.load(ROOT / "patterns.npz", allow_pickle=False) as archive:
    B = np.asarray(archive["B"], dtype=float)
    n_telemetry = int(archive["n_telemetry"])

assert B.shape == (1100, 583)
assert np.isfinite(B).all()
assert np.array_equal(np.unique(B), np.array([0.5, 1.0e12]))
assert np.array_equal(np.minimum(B, 1.0e12), B)

scenarios = pd.read_csv(ROOT / "scenarios.csv")
assert len(scenarios) == 1100
assert set(scenarios["arm"].astype(str)) == {"C", "G", "T"}
assert set(scenarios["regime"].dropna().astype(str)) == {
    "ample",
    "moderate",
    "severe",
}

source_markers = {
    "net_fed.cc": (
        "bandwidthCapBps",
        "effectiveBandwidth",
        "bandwidth_level",
    ),
    "twin_fed.py": (
        "apply_bandwidth_cap",
        "--bandwidth-cap-bps",
        "--compute-delta-check",
        "s_delta_lmax",
    ),
    "docker-compose.yml": (
        "BANDWIDTH_CAP_BPS",
        "BANDWIDTH_LEVEL",
        "--compute-delta-check",
    ),
    "run_experiment.ps1": (
        "refusing to overwrite",
        "BANDWIDTH_CAP_BPS",
    ),
}

for name, markers in source_markers.items():
    text = (ROOT / name).read_text(encoding="utf-8")
    for marker in markers:
        assert marker in text, (name, marker)

telemetry_B = B[:, :n_telemetry]
rows = []
for level in levels:
    cap = float(level["bandwidth_cap_bps"])
    effective = np.minimum(telemetry_B, cap)
    starved = (effective <= float(design["b_min"])).sum(axis=1)
    rows.append(
        {
            "id": level["id"],
            "cap_bps": cap,
            "minimum_starved_channels": int(starved.min()),
            "median_starved_channels": float(np.median(starved)),
            "maximum_starved_channels": int(starved.max()),
        }
    )

assert rows[0]["minimum_starved_channels"] == n_telemetry
assert rows[-1]["maximum_starved_channels"] < n_telemetry

print("FACTOR_V3_STATIC_VERIFY_OK")
print("DESIGN_SHA256=", sha256(design_path))
print("CAMPAIGN_CELLS=150")
print("SEEDS=30")
print("BANDWIDTH_LEVELS=5")
print("BANDWIDTH_DIAGNOSTICS=", json.dumps(rows, sort_keys=True))
