#!/usr/bin/env python3
"""Run a small real-OpenDSS repeatability and schema test."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True, cwd=ROOT)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="opendss-exporter-test-") as directory:
        temporary = Path(directory)
        weight = temporary / "W.npy"
        first = temporary / "truth-a.npz"
        second = temporary / "truth-b.npz"
        np.save(weight, np.eye(491, dtype=np.float64))
        common = (
            str(ROOT / "export_opendss_truth.py"),
            "--master",
            str(ROOT / "opendss/123Bus/IEEE123Master.dss"),
            "--model-root",
            str(ROOT / "opendss"),
            "--feeder",
            str(ROOT / "feeder.npz"),
            "--design",
            str(ROOT / "physical_design.production.v1.json"),
            "--weight-source",
            str(weight),
            "--role",
            "self-test",
            "--seed",
            "12345",
            "--events",
            "20",
            "--steps-per-event",
            "12",
        )
        run(*common, "--output", str(first))
        run(*common, "--output", str(second))
        run(
            str(ROOT / "validate_opendss_truth.py"),
            str(first),
            "--feeder",
            str(ROOT / "feeder.npz"),
            "--expected-role",
            "self-test",
            "--expected-seed",
            "12345",
            "--expected-events",
            "20",
        )
        keys = (
            "x_true",
            "z_true",
            "z_physical",
            "time",
            "event_id",
            "drift_family",
            "is_nominal",
            "trajectory_id",
            "event_mechanism",
        )
        with np.load(first, allow_pickle=False) as a, np.load(
            second, allow_pickle=False
        ) as b, np.load(ROOT / "feeder.npz", allow_pickle=False) as feeder:
            for key in keys:
                if not np.array_equal(a[key], b[key]):
                    raise AssertionError(f"repeatability mismatch in {key}")
            families = set(a["drift_family"].astype(str).tolist())
            if families != {
                "nominal",
                "load_ramp",
                "parameter_change",
                "topology_change",
            }:
                raise AssertionError(f"event-family coverage is incomplete: {families}")
            linear = a["x_true"].astype(float) @ feeder["H_telemetry"].T
            nonlinear_difference = float(
                np.max(np.abs(a["z_true"].astype(float) - linear))
            )
            if nonlinear_difference < 1e-5:
                raise AssertionError("z_true unexpectedly collapsed to H_telemetry @ x_true")
        print(
            "OPENDSS_EXPORTER_SELF_TEST_OK",
            f"repeatable_arrays={len(keys)}",
            "events=20",
            "steps=240",
            f"nonlinear_vs_linear_max_diff={nonlinear_difference:.9g}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
