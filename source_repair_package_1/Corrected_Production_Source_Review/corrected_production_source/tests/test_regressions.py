from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_twin
import oracle_fed
import power_fed

# The review archive intentionally contains only the production files under
# repair, while twin_fed imports the separately distributed trust_metric
# module.  The calibration-schema test below does not instantiate that metric,
# so provide a narrow import stub when the dependency is absent.
if "trust_metric" not in sys.modules:
    try:
        __import__("trust_metric")
    except ModuleNotFoundError:
        trust_metric_stub = types.ModuleType("trust_metric")
        trust_metric_stub.MetricConfig = object
        trust_metric_stub.TrustMetric = object
        sys.modules["trust_metric"] = trust_metric_stub
import twin_fed


class EventAggregationTests(unittest.TestCase):
    def test_oracle_uses_minimum_trust_and_maximum_score(self) -> None:
        scores = np.array([1.0, 3.0])
        frame = pd.DataFrame(
            {
                "event_id": [0, 0],
                "pattern_id": [0, 0],
                "arm": ["G", "G"],
                "regime": ["severe", "severe"],
                "stratum": [-1, -1],
                "drift_family": ["load_ramp", "load_ramp"],
                "trajectory_id": ["trajectory-0", "trajectory-0"],
                "label": [False, True],
                "d": [2.0, 4.0],
                "is_nominal": [False, False],
                "s": scores,
                "s_lmax": scores,
                "T": np.exp(-scores),
                "r": [0.2, 0.5],
                "u_lmax": [0.8, 2.5],
                "b1": [0.9, 0.4],
                "b2": [1.0, 5.0],
                "n_rx": [500, 300],
                "n_rx_telemetry": [40, 20],
                "held": [False, True],
                "solve_exact": [True, False],
                "loss_quantile": [1, 3],
                "step_index": [0, 1],
                "time": [1.0, 2.0],
            }
        )
        event = oracle_fed.aggregate_events(frame).iloc[0]
        self.assertEqual(event["s"], 3.0)
        self.assertEqual(event["T"], math.exp(-3.0))
        self.assertEqual(event["b1"], 0.4)
        self.assertEqual(event["n_rx"], 300)
        self.assertEqual(event["n_rx_telemetry"], 20)
        self.assertEqual(event["d"], 4.0)
        self.assertTrue(event["label"])
        self.assertEqual(event["held"], 0.5)
        self.assertTrue(event["held_any"])
        self.assertFalse(event["solve_exact"])
        self.assertEqual(event["solve_exact_fraction"], 0.5)


class CalibrationTests(unittest.TestCase):
    def test_combined_event_score_is_maximum_of_stepwise_sum(self) -> None:
        data = pd.DataFrame(
            {
                "_event_key": ["0:0", "0:0", "0:1", "0:1"],
                "_source_id": [0, 0, 0, 0],
                "event_id": [0, 0, 1, 1],
                "step_index": [0, 1, 2, 3],
                "regime": ["ample", "ample", "moderate", "moderate"],
                "is_nominal": [True, True, True, True],
                "drift_family": ["nominal"] * 4,
                "label": [False] * 4,
                "r": [10.0, 0.0, 1.0, 2.0],
                "u_lmax": [0.0, 10.0, 3.0, 4.0],
                "u_trace": [0.0, 20.0, 6.0, 8.0],
                "b1": [1.0, 0.5, 0.8, 0.7],
                "b2": [0.0, 2.0, 1.0, 3.0],
            }
        )
        events = calibrate_twin.build_event_frame(data)
        step_score = data["r"].to_numpy() + data["u_lmax"].to_numpy()
        event_score = calibrate_twin._event_max(data, step_score, events.index)
        np.testing.assert_allclose(event_score, [10.0, 6.0])
        self.assertEqual(events.loc["0:0", "r"] + events.loc["0:0", "u_lmax"], 20.0)

    def test_legacy_calibration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"schema": "twin.calibration.v1"}))
            with self.assertRaisesRegex(ValueError, "unsupported calibration schema"):
                twin_fed.load_calibration(path)
            path.write_text(json.dumps({"schema": "twin.calibration.v2"}))
            self.assertEqual(twin_fed.load_calibration(path)["schema"], "twin.calibration.v2")

    def test_calibration_cli_writes_step_derived_v2_contract(self) -> None:
        rows = []
        for event_id in range(60):
            regime = "ample" if event_id < 40 else "moderate"
            for offset in range(2):
                rows.append(
                    {
                        "step_index": 2 * event_id + offset,
                        "event_id": event_id,
                        "regime": regime,
                        "is_nominal": True,
                        "drift_family": "nominal",
                        "label": event_id in {5, 45},
                        "r": 1.0 + 0.01 * event_id + 0.02 * offset,
                        "u_lmax": 2.0 + 0.03 * event_id + 0.01 * offset,
                        "u_trace": 4.0 + 0.05 * event_id + 0.01 * offset,
                        "b1": 0.9 - 0.001 * event_id,
                        "b2": 0.1 + 0.002 * event_id,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "oracle_scores.csv"
            output = Path(directory) / "calibration.json"
            pd.DataFrame(rows).to_csv(source, index=False)
            self.assertEqual(
                calibrate_twin.main(
                    ["--input", str(source), "--output", str(output)]
                ),
                0,
            )
            frozen = json.loads(output.read_text())
            self.assertEqual(frozen["schema"], "twin.calibration.v2")
            self.assertEqual(
                frozen["threshold"]["event_score_construction"],
                "max_step(r/r0 + u/u0)",
            )
            self.assertEqual(frozen["selection"]["all_steps"], 120)

    def test_event_level_calibration_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "oracle_events.csv"
            pd.DataFrame({"event_id": [0]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "event-level"):
                calibrate_twin.main(["--input", str(source)])


class TruthContractTests(unittest.TestCase):
    def _write_inputs(self, directory: Path, *, include_z: bool) -> tuple[Path, Path]:
        feeder = directory / "feeder.npz"
        truth = directory / "truth.npz"
        np.savez(
            feeder,
            H=np.array([[1.0], [2.0]]),
            sigma2=np.array([0.1, 0.2]),
            n_telemetry=np.array(1),
        )
        values = {
            "x_true": np.array([[1.0], [2.0]]),
            "time": np.array([1.0, 2.0]),
        }
        if include_z:
            values["z_true"] = np.array([[1.1], [2.2]])
        np.savez(truth, **values)
        return feeder, truth

    def test_production_requires_physical_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feeder, truth = self._write_inputs(Path(directory), include_z=False)
            with self.assertRaisesRegex(ValueError, "missing z_true"):
                power_fed.load_inputs(feeder, truth, dt=1.0)
            loaded = power_fed.load_inputs(
                feeder, truth, dt=1.0, allow_linearized_telemetry=True
            )
            self.assertEqual(loaded[-1], "linearized_smoke_fallback")

    def test_physical_telemetry_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feeder, truth = self._write_inputs(Path(directory), include_z=True)
            loaded = power_fed.load_inputs(feeder, truth, dt=1.0)
            self.assertEqual(loaded[-1], "truth.z_true")


if __name__ == "__main__":
    unittest.main()
