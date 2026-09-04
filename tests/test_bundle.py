from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import calibrate_twin
import oracle_fed
import twin_fed


class ProtocolTests(unittest.TestCase):
    def test_telemetry_contract_accepts_routing_fields(self) -> None:
        payload = {
            "schema": "twin.telemetry.v1",
            "channel_id": 3,
            "value": 1.25,
            "source_time": 4.0,
            "sequence": 4,
            "step_index": 3,
            "event_id": 0,
        }
        sample = twin_fed.parse_telemetry_payload(
            json.dumps(payload), arrival_time=4.0, n_telemetry=45,
            future_tolerance=1.0e-9,
        )
        self.assertEqual(sample.channel_id, 3)
        self.assertEqual(sample.sequence, 4)

    def test_telemetry_rejects_future_source(self) -> None:
        payload = {
            "schema": "twin.telemetry.v1", "channel_id": 0, "value": 1.0,
            "source_time": 2.0, "sequence": 0,
        }
        with self.assertRaises(twin_fed.ProtocolError):
            twin_fed.parse_telemetry_payload(
                json.dumps(payload), arrival_time=1.0, n_telemetry=45,
                future_tolerance=1.0e-9,
            )

    def test_score_contract_requires_state(self) -> None:
        payload = {
            "schema": "twin.score.v1", "step_index": 0, "event_id": 0,
            "time": 1.0, "s": 0.2, "r": 0.1, "chi2": 0.1,
            "x_hat": [0.0, 1.0],
        }
        parsed = oracle_fed.parse_score(json.dumps(payload), n_state=2)
        np.testing.assert_array_equal(parsed["x_hat"], [0.0, 1.0])

    def test_event_labels_are_oracle_error_only(self) -> None:
        frame = pd.DataFrame(
            {
                "event_id": [0, 0, 1, 1], "pattern_id": [0, 0, 1, 1],
                "label": [False, True, False, False], "d": [0.1, 1.1, 0.2, 0.3],
                "is_nominal": [True, False, True, True], "s": [10.0, 0.0, 9.0, 8.0],
            }
        )
        events = oracle_fed.aggregate_events(frame)
        self.assertEqual(events["label"].tolist(), [True, False])
        self.assertEqual(events["d"].tolist(), [1.1, 0.3])


class CalibrationTests(unittest.TestCase):
    def test_auc_tie_convention(self) -> None:
        value = calibrate_twin.auc(
            np.array([True, True, False, False]),
            np.array([1.0, 2.0, 1.0, 0.0]),
        )
        self.assertAlmostEqual(value, 0.875)

    def test_beta_zero_when_feature_constant(self) -> None:
        beta, value = calibrate_twin.tune_beta(
            np.array([False, False, True, True]),
            np.array([0.0, 0.1, 0.9, 1.0]),
            np.ones(4),
        )
        self.assertEqual(beta, 0.0)
        self.assertEqual(value, 1.0)


class BundleStructureTests(unittest.TestCase):
    def test_compose_has_exact_federation(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        services = set(compose["services"])
        self.assertEqual(
            services,
            {"dev", "broker", "power-fed", "net-fed", "twin-fed", "oracle-fed"},
        )
        self.assertIn("--federates=4", compose["services"]["broker"]["command"])

    def test_network_source_uses_real_links_and_helics_dependency(self) -> None:
        source = (ROOT / "net_fed.cc").read_text()
        for token in (
            '"ns3/helics-module.h"', "PointToPointNetDevice", "DropTailQueue",
            "HELICS_FLAG_WAIT_FOR_CURRENT_TIME_UPDATE", 'ExtractUnsignedJson(payload, "event_id")',
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
