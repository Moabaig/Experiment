from __future__ import annotations

import json
import math
import types
import unittest
from pathlib import Path

import numpy as np

import twin_fed


class BandwidthFactorTests(unittest.TestCase):
    def test_cap_preserves_starvation_and_caps_feasible_links(self) -> None:
        original = np.array([[0.5, 1.0e12, 10_000.0]])
        capped = twin_fed.apply_bandwidth_cap(original, 100_000.0)
        np.testing.assert_array_equal(
            capped,
            np.array([[0.5, 100_000.0, 10_000.0]]),
        )
        self.assertLessEqual(capped[0, 0], 1.0)
        self.assertGreater(capped[0, 1], 1.0)

    def test_oracle_cap_is_backward_compatible_for_frozen_patterns(self) -> None:
        original = np.array([[0.5, 1.0e12]])
        np.testing.assert_array_equal(
            twin_fed.apply_bandwidth_cap(original, 1.0e12),
            original,
        )

    def test_invalid_cap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            twin_fed.apply_bandwidth_cap(np.ones((1, 1)), 0.0)

    def test_parser_records_factor_arguments(self) -> None:
        args = twin_fed.build_parser().parse_args(
            [
                "--bandwidth-level",
                "bw02_100kbps",
                "--bandwidth-cap-bps",
                "100000",
                "--compute-delta-check",
                "--allow-uncalibrated",
                "--validate-only",
            ]
        )
        twin_fed.check_args(args)
        self.assertEqual(args.bandwidth_level, "bw02_100kbps")
        self.assertEqual(args.bandwidth_cap_bps, 100_000.0)
        self.assertTrue(args.compute_delta_check)


class DualExposureTests(unittest.TestCase):
    def test_mean_and_delta_are_computed_on_one_realization(self) -> None:
        cfg = types.SimpleNamespace(
            delta=0.05,
            omega=1.0,
            eps=1.0e-3,
        )
        metric = types.SimpleNamespace(
            H=np.array([[1.0], [2.0]]),
            s2=np.array([1.0, 1.0]),
            cfg=cfg,
            nu_l=1.0,
            nu_t=1.0,
            L0=0.0,
            T0=0.0,
        )
        exposure = twin_fed.FastExposure(
            metric,
            exposure_form="mean",
            compute_delta_check=True,
        )
        result = exposure.evaluate(
            np.array([0.1, 0.2]),
            np.array([1.0, 1.0]),
        )
        self.assertEqual(result["floor_kind"], "mean")
        self.assertEqual(result["u_lmax"], result["u_lmax_mean"])
        self.assertTrue(math.isfinite(result["u_lmax_delta"]))
        self.assertIn(
            result["delta_floor_kind"],
            {"bernstein", "loss_quantile"},
        )


class FactorDesignContractTests(unittest.TestCase):
    def test_factor_design_has_150_predeclared_cells(self) -> None:
        root = Path(__file__).resolve().parents[1]
        design = json.loads(
            (root / "factor_design.production.v3.json").read_text()
        )
        self.assertEqual(design["schema"], "twin.factor.design.v3")
        self.assertEqual(design["seed_policy"]["count"], 30)
        self.assertEqual(len(design["bandwidth_levels"]), 5)
        self.assertEqual(design["campaign_cells"], 150)
        self.assertFalse(
            design["primary_evaluation"]["superseded_by_this_campaign"]
        )


if __name__ == "__main__":
    unittest.main()
