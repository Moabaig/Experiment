#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import paper1_final_results as paper1


class StatisticalPrimitiveTests(unittest.TestCase):
    def test_auc_uses_half_credit_for_ties(self) -> None:
        self.assertAlmostEqual(
            paper1.empirical_auc([False, False, True, True], [0.0, 1.0, 1.0, 2.0]),
            0.875,
        )

    def test_roc_has_origin_and_monotone_coordinates(self) -> None:
        frame = paper1.roc_points([False, True, False, True], [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(tuple(frame.iloc[0][["fpr", "tpr"]]), (0.0, 0.0))
        self.assertTrue((np.diff(frame["fpr"]) >= 0).all())
        self.assertTrue((np.diff(frame["tpr"]) >= 0).all())

    def test_q_is_fraction_of_discordant_pairs_tied_on_control(self) -> None:
        y = np.array([False, False, True, True])
        score = np.array([0, 1, 0, 2])
        self.assertAlmostEqual(paper1.q_tie_fraction(y, score), 0.25)
        self.assertAlmostEqual(1.0 - paper1.q_tie_fraction(y, score) / 2.0, 0.875)

    def test_holm_is_monotone_in_sorted_order(self) -> None:
        adjusted = paper1.holm([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])


class GateAndRendererTests(unittest.TestCase):
    def test_partial_raw_campaign_is_analyzed_but_not_rendered_as_final(self) -> None:
        package_root = Path(__file__).resolve().parent.parent
        design = (
            package_root
            / "restored_package"
            / "paper1_diagnostics_v1"
            / "production_overlay"
            / "factor_design.paper1.v4.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "paper1_s002_bw00_floor"
            for service in ("twin", "oracle", "net"):
                (run / service).mkdir(parents=True, exist_ok=True)
            threshold_path = root / "thresholds.json"
            threshold_path.write_text(
                json.dumps(
                    {
                        "schema": "paper1.matched_far.thresholds.v1",
                        "source_split": "calibration_only",
                        "target_far": 0.01,
                        "quantile_method": "higher",
                        "thresholds": {
                            "s": {"threshold": 0.5},
                            "chi2": {"threshold": 0.5},
                            "sB1": {"threshold": 0.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            event_id = np.arange(10)
            labels = event_id >= 5
            twin_event = pd.DataFrame(
                {
                    "event_id": event_id,
                    "s": np.linspace(0, 1, 10),
                    "chi2": np.linspace(0.1, 0.8, 10),
                    "sB1": np.linspace(0.05, 0.9, 10),
                    "b1": np.linspace(0.4, 0.9, 10),
                    "b2": np.linspace(1.0, 0.1, 10),
                    "n_rx_telemetry": np.arange(10) + 20,
                    "held_any": False,
                    "residual_available": True,
                    "arm": ["G"] * 10,
                    "regime": ["moderate"] * 10,
                }
            )
            oracle_event = pd.DataFrame(
                {
                    "event_id": event_id,
                    "label": labels,
                    "is_nominal": ~labels,
                    "drift_family": np.where(labels, "load_ramp", "nominal"),
                    "d": np.where(labels, 2.0, 0.0),
                }
            )
            step_rows = []
            oracle_step_rows = []
            for event, label in zip(event_id, labels):
                for offset in range(2):
                    step_rows.append(
                        {
                            "step_index": 2 * event + offset,
                            "event_id": event,
                            "s": float(event / 9 + offset * 0.01),
                            "chi2": float(event / 12 + offset * 0.01),
                            "sB1": float(event / 10 + offset * 0.01),
                        }
                    )
                    oracle_step_rows.append(
                        {
                            "step_index": 2 * event + offset,
                            "event_id": event,
                            "label": bool(label),
                            "d": 2.0 if label else 0.0,
                        }
                    )
            twin_event.to_csv(run / "twin" / "scores_events.csv", index=False)
            oracle_event.to_csv(run / "oracle" / "oracle_events.csv", index=False)
            pd.DataFrame(step_rows).to_csv(run / "twin" / "scores.csv", index=False)
            pd.DataFrame(oracle_step_rows).to_csv(run / "oracle" / "oracle_scores.csv", index=False)
            (run / "net" / "meta.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "counts": {
                            "received": 100,
                            "delivered": 80,
                            "dropped_random": 10,
                            "dropped_starved": 10,
                            "dropped_queue": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            paper1.analyze_confirmatory(
                runs_root=runs,
                design_path=design,
                threshold_path=threshold_path,
                output=output,
                seeds=[2],
                draws=50,
                allow_partial=True,
            )
            manifest = json.loads((output / "paper1_analysis_manifest.json").read_text())
            self.assertEqual(manifest["status"], "partial_nonconfirmatory")
            self.assertEqual(manifest["cells_found"], 1)
            self.assertTrue((output / "paper1_arm_paired_contrasts.csv").exists())

    def test_renderer_rejects_partial_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper1_analysis_manifest.json").write_text(
                json.dumps({"status": "partial_nonconfirmatory"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "confirmatory_complete"):
                paper1.render_confirmatory(root)

    def test_confirmatory_renderer_builds_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper1_analysis_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "confirmatory_complete",
                        "cells_found": 150,
                        "confirmatory_seed_indices": list(range(2, 32)),
                    }
                ),
                encoding="utf-8",
            )
            summary_rows = []
            for bandwidth_index, bandwidth in enumerate(paper1.BANDWIDTH_ORDER):
                for metric_index, metric in enumerate(("s", "chi2", "sB1", "sB2", "s_gated_lmax")):
                    for outcome, value in (
                        ("auc", 0.62 - 0.02 * metric_index + 0.01 * bandwidth_index),
                        ("recall", 0.50 - 0.03 * metric_index),
                        ("false_alarm_rate", 0.01 + 0.002 * metric_index),
                        ("residual_silence_rate", 0.30),
                        ("abstention_rate", 0.05),
                    ):
                        summary_rows.append(
                            {
                                "bandwidth_level": bandwidth,
                                "metric": metric,
                                "outcome": outcome,
                                "seeds": 30,
                                "mean": value,
                                "median": value,
                                "ci95_low": max(0.0, value - 0.02),
                                "ci95_high": min(1.0, value + 0.02),
                            }
                        )
            pd.DataFrame(summary_rows).to_csv(root / "paper1_cluster_summary.csv", index=False)

            contrast_rows = []
            for bandwidth in paper1.BANDWIDTH_ORDER:
                for comparator in ("chi2", "sB1", "sB2", "s_gated_lmax"):
                    for outcome in ("auc", "recall", "false_alarm_rate"):
                        contrast_rows.append(
                            {
                                "outcome": outcome,
                                "metric": "s",
                                "comparator": comparator,
                                "bandwidth_level": bandwidth,
                                "paired_seeds": 30,
                                "mean_paired_difference": 0.02,
                                "median_paired_difference": 0.02,
                                "ci95_low": 0.01,
                                "ci95_high": 0.03,
                                "wilcoxon_p_raw": 0.01,
                                "wilcoxon_p_holm": 0.05,
                            }
                        )
            pd.DataFrame(contrast_rows).to_csv(root / "paper1_paired_metric_contrasts.csv", index=False)

            latency_rows = []
            for bandwidth in paper1.BANDWIDTH_ORDER:
                for metric in ("s", "chi2", "sB1"):
                    for event_id in range(20):
                        latency_rows.append(
                            {
                                "seed_index": 2 + event_id % 5,
                                "bandwidth_level": bandwidth,
                                "metric": metric,
                                "event_id": event_id,
                                "residual_silent": True,
                                "detected": metric != "chi2",
                                "latency_steps": event_id % 5 if metric != "chi2" else np.nan,
                            }
                        )
            pd.DataFrame(latency_rows).to_csv(root / "paper1_latency_events.csv", index=False)
            pd.DataFrame(
                [{"bandwidth_level": bandwidth, "metric": metric, "residual_silent": True, "outcome": "detection_fraction", "mean": 0.5, "ci95_low": 0.4, "ci95_high": 0.6, "seeds": 30}
                 for bandwidth in paper1.BANDWIDTH_ORDER for metric in ("s", "chi2", "sB1")]
            ).to_csv(root / "paper1_latency_summary.csv", index=False)

            roc_rows = []
            for bandwidth in paper1.BANDWIDTH_ORDER:
                for metric in ("s", "chi2", "sB1"):
                    for point in np.linspace(0, 1, 11):
                        roc_rows.append({"threshold": 1 - point, "fpr": point, "tpr": min(1.0, point + 0.1), "bandwidth_level": bandwidth, "metric": metric})
            pd.DataFrame(roc_rows).to_csv(root / "paper1_roc_points.csv", index=False)

            pd.DataFrame(
                [{"seed_index": seed, "bandwidth_level": bandwidth, "packets_received": 100, "packets_delivered": 80, "realized_drop_fraction": 0.2, "dropped_random": 10, "dropped_starved": 10, "dropped_queue": 0, "mean_b1": 0.8, "median_b1": 0.8, "mean_b2": 0.2, "p90_b2": 0.3, "held_rate": 0.05}
                 for seed in range(2, 7) for bandwidth in paper1.BANDWIDTH_ORDER]
            ).to_csv(root / "paper1_network_conditions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dimension": dimension,
                        "condition_bin": condition_bin,
                        "condition_mean": value,
                        "metric": metric,
                        "outcome": outcome,
                        "seeds": 30,
                        "mean": 0.05 if outcome == "false_alarm_rate" else 0.25,
                        "ci95_low": 0.03 if outcome == "false_alarm_rate" else 0.20,
                        "ci95_high": 0.07 if outcome == "false_alarm_rate" else 0.30,
                    }
                    for dimension in ("loss_proxy", "mean_age")
                    for condition_bin, value in (("low", 0.1), ("high", 0.8))
                    for metric in ("s", "chi2", "sB1")
                    for outcome in ("false_alarm_rate", "missed_drift_rate")
                ]
            ).to_csv(root / "paper1_communication_bin_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "regime": regime,
                        "metric": "s",
                        "outcome": "residual_silence_rate",
                        "seeds": 30,
                        "mean": value,
                        "ci95_low": value - 0.03,
                        "ci95_high": value + 0.03,
                    }
                    for regime, value in (("ample", 0.1), ("moderate", 0.3), ("severe", 0.6))
                ]
            ).to_csv(root / "paper1_regime_cluster_summary.csv", index=False)

            condition_rows = []
            arm_contrast_rows = []
            for bandwidth in paper1.BANDWIDTH_ORDER:
                for metric in ("s", "chi2", "sB1", "sB2", "s_gated_lmax"):
                    condition_rows.append({"bandwidth_level": bandwidth, "arm": "G", "regime": "ALL", "drift_family": "ALL", "metric": metric, "outcome": "auc", "seeds": 30, "mean": 0.6, "median": 0.6, "ci95_low": 0.55, "ci95_high": 0.65})
                arm_contrast_rows.append({"arm": "G", "outcome": "auc", "metric": "s", "comparator": "sB1", "bandwidth_level": bandwidth, "paired_seeds": 30, "mean_paired_difference": 0.01, "median_paired_difference": 0.01, "ci95_low": -0.01, "ci95_high": 0.03, "wilcoxon_p_raw": 0.2, "wilcoxon_p_holm": 1.0})
            pd.DataFrame(condition_rows).to_csv(root / "paper1_condition_cluster_summary.csv", index=False)
            pd.DataFrame(arm_contrast_rows).to_csv(root / "paper1_arm_paired_contrasts.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "seed_index": seed,
                        "bandwidth_level": bandwidth,
                        "arm": arm,
                        "spearman_b1_u_lmax": -0.75,
                        "q_b1_tied_pos_neg_pairs": 0.4,
                        "auc_b1_ceiling": 0.8,
                        "q_b1_b2_tied_pos_neg_pairs": 0.2,
                    }
                    for seed in range(2, 32)
                    for bandwidth in paper1.BANDWIDTH_ORDER
                    for arm in ("C", "G", "T")
                ]
            ).to_csv(root / "paper1_collinearity_and_q.csv", index=False)

            paper1.render_confirmatory(root)
            self.assertTrue((root / "figures" / "fig_confirmatory_roc_by_bandwidth.pdf").exists())
            self.assertTrue((root / "figures" / "fig_confirmatory_collinearity_q.pdf").exists())
            self.assertTrue((root / "latex" / "table_confirmatory_f3.tex").exists())
            self.assertTrue((root / "latex" / "table_confirmatory_collinearity_q.tex").exists())
            self.assertTrue((root / "latex" / "paper1_confirmatory_results.tex").exists())
            self.assertTrue((root / "latex" / "paper1_confirmatory_abstract.tex").exists())
            self.assertTrue((root / "latex" / "paper1_confirmatory_discussion.tex").exists())
            self.assertTrue((root / "latex" / "paper1_confirmatory_conclusion.tex").exists())
            headline = json.loads((root / "paper1_confirmatory_headline.json").read_text())
            self.assertEqual(headline["cells"], 150)
            self.assertAlmostEqual(headline["severe_residual_silence_rate"], 0.6)
            self.assertTrue((root / "paper1_output_sha256.json").exists())


if __name__ == "__main__":
    unittest.main()
