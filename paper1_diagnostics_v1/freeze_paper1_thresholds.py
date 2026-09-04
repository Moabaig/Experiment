#!/usr/bin/env python3
"""Freeze Paper-1 matched-FAR thresholds from calibration-only step scores.

The calibration run was executed before the final normalizers were frozen, so
its stored ``s`` column is on the pre-calibration scale.  This utility therefore
reconstructs every normalized score from step-level ``r``, ``u``, ``b1`` and
``b2`` using ``calibration.v2.json``, then takes the exact within-event maximum.
Raw baselines such as ``chi2`` are aggregated from the same step table without
rescaling.  The output remains the downstream-compatible v1 threshold contract.

Run this before any Paper-1 v4 campaign cell.  Existing output is never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"calibration step table is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"calibration column {column} contains nonfinite values")
    return values


def event_max(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return series.groupby(frame["event_id"], sort=True).max()


def event_constant(frame: pd.DataFrame, column: str) -> pd.Series:
    counts = frame.groupby("event_id", sort=True)[column].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"{column} changes within {len(bad)} calibration events")
    return frame.groupby("event_id", sort=True)[column].first()


def threshold_record(values: np.ndarray, target_far: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("threshold population is empty or nonfinite")
    threshold = float(np.quantile(values, 1.0 - target_far, method="higher"))
    exceedances = int(np.sum(values > threshold))
    return {
        "threshold": threshold,
        "exceedances": exceedances,
        "population": int(values.size),
        "realized_far": float(exceedances / values.size),
    }


def require_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-far", type=float, default=0.01)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen thresholds: {args.output}")
    if not 0.0 < args.target_far < 0.5:
        raise ValueError("target FAR must be between zero and 0.5")

    step_path = args.calibration_run / "oracle" / "oracle_scores.parquet"
    oracle_event_path = args.calibration_run / "oracle" / "oracle_events.parquet"
    twin_event_path = args.calibration_run / "twin" / "scores_events.parquet"
    for path in (step_path, oracle_event_path, twin_event_path, args.calibration_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    steps = pd.read_parquet(step_path).copy()
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))

    if calibration.get("schema") != "twin.calibration.v2":
        raise ValueError("calibration JSON is not twin.calibration.v2")
    if calibration.get("threshold", {}).get("event_score_construction") != (
        "max_step(r/r0 + u/u0)"
    ):
        raise ValueError("calibration JSON uses an incompatible event score")
    if calibration.get("threshold", {}).get("scalarization") not in {"lmax", "trace"}:
        raise ValueError("calibration scalarization is invalid")

    required = {
        "step_index", "event_id", "regime", "is_nominal",
        "r", "u_lmax", "u_trace", "b1", "b2", "chi2",
    }
    missing = sorted(required - set(steps.columns))
    if missing:
        raise ValueError(f"calibration step table lacks {missing}")
    if len(steps) != int(calibration["selection"]["all_steps"]):
        raise ValueError("calibration step cardinality does not match calibration JSON")
    if steps["step_index"].duplicated().any():
        raise ValueError("calibration step_index contains duplicates")

    event_sizes = steps.groupby("event_id", sort=True).size()
    expected_events = int(calibration["selection"]["all_events"])
    if len(event_sizes) != expected_events:
        raise ValueError("calibration event cardinality does not match calibration JSON")
    if not (event_sizes == 12).all():
        raise ValueError("every calibration event must contain exactly 12 steps")

    regime = event_constant(steps, "regime").astype(str)
    nominal = event_constant(steps, "is_nominal").astype(bool)
    eligible = nominal & regime.eq("ample")
    expected_population = int(calibration["selection"]["nominal_ample_events"])
    if int(eligible.sum()) != expected_population:
        raise ValueError(
            f"nominal/ample cardinality mismatch: expected {expected_population}, "
            f"got {int(eligible.sum())}"
        )

    r0 = require_positive(calibration["normalizers"]["r0"], "r0")
    u0_lmax = require_positive(calibration["normalizers"]["u0_lmax"], "u0_lmax")
    u0_trace = require_positive(calibration["normalizers"]["u0_trace"], "u0_trace")
    beta1 = float(calibration["hybrid_calibration"]["beta1"])
    beta2 = float(calibration["hybrid_calibration"]["beta2"])
    if not all(math.isfinite(value) and value >= 0.0 for value in (beta1, beta2)):
        raise ValueError("hybrid calibration coefficients are invalid")

    r = numeric(steps, "r")
    u_lmax = numeric(steps, "u_lmax")
    u_trace = numeric(steps, "u_trace")
    b1 = numeric(steps, "b1")
    b2 = numeric(steps, "b2")
    residual = r / r0
    s_lmax = residual + u_lmax / u0_lmax
    s_trace = residual + u_trace / u0_trace
    scalarization = calibration["threshold"]["scalarization"]
    active_s = s_lmax if scalarization == "lmax" else s_trace
    s_b1 = residual + beta1 * (1.0 - b1)
    s_b2 = residual + beta2 * b2
    s_gated_lmax = b1 * residual + u_lmax / u0_lmax
    s_gated_trace = b1 * residual + u_trace / u0_trace
    active_gated = s_gated_lmax if scalarization == "lmax" else s_gated_trace

    step_metrics: dict[str, np.ndarray] = {
        "s": active_s,
        "s_lmax": s_lmax,
        "s_trace": s_trace,
        "sB1": s_b1,
        "sB2": s_b2,
        "s_hyb_b1": s_b1,
        "s_hyb_b2": s_b2,
        "s_gated": active_gated,
        "s_gated_lmax": s_gated_lmax,
        "s_gated_trace": s_gated_trace,
        "chi2": numeric(steps, "chi2"),
    }
    skipped_optional_metrics: dict[str, dict[str, int | str]] = {}
    for optional in ("huber", "lnr"):
        if optional not in steps.columns:
            skipped_optional_metrics[optional] = {
                "reason": "column_missing",
                "total_steps": int(len(steps)),
                "nonfinite_steps": int(len(steps)),
            }
            continue

        optional_values = pd.to_numeric(
            steps[optional], errors="coerce"
        ).to_numpy(float)
        nonfinite_steps = int((~np.isfinite(optional_values)).sum())
        if nonfinite_steps:
            skipped_optional_metrics[optional] = {
                "reason": "column_contains_nonfinite_values",
                "total_steps": int(len(optional_values)),
                "nonfinite_steps": nonfinite_steps,
            }
            continue

        step_metrics[optional] = optional_values

    thresholds: dict[str, dict[str, float | int]] = {}
    for name, step_values in step_metrics.items():
        maxima = event_max(steps, step_values)
        if not maxima.index.equals(eligible.index):
            raise ValueError(f"event alignment failed for {name}")
        thresholds[name] = threshold_record(
            maxima.loc[eligible].to_numpy(float), args.target_far
        )

    frozen_score = float(calibration["threshold"]["score_threshold"])
    calculated_score = float(thresholds["s"]["threshold"])
    score_match = bool(np.isclose(frozen_score, calculated_score, rtol=0.0, atol=1e-12))
    if not score_match:
        raise AssertionError(
            "exact step reconstruction does not reproduce calibration.v2 score threshold: "
            f"{calculated_score:.17g} != {frozen_score:.17g}"
        )

    calibration_source = calibration.get("sources", [{}])[0]
    if calibration_source.get("sha256") != sha256(step_path):
        raise AssertionError("calibration.v2 source hash does not match oracle step table")

    result = {
        "schema": "paper1.matched_far.thresholds.v1",
        "implementation_revision": "exact_step_reconstruction.v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "calibration_only",
        "selection_mask": "is_nominal == true and regime == ample",
        "target_far": args.target_far,
        "quantile": 1.0 - args.target_far,
        "quantile_method": "higher",
        "alarm_rule": "score > threshold",
        "event_score_construction": "max_step(metric)",
        "population": expected_population,
        "thresholds": thresholds,
        "skipped_optional_metrics": skipped_optional_metrics,
        "proposed_score_threshold_in_calibration_json": frozen_score,
        "proposed_score_threshold_recomputed": calculated_score,
        "proposed_threshold_exact_match": score_match,
        "sources": {
            "oracle_scores": str(step_path.resolve()),
            "oracle_scores_sha256": sha256(step_path),
            "twin_events": str(twin_event_path.resolve()),
            "twin_events_sha256": sha256(twin_event_path),
            "oracle_events": str(oracle_event_path.resolve()),
            "oracle_events_sha256": sha256(oracle_event_path),
            "calibration_json": str(args.calibration_json.resolve()),
            "calibration_json_sha256": sha256(args.calibration_json),
        },
        "pre_campaign_correction": {
            "reason": (
                "stored calibration-run s used the pre-calibration score scale; "
                "thresholds are reconstructed from step components with frozen normalizers"
            ),
            "method_changed": False,
            "calibration_changed": False,
            "evaluation_data_used": False,
        },
        "prohibitions": [
            "do not recompute thresholds from evaluation or factor-campaign outputs",
            "do not count abstention as a correct drift detection",
            "do not change gamma after this file is frozen",
        ],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PAPER1_MATCHED_FAR_THRESHOLDS_FROZEN_OK")
    print("IMPLEMENTATION_REVISION=exact_step_reconstruction.v2.1")
    print("OUTPUT=", args.output.resolve())
    print("SHA256=", sha256(args.output))
    print("POPULATION=", expected_population)
    print("S_THRESHOLD=", calculated_score)
    print("CHI2_THRESHOLD=", thresholds["chi2"]["threshold"])
    print("METRICS=", ",".join(sorted(thresholds)))
    print("SKIPPED_OPTIONAL_METRICS=", ",".join(sorted(skipped_optional_metrics)))


if __name__ == "__main__":
    main()
