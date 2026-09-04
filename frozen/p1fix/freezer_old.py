#!/usr/bin/env python3
"""Freeze matched-FAR detector thresholds from calibration-only events.

Run this before inspecting any factor-campaign output. The script refuses to
overwrite an existing threshold file and records source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    twin_path = args.calibration_run / "twin" / "scores_events.parquet"
    oracle_path = args.calibration_run / "oracle" / "oracle_events.parquet"
    twin = pd.read_parquet(twin_path)
    oracle = pd.read_parquet(oracle_path)
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))

    if calibration.get("schema") != "twin.calibration.v2":
        raise ValueError("calibration JSON is not twin.calibration.v2")
    required_oracle = {"event_id", "regime"}
    if not required_oracle.issubset(oracle.columns):
        raise ValueError(f"oracle events lack {sorted(required_oracle - set(oracle.columns))}")

    if "is_nominal" in oracle:
        nominal = oracle["is_nominal"].astype(bool)
    elif "drift_family" in oracle:
        nominal = oracle["drift_family"].astype(str).eq("nominal")
    else:
        raise ValueError("oracle events do not identify nominal events")
    eligible_ids = set(
        oracle.loc[nominal & oracle["regime"].astype(str).eq("ample"), "event_id"].astype(int)
    )
    population = twin[twin["event_id"].astype(int).isin(eligible_ids)].copy()
    expected = int(calibration["selection"]["nominal_ample_events"])
    if len(population) != expected:
        raise AssertionError(
            f"nominal/ample cardinality mismatch: expected {expected}, got {len(population)}"
        )

    candidates = (
        "s", "s_lmax", "s_trace", "chi2", "huber", "lnr",
        "sB1", "sB2", "s_hyb_b1", "s_hyb_b2",
        "s_gated_lmax", "s_gated_trace", "s_gated",
    )
    thresholds: dict[str, dict[str, float | int]] = {}
    for metric in candidates:
        if metric not in population:
            continue
        values = pd.to_numeric(population[metric], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            continue
        threshold = float(
            np.quantile(values, 1.0 - args.target_far, method="higher")
        )
        exceedances = int(np.sum(values > threshold))
        thresholds[metric] = {
            "threshold": threshold,
            "exceedances": exceedances,
            "population": int(len(values)),
            "realized_far": float(exceedances / len(values)),
        }

    if "s" not in thresholds or "chi2" not in thresholds:
        raise AssertionError("both proposed s and estimator-matched chi2 are required")

    frozen_score = float(calibration["threshold"]["score_threshold"])
    calculated_score = float(thresholds["s"]["threshold"])
    score_match = bool(np.isclose(frozen_score, calculated_score, rtol=0.0, atol=1e-12))

    result = {
        "schema": "paper1.matched_far.thresholds.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "calibration_only",
        "selection_mask": "is_nominal == true and regime == ample",
        "target_far": args.target_far,
        "quantile": 1.0 - args.target_far,
        "quantile_method": "higher",
        "alarm_rule": "score > threshold",
        "population": int(len(population)),
        "thresholds": thresholds,
        "proposed_score_threshold_in_calibration_json": frozen_score,
        "proposed_score_threshold_recomputed": calculated_score,
        "proposed_threshold_exact_match": score_match,
        "sources": {
            "twin_events": str(twin_path.resolve()),
            "twin_events_sha256": sha256(twin_path),
            "oracle_events": str(oracle_path.resolve()),
            "oracle_events_sha256": sha256(oracle_path),
            "calibration_json": str(args.calibration_json.resolve()),
            "calibration_json_sha256": sha256(args.calibration_json),
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
    print("OUTPUT=", args.output.resolve())
    print("SHA256=", sha256(args.output))
    print("POPULATION=", len(population))
    print("METRICS=", ",".join(sorted(thresholds)))


if __name__ == "__main__":
    main()

