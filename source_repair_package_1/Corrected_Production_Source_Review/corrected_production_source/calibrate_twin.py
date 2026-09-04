#!/usr/bin/env python3
"""Freeze production Twin calibration from *step-level* Oracle scores.

The combined trust score is nonlinear at event level: an event score is the
maximum, over its steps, of ``r/r0 + u/u0``. It cannot be reconstructed from
separately aggregated event maxima of ``r`` and ``u``. Consequently this
utility rejects ``oracle_events.parquet`` and accepts only the step-level
``oracle_scores.parquet`` output from dedicated calibration runs.

Calibration order:

1. choose ``r0`` from nominal/ample event residual maxima;
2. choose each ``u0`` from nominal/moderate event exposure maxima;
3. recompute the frozen score at every step, then take its event maximum;
4. choose ``T_th`` for the requested nominal/ample event false-alarm rate;
5. tune hybrid controls using their exact step-to-event maxima.

Evaluation files must never be passed to this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    positive, negative = scores[labels], scores[~labels]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires positive and negative calibration events")
    favorable = ties = 0
    for start in range(0, positive.size, 512):
        comparison = positive[start : start + 512, None] - negative[None, :]
        favorable += int(np.count_nonzero(comparison > 0.0))
        ties += int(np.count_nonzero(comparison == 0.0))
    return (favorable + 0.5 * ties) / (positive.size * negative.size)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"calibration data is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"calibration column {column} contains nonfinite values")
    return values


def positive_quantile(values: np.ndarray, q: float, name: str) -> float:
    value = float(np.quantile(values, q, method="linear"))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} calibration statistic must be positive")
    return value


def _assert_event_constant(data: pd.DataFrame, column: str) -> None:
    counts = data.groupby("_event_key", sort=False)[column].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(
            f"{column} changes within {len(bad)} calibration events; "
            "truth/scenario metadata must be event-constant"
        )


def _event_max(
    data: pd.DataFrame,
    values: np.ndarray,
    event_index: pd.Index,
) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float), index=data.index)
    maxima = series.groupby(data["_event_key"], sort=False).max()
    return maxima.reindex(event_index).to_numpy(float)


def build_event_frame(data: pd.DataFrame) -> pd.DataFrame:
    for column in ("regime", "is_nominal", "drift_family"):
        _assert_event_constant(data, column)
    grouped = data.groupby("_event_key", sort=False)
    events = grouped.agg(
        source_id=("_source_id", "first"),
        event_id=("event_id", "first"),
        regime=("regime", "first"),
        is_nominal=("is_nominal", "all"),
        drift_family=("drift_family", "first"),
        label=("label", "any"),
        r=("r", "max"),
        u_lmax=("u_lmax", "max"),
        u_trace=("u_trace", "max"),
        b1=("b1", "min"),
        b2=("b2", "max"),
        steps=("step_index", "size"),
    )
    events.index.name = "_event_key"
    return events


def choose_u0(
    moderate: pd.DataFrame,
    *,
    u_column: str,
    r0: float,
    target_trust: float,
) -> float:
    residual = float(np.median(numeric(moderate, "r") / r0))
    exposure = float(np.median(numeric(moderate, u_column)))
    available = -math.log(target_trust) - residual
    if available <= 0.0:
        raise ValueError(
            f"residual alone exceeds the {target_trust:g} moderate-boundary target; "
            f"cannot calibrate {u_column} without changing the predeclared target"
        )
    if exposure <= 0.0 or not math.isfinite(exposure):
        raise ValueError(f"moderate-boundary median {u_column} must be positive")
    return exposure / available


def tune_beta_event(
    labels: np.ndarray,
    data: pd.DataFrame,
    event_index: pd.Index,
    residual_step: np.ndarray,
    feature_step: np.ndarray,
) -> tuple[float, float]:
    spread = float(np.std(feature_step))
    if spread <= 1.0e-15:
        event_residual = _event_max(data, residual_step, event_index)
        return 0.0, auc(labels, event_residual)
    reference = max(float(np.std(residual_step)) / spread, 1.0e-12)
    grid = np.r_[0.0, reference * np.logspace(-4.0, 4.0, 321)]
    best_beta, best_auc = 0.0, -math.inf
    for beta in grid:
        event_score = _event_max(
            data, residual_step + float(beta) * feature_step, event_index
        )
        value = auc(labels, event_score)
        if value > best_auc + 1.0e-15:
            best_beta, best_auc = float(beta), float(value)
    return best_beta, best_auc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze production twin calibration")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="step-level oracle_scores.parquet files from calibration runs",
    )
    parser.add_argument("--output", default="calibration.json")
    parser.add_argument("--scalarization", choices=("lmax", "trace"), default="lmax")
    parser.add_argument("--ample-regime", default="ample")
    parser.add_argument("--moderate-regime", default="moderate")
    parser.add_argument("--far", type=float, default=0.01)
    parser.add_argument("--nominal-trust", type=float, default=0.90)
    parser.add_argument("--moderate-target-trust", type=float, default=0.70)
    parser.add_argument("--normalizer-quantile", type=float, default=0.50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name, value in (
        ("far", args.far),
        ("nominal-trust", args.nominal_trust),
        ("moderate-target-trust", args.moderate_target_trust),
        ("normalizer-quantile", args.normalizer_quantile),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name} must lie in (0,1)")

    paths = [Path(value) for value in args.input]
    frames: list[pd.DataFrame] = []
    for source_id, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
        if "step_index" not in frame.columns:
            raise ValueError(
                f"{path} is event-level; use oracle/oracle_scores.parquet, "
                "not oracle_events.parquet"
            )
        if frame["step_index"].duplicated().any():
            raise ValueError(f"{path} contains duplicate step_index values")
        frame = frame.copy()
        frame["_source_id"] = source_id
        frame["_event_key"] = str(source_id) + ":" + frame["event_id"].astype(str)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    required = {
        "step_index", "event_id", "label", "regime", "is_nominal",
        "drift_family", "r", "u_lmax", "u_trace", "b1", "b2",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"calibration data is missing columns: {missing}")
    for column in ("r", "u_lmax", "u_trace", "b1", "b2"):
        numeric(data, column)

    events = build_event_frame(data)
    labels = events["label"].astype(bool).to_numpy()
    ample = events[
        (events["regime"] == args.ample_regime) & events["is_nominal"].astype(bool)
    ]
    moderate = events[
        (events["regime"] == args.moderate_regime)
        & events["is_nominal"].astype(bool)
    ]
    if len(ample) < 20:
        raise ValueError("need at least 20 nominal/ample calibration events")
    if len(moderate) < 20:
        raise ValueError("need at least 20 nominal moderate-boundary calibration events")
    if np.unique(labels).size != 2:
        raise ValueError("hybrid calibration requires both oracle label classes")

    r_stat = positive_quantile(
        numeric(ample, "r"), args.normalizer_quantile, "nominal residual"
    )
    r0 = r_stat / -math.log(args.nominal_trust)
    u0_lmax = choose_u0(
        moderate,
        u_column="u_lmax",
        r0=r0,
        target_trust=args.moderate_target_trust,
    )
    u0_trace = choose_u0(
        moderate,
        u_column="u_trace",
        r0=r0,
        target_trust=args.moderate_target_trust,
    )

    residual_step = numeric(data, "r") / r0
    exposure_column = "u_lmax" if args.scalarization == "lmax" else "u_trace"
    exposure_scale = u0_lmax if args.scalarization == "lmax" else u0_trace
    frozen_step_score = residual_step + numeric(data, exposure_column) / exposure_scale
    frozen_event_score = _event_max(data, frozen_step_score, events.index)
    ample_mask = events.index.isin(ample.index)
    nominal_score = frozen_event_score[ample_mask]
    score_threshold = float(
        np.quantile(nominal_score, 1.0 - args.far, method="higher")
    )
    T_th = math.exp(-score_threshold)
    if not 0.0 < T_th < 1.0:
        raise ValueError("derived T_th is outside (0,1)")
    realized_far = float(np.mean(nominal_score > score_threshold))

    beta1, auc_b1 = tune_beta_event(
        labels,
        data,
        events.index,
        residual_step,
        1.0 - numeric(data, "b1"),
    )
    beta2, auc_b2 = tune_beta_event(
        labels,
        data,
        events.index,
        residual_step,
        numeric(data, "b2"),
    )

    output = {
        "schema": "twin.calibration.v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "normalizers": {
            "r0": r0,
            "u0_lmax": u0_lmax,
            "u0_trace": u0_trace,
        },
        "threshold": {
            "T_th": T_th,
            "score_threshold": score_threshold,
            "target_far": args.far,
            "calibration_population": "nominal_ample_events",
            "realized_nominal_ample_far": realized_far,
            "realized_calibration_far": realized_far,
            "scalarization": args.scalarization,
            "event_score_construction": "max_step(r/r0 + u/u0)",
        },
        "hybrid_calibration": {
            "beta1": beta1,
            "beta2": beta2,
            "auc_sB1": auc_b1,
            "auc_sB2": auc_b2,
            "objective": "maximize calibration event AUC from exact step maxima",
        },
        "targets": {
            "nominal_trust": args.nominal_trust,
            "moderate_target_trust": args.moderate_target_trust,
            "normalizer_quantile": args.normalizer_quantile,
        },
        "selection": {
            "ample_regime": args.ample_regime,
            "moderate_regime": args.moderate_regime,
            "nominal_ample_events": int(len(ample)),
            "moderate_events": int(len(moderate)),
            "all_events": int(len(events)),
            "all_steps": int(len(data)),
            "positive_events": int(labels.sum()),
            "negative_events": int((~labels).sum()),
        },
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in paths
        ],
    }
    output_path = Path(args.output)
    write_json_atomic(output_path, output)
    print(
        f"CALIBRATION_OK r0={r0:.9g} u0_lmax={u0_lmax:.9g} "
        f"u0_trace={u0_trace:.9g} T_th={T_th:.9g} "
        f"nominal_ample_far={realized_far:.9g} "
        f"beta1={beta1:.9g} beta2={beta2:.9g}",
        flush=True,
    )
    print(f"Wrote frozen calibration: {output_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
