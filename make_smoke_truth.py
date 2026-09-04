#!/usr/bin/env python3
"""Create a quarantined linear truth fixture for integration testing only.

The output proves the four-federate plumbing; it is not OpenDSS/Simscape truth
and must never be included in paper results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate smoke-only truth.npz")
    parser.add_argument("--feeder", default="feeder.npz")
    parser.add_argument("--output", default="truth.smoke.npz")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--steps-per-event", type=int, default=12)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--drift-amplitude", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 2 or args.steps_per_event < 1 or args.dt <= 0.0:
        raise ValueError("steps and timing arguments must be positive")
    feeder_path = Path(args.feeder)
    if not feeder_path.is_file():
        raise FileNotFoundError(feeder_path)
    feeder = np.load(feeder_path, allow_pickle=False)
    H = np.asarray(feeder["H"], dtype=float)
    Q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])
    if H.ndim != 2 or Q.shape != (H.shape[1], H.shape[1]):
        raise ValueError("invalid feeder H/Q dimensions")

    eigenvalues, eigenvectors = np.linalg.eigh((Q + Q.T) * 0.5)
    root_Q = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(args.seed)
    increments = rng.standard_normal((args.steps, H.shape[1])) @ root_Q.T
    x_true = np.cumsum(increments, axis=0)

    weighted_H = H / np.sqrt(np.asarray(feeder["sigma2"], dtype=float))[:, None]
    weakest = np.linalg.eigh(weighted_H.T @ weighted_H)[1][:, 0]
    start = args.steps // 2
    ramp = np.linspace(0.0, args.drift_amplitude, args.steps - start)
    x_true[start:] += ramp[:, None] * weakest[None, :]
    z_true = x_true @ H[:n_telemetry].T

    event_id = np.arange(args.steps, dtype=np.int64) // args.steps_per_event
    is_nominal = np.arange(args.steps) < start
    drift_family = np.where(is_nominal, "nominal", "smoke_weak_axis_ramp")
    trajectory_id = np.full(args.steps, "SMOKE_ONLY_NOT_FOR_RESULTS")
    W = np.eye(H.shape[1], dtype=float)
    time_axis = (np.arange(args.steps, dtype=float) + 1.0) * args.dt

    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        x_true=x_true,
        z_true=z_true,
        time=time_axis,
        event_id=event_id,
        is_nominal=is_nominal,
        drift_family=drift_family,
        trajectory_id=trajectory_id,
        W=W,
        meta=np.array(
            "SMOKE ONLY: linear H/Q fixture; never use for scientific results"
        ),
    )
    print(
        f"SMOKE_TRUTH_OK path={output_path.resolve()} steps={args.steps} "
        f"states={H.shape[1]} telemetry={n_telemetry}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
