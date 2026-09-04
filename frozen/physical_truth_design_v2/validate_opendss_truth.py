#!/usr/bin/env python3
"""Strict contract validator for ``opendss.truth.v2`` NPZ artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--feeder", type=Path, required=True)
    parser.add_argument("--expected-role")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-events", type=int, default=1100)
    parser.add_argument("--expected-steps-per-event", type=int, default=12)
    parser.add_argument("--disjoint-with", type=Path)
    return parser.parse_args()


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar, got shape {array.shape}")
    return array.item()


def main() -> int:
    args = parse_args()
    truth_path = args.truth.resolve(strict=True)
    feeder_path = args.feeder.resolve(strict=True)
    with np.load(feeder_path, allow_pickle=False) as feeder:
        frozen = {
            key: np.asarray(feeder[key])
            for key in ("node_order", "node2sn", "theta_idx", "slack_supernodes")
        }
        h_telemetry = np.asarray(feeder["H_telemetry"], dtype=np.float64)

    with np.load(truth_path, allow_pickle=False) as truth:
        required = {
            "x_true",
            "z_true",
            "z_physical",
            "time",
            "event_id",
            "drift_family",
            "is_nominal",
            "trajectory_id",
            "event_mechanism",
            "physical_seed",
            "W",
            "node_order",
            "node2sn",
            "theta_idx",
            "slack_supernodes",
            "telemetry_names",
            "telemetry_offset",
            "x0",
            "meta",
        }
        missing = sorted(required - set(truth.files))
        if missing:
            raise ValueError(f"missing required arrays: {missing}")
        meta = json.loads(str(scalar(truth["meta"])))
        if meta.get("schema") != "opendss.truth.v2":
            raise ValueError(f"wrong schema: {meta.get('schema')!r}")
        events = args.expected_events
        steps_per_event = args.expected_steps_per_event
        steps = events * steps_per_event
        x = np.asarray(truth["x_true"])
        z = np.asarray(truth["z_true"])
        raw = np.asarray(truth["z_physical"])
        if x.shape != (steps, 491):
            raise ValueError(f"x_true shape is {x.shape}, expected {(steps, 491)}")
        if z.shape != (steps, 45) or raw.shape != (steps, 45):
            raise ValueError(f"telemetry shapes are z={z.shape}, raw={raw.shape}")
        if not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isfinite(raw).all():
            raise ValueError("truth contains non-finite states or measurements")
        expected_time = np.arange(1, steps + 1)
        expected_event = np.arange(steps) // steps_per_event
        if not np.array_equal(truth["time"], expected_time):
            raise ValueError("time must be exactly 1..steps")
        if not np.array_equal(truth["event_id"], expected_event):
            raise ValueError("event_id does not match the steps-per-event contract")
        for key, expected in frozen.items():
            if not np.array_equal(truth[key], expected):
                raise ValueError(f"truth {key} differs from feeder.npz")
        if truth["drift_family"].shape != (events,):
            raise ValueError("drift_family must be event-level")
        if truth["is_nominal"].shape != (events,):
            raise ValueError("is_nominal must be event-level")
        if truth["trajectory_id"].shape != (events,):
            raise ValueError("trajectory_id must be event-level")
        if len(set(truth["trajectory_id"].astype(str).tolist())) != events:
            raise ValueError("trajectory_id values are not unique")
        nominal = truth["drift_family"].astype(str) == "nominal"
        if not np.array_equal(truth["is_nominal"].astype(bool), nominal):
            raise ValueError("is_nominal is inconsistent with drift_family")
        families = set(truth["drift_family"].astype(str).tolist())
        allowed = {"nominal", "load_ramp", "parameter_change", "topology_change"}
        if not families <= allowed:
            raise ValueError(f"unexpected drift families: {sorted(families - allowed)}")
        offset = np.asarray(truth["telemetry_offset"], dtype=np.float64)
        affine_error = float(np.max(np.abs(z.astype(float) - raw.astype(float) - offset)))
        if affine_error > 2e-6:
            raise ValueError(f"z_true affine-coordinate error is {affine_error:g}")
        x0 = np.asarray(truth["x0"], dtype=np.float64)
        if x0.shape != (491,) or offset.shape != (45,):
            raise ValueError("x0 or telemetry_offset has an invalid shape")
        if h_telemetry.shape != (45, 491):
            raise ValueError("frozen H_telemetry has an invalid shape")
        weight = np.asarray(truth["W"], dtype=np.float64)
        if weight.shape != (491, 491):
            raise ValueError("W has an invalid shape")
        if np.max(np.abs(weight - weight.T)) > 1e-8:
            raise ValueError("W is not symmetric")
        if float(np.linalg.eigvalsh(weight).min()) < -1e-8:
            raise ValueError("W is not positive semidefinite")
        seed = int(scalar(truth["physical_seed"]))
        if args.expected_seed is not None and seed != args.expected_seed:
            raise ValueError(f"physical_seed is {seed}, expected {args.expected_seed}")
        if args.expected_role is not None and meta.get("role") != args.expected_role:
            raise ValueError(f"role is {meta.get('role')!r}, expected {args.expected_role!r}")
        if int(meta.get("events", -1)) != events or int(meta.get("steps", -1)) != steps:
            raise ValueError("metadata event/step counts are inconsistent")
        trajectories = set(truth["trajectory_id"].astype(str).tolist())

    if args.disjoint_with is not None:
        with np.load(args.disjoint_with.resolve(strict=True), allow_pickle=False) as other:
            other_ids = set(other["trajectory_id"].astype(str).tolist())
        overlap = trajectories & other_ids
        if overlap:
            raise ValueError(f"trajectory split overlap: {sorted(overlap)[:10]}")

    print(
        "OPENDSS_TRUTH_VALIDATE_OK",
        f"path={truth_path}",
        f"steps={steps}",
        f"events={events}",
        "states=491",
        "telemetry=45",
        f"affine_max_error={affine_error:.3g}",
        f"families={','.join(sorted(families))}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
