#!/usr/bin/env python3
"""Production HELICS oracle/label federate.

The oracle receives ``twin.score.v1`` messages, compares each published state
estimate with the independently generated truth trajectory, and creates the
only admissible label in the experiment::

    d(t) = sqrt((x_true - x_hat)^T W (x_true - x_hat))
    label(t) = d(t) > gamma

No detector score, residual, alarm, or baseline enters the label.  Step and
event Parquet files are written for downstream ROC, F1, F2, and F3 analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORE_SCHEMA = "twin.score.v1"


class ProtocolError(ValueError):
    """A score message violates the declared wire contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_sha(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def finite_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return array


def load_truth(
    path: Path,
    *,
    n_state: int,
    dt: float,
    steps_per_event: int,
) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=False)
    x_true = finite_array(source["x_true"], "truth.x_true", 2)
    if x_true.shape[1] != n_state:
        raise ValueError(
            f"truth.x_true has {x_true.shape[1]} states; feeder requires {n_state}"
        )
    steps = x_true.shape[0]
    expected_time = (np.arange(steps, dtype=float) + 1.0) * dt
    time_axis = (
        finite_array(source["time"], "truth.time", 1)
        if "time" in source.files
        else expected_time
    )
    if time_axis.shape != (steps,) or not np.allclose(
        time_axis, expected_time, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("truth.time must equal dt, 2*dt, ...")

    expected_event = np.arange(steps, dtype=np.int64) // steps_per_event
    event_id = (
        np.asarray(source["event_id"], dtype=np.int64)
        if "event_id" in source.files
        else expected_event
    )
    if event_id.shape != (steps,) or not np.array_equal(event_id, expected_event):
        raise ValueError("truth.event_id must equal step_index // steps_per_event")

    result: dict[str, np.ndarray] = {
        "x_true": x_true,
        "time": time_axis,
        "event_id": event_id,
    }
    for key in ("drift_family", "is_nominal", "trajectory_id"):
        if key in source.files:
            array = np.asarray(source[key])
            if array.shape not in {(steps,), (int(event_id.max()) + 1,)}:
                raise ValueError(
                    f"truth.{key} must contain one value per step or per event"
                )
            result[key] = array
    if "W" in source.files:
        result["W"] = finite_array(source["W"], "truth.W", 2)
    return result


def load_weight(path: Path | None, truth: dict[str, np.ndarray], n_state: int) -> np.ndarray:
    if path is not None:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "W" not in loaded.files:
                raise ValueError("weight NPZ must contain W")
            weight = finite_array(loaded["W"], "weight.W", 2)
        else:
            weight = finite_array(loaded, "weight", 2)
    elif "W" in truth:
        weight = truth["W"]
    else:
        weight = np.eye(n_state, dtype=float)
    if weight.shape != (n_state, n_state):
        raise ValueError(f"W must have shape ({n_state},{n_state})")
    if not np.allclose(weight, weight.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("W must be symmetric")
    eigenvalues = np.linalg.eigvalsh(weight)
    if eigenvalues[0] < -1.0e-10 * max(1.0, abs(eigenvalues[-1])):
        raise ValueError("W must be positive semidefinite")
    return weight


def metadata_value(truth: dict[str, np.ndarray], key: str, step: int, event: int) -> Any:
    if key not in truth:
        return None
    values = truth[key]
    value = values[step] if values.shape[0] == truth["x_true"].shape[0] else values[event]
    return value.item() if hasattr(value, "item") else value


def parse_score(payload: str | bytes, *, n_state: int) -> dict[str, Any]:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("score payload is not UTF-8") from exc
    else:
        text = str(payload)
    try:
        item = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"score payload is not valid JSON: {exc}") from exc
    if not isinstance(item, dict) or item.get("schema") != SCORE_SCHEMA:
        raise ProtocolError(f"score schema must be {SCORE_SCHEMA!r}")
    for key in ("step_index", "event_id", "time", "s", "r", "chi2"):
        if key not in item:
            raise ProtocolError(f"score payload is missing {key}")
    try:
        step_index = int(item["step_index"])
        event_id = int(item["event_id"])
        timestamp = float(item["time"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError("step_index, event_id, and time have invalid types") from exc
    if step_index < 0 or event_id < 0 or not math.isfinite(timestamp):
        raise ProtocolError("step_index, event_id, and time are out of range")
    for key in ("s", "r", "chi2"):
        try:
            number = float(item[key])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"score field {key} must be numeric") from exc
        if not math.isfinite(number):
            raise ProtocolError(f"score field {key} must be finite")
        item[key] = number
    x_hat = np.asarray(item.get("x_hat"), dtype=float)
    if x_hat.shape != (n_state,) or not np.all(np.isfinite(x_hat)):
        raise ProtocolError(
            "x_hat is required and must be a finite vector matching feeder states"
        )
    item["step_index"] = step_index
    item["event_id"] = event_id
    item["time"] = timestamp
    item["x_hat"] = x_hat
    return item


def message_text(helics: Any, message: Any) -> str:
    if hasattr(helics, "helicsMessageIsValid") and not bool(
        helics.helicsMessageIsValid(message)
    ):
        raise ProtocolError("received an invalid HELICS score message")
    if hasattr(helics, "helicsMessageGetBytes"):
        data = bytes(helics.helicsMessageGetBytes(message))
    else:
        value = helics.helicsMessageGetString(message)
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    if not data:
        raise ProtocolError("received an empty HELICS score payload")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("HELICS score payload is not UTF-8") from exc


def free_message(helics: Any, message: Any) -> None:
    if hasattr(helics, "helicsMessageFree"):
        helics.helicsMessageFree(message)


def first_attribute(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"none of {names!r} exists in the HELICS Python module")


def disconnect_federate(helics: Any, federate: Any) -> None:
    if hasattr(helics, "helicsFederateDisconnect"):
        helics.helicsFederateDisconnect(federate)
    else:
        helics.helicsFederateFinalize(federate)


def build_federate(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    try:
        import helics as h
    except ImportError as exc:
        raise RuntimeError("HELICS Python bindings are not installed") from exc
    info = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreTypeFromString(info, args.core_type)
    core_init = args.core_init.strip() or (
        f"--federates=1 --broker_address={args.broker}"
    )
    h.helicsFederateInfoSetCoreInitString(info, core_init)
    if hasattr(h, "helicsFederateInfoSetBroker"):
        h.helicsFederateInfoSetBroker(info, args.broker)
    time_delta = first_attribute(
        h, "HELICS_PROPERTY_TIME_DELTA", "helics_property_time_delta"
    )
    h.helicsFederateInfoSetTimeProperty(info, time_delta, args.helics_time_delta)
    uninterruptible = first_attribute(
        h,
        "HELICS_FLAG_UNINTERRUPTIBLE",
        "helics_flag_uninterruptible",
    )
    h.helicsFederateInfoSetFlagOption(info, uninterruptible, True)
    federate = h.helicsCreateMessageFederate(args.federate_name, info)
    h.helicsFederateAddDependency(federate, args.upstream_federate)
    endpoint = h.helicsFederateRegisterGlobalEndpoint(
        federate, args.input_endpoint, "json"
    )
    registered_name = h.helicsEndpointGetName(endpoint)
    if registered_name != args.input_endpoint:
        raise RuntimeError(
            f"HELICS registered input endpoint {registered_name!r}; "
            f"expected {args.input_endpoint!r}"
        )
    return h, info, federate, endpoint


def aggregate_events(frame: pd.DataFrame) -> pd.DataFrame:
    identity = {
        "event_id", "pattern_id", "arm", "regime", "stratum", "drift_family",
        "trajectory_id",
    }
    rows: list[dict[str, Any]] = []
    for event_id, block in frame.groupby("event_id", sort=True):
        first = block.iloc[0]
        row: dict[str, Any] = {
            key: first.get(key) for key in identity if key in block.columns
        }
        row["event_id"] = int(event_id)
        row["label"] = bool(block["label"].any())
        row["d"] = float(block["d"].max())
        row["steps"] = int(len(block))
        row["is_nominal"] = bool(block["is_nominal"].all())
        row["nominal_fraction"] = float(block["is_nominal"].mean())
        for column in block.columns:
            if column in identity or column in {
                "time", "step_index", "label", "d", "is_nominal", "x_error_norm"
            }:
                continue
            values = pd.to_numeric(block[column], errors="coerce").to_numpy(float)
            if np.any(np.isfinite(values)):
                row[column] = float(np.nanmax(values))
        rows.append(row)
    return pd.DataFrame(rows)


def score_collection_deadline(
    expected_step: int,
    *,
    dt: float,
    helics_time_delta: float,
    lag_steps: int,
) -> float:
    """Return the HELICS time by which a twin score must be observable.

    The oracle is staged three HELICS microsteps after the physical logical
    time, after power, network, and twin execution. This preserves the score's
    original logical timestamp without relying on ambiguous same-time delivery.
    A nonzero lag remains available only as a deliberate collection grace
    period. The score payload always retains its original logical time and step
    index; this deadline controls only how long the oracle remains connected.
    """
    if (
        expected_step < 0
        or dt <= 0.0
        or helics_time_delta <= 0.0
        or lag_steps < 0
    ):
        raise ValueError("invalid score collection timing")
    return (
        (expected_step + 1 + lag_steps) * dt
        + 3.0 * helics_time_delta
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production HELICS oracle federate")
    parser.add_argument("--feeder", default="feeder.npz")
    parser.add_argument("--truth", default="truth.npz")
    parser.add_argument("--weight")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--out-dir", default="runs/oracle")
    parser.add_argument("--federate-name", default="oracle_fed")
    parser.add_argument("--input-endpoint", default="oracle_fed/in")
    parser.add_argument("--upstream-federate", default="twin_fed")
    parser.add_argument(
        "--broker",
        default=os.environ.get("HELICS_BROKER_ADDRESS", "tcp://broker:23404"),
    )
    parser.add_argument("--core-type", default="zmq")
    parser.add_argument("--core-init", default="")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--helics-time-delta", type=float, default=1.0e-6)
    parser.add_argument("--steps-per-event", type=int, default=12)
    parser.add_argument(
        "--score-lag-steps",
        type=int,
        default=0,
        help=(
            "additional logical steps allowed to collect each twin score; "
            "0 uses the explicit twin_fed time dependency for same-time ordering"
        ),
    )
    parser.add_argument("--stop-time", type=float, default=0.0)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gamma <= 0.0 or not math.isfinite(args.gamma):
        raise ValueError("--gamma must be a finite positive, predeclared threshold")
    if (
        args.dt <= 0.0
        or args.helics_time_delta <= 0.0
        or args.steps_per_event < 1
        or args.score_lag_steps < 0
        or not args.upstream_federate
    ):
        raise ValueError(
            "dt/time delta/steps must be positive, score lag nonnegative, "
            "and upstream federate nonempty"
        )
    feeder_path, truth_path = Path(args.feeder), Path(args.truth)
    weight_path = Path(args.weight) if args.weight else None
    for path in (feeder_path, truth_path, weight_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    feeder = np.load(feeder_path, allow_pickle=False)
    H = finite_array(feeder["H"], "feeder.H", 2)
    truth = load_truth(
        truth_path, n_state=H.shape[1], dt=args.dt, steps_per_event=args.steps_per_event
    )
    W = load_weight(weight_path, truth, H.shape[1])
    available = truth["x_true"].shape[0]
    if args.stop_time <= 0.0:
        steps = available
    else:
        raw = args.stop_time / args.dt
        steps = int(round(raw))
        if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1.0e-9) or not 1 <= steps <= available:
            raise ValueError("--stop-time must select an integer number of available steps")
    print(
        f"oracle_fed preflight: steps={steps} states={H.shape[1]} gamma={args.gamma:g} "
        f"weight={'file/truth' if weight_path or 'W' in truth else 'identity'} "
        f"score_lag={args.score_lag_steps} "
        f"stage_offset={3.0 * args.helics_time_delta:g}s",
        flush=True,
    )
    if args.validate_only:
        print("PRODUCTION_ORACLE_VALIDATE_OK", flush=True)
        return 0

    output_directory = Path(args.out_dir)
    score_path = output_directory / "oracle_scores.parquet"
    event_path = output_directory / "oracle_events.parquet"
    meta_path = output_directory / "meta.json"
    meta: dict[str, Any] = {
        "schema": "oracle.run.meta.v1",
        "status": "starting",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name) for name in ("helics", "numpy", "pandas", "pyarrow")
        },
        "git_sha": git_sha(Path.cwd()),
        "inputs": {
            "feeder": {"path": str(feeder_path.resolve()), "sha256": sha256_file(feeder_path)},
            "truth": {"path": str(truth_path.resolve()), "sha256": sha256_file(truth_path)},
        },
        "label_definition": {
            "formula": "sqrt((x_true-x_hat)^T W (x_true-x_hat)) > gamma",
            "gamma": args.gamma,
            "uses_detector_or_baseline": False,
            "weight_source": str(weight_path.resolve()) if weight_path else (
                "truth.npz:W" if "W" in truth else "identity"
            ),
        },
        "resolved_arguments": vars(args),
        "timing": {
            "logical_dt": args.dt,
            "helics_time_delta": args.helics_time_delta,
            "stage_offset": 3.0 * args.helics_time_delta,
        },
    }
    if weight_path:
        meta["inputs"]["weight"] = {
            "path": str(weight_path.resolve()), "sha256": sha256_file(weight_path)
        }
    write_json_atomic(meta_path, meta)

    helics = info = federate = None
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    malformed = duplicates = 0
    completed = False
    wall_start = time.time()
    try:
        helics, info, federate, endpoint = build_federate(args)
        helics.helicsFederateEnterExecutingMode(federate)
        print(
            f"HELICS connected: {args.federate_name} "
            f"input={args.input_endpoint} upstream={args.upstream_federate}",
            flush=True,
        )
        meta["status"] = "running"
        write_json_atomic(meta_path, meta)
        for expected_step in range(steps):
            requested = score_collection_deadline(
                expected_step,
                dt=args.dt,
                helics_time_delta=args.helics_time_delta,
                lag_steps=args.score_lag_steps,
            )
            while True:
                granted = float(helics.helicsFederateRequestTime(federate, requested))
                while bool(helics.helicsEndpointHasMessage(endpoint)):
                    message = helics.helicsEndpointGetMessage(endpoint)
                    if message is None:
                        break
                    try:
                        item = parse_score(message_text(helics, message), n_state=H.shape[1])
                        step_index = item["step_index"]
                        if step_index in seen:
                            duplicates += 1
                            raise ProtocolError(f"duplicate score for step {step_index}")
                        if not 0 <= step_index < steps:
                            raise ProtocolError(f"score step {step_index} is outside this run")
                        event_id = item["event_id"]
                        if event_id != int(truth["event_id"][step_index]):
                            raise ProtocolError("score/truth event_id mismatch")
                        expected_time = float(truth["time"][step_index])
                        if not math.isclose(item["time"], expected_time, rel_tol=0.0, abs_tol=1.0e-9):
                            raise ProtocolError("score/truth time mismatch")

                        error = truth["x_true"][step_index] - item.pop("x_hat")
                        quadratic = float(error @ W @ error)
                        d_value = math.sqrt(max(quadratic, 0.0))
                        row = {key: json_safe(value) for key, value in item.items() if key != "schema"}
                        row.update(
                            {
                                "d": d_value,
                                "x_error_norm": float(np.linalg.norm(error)),
                                "label": bool(d_value > args.gamma),
                                "drift_family": metadata_value(
                                    truth, "drift_family", step_index, event_id
                                ),
                                "trajectory_id": metadata_value(
                                    truth, "trajectory_id", step_index, event_id
                                ),
                            }
                        )
                        nominal = metadata_value(truth, "is_nominal", step_index, event_id)
                        row["is_nominal"] = bool(nominal) if nominal is not None else not row["label"]
                        rows.append(row)
                        seen.add(step_index)
                    except ProtocolError as exc:
                        malformed += 1
                        if args.strict_protocol:
                            raise
                        print(f"[WARN] rejected score at t={granted:g}: {exc}", flush=True)
                    finally:
                        free_message(helics, message)
                if granted + 1.0e-9 >= requested:
                    break
            if expected_step not in seen:
                raise RuntimeError(f"missing twin score for step {expected_step}")
        completed = True
    finally:
        if federate is not None and helics is not None:
            try:
                disconnect_federate(helics, federate)
            except Exception as exc:
                print(f"[WARN] HELICS disconnect failed: {exc}", file=sys.stderr)
            try:
                helics.helicsFederateFree(federate)
            except Exception:
                pass
        if info is not None and helics is not None:
            try:
                helics.helicsFederateInfoFree(info)
            except Exception:
                pass
        if helics is not None:
            try:
                helics.helicsCloseLibrary()
            except Exception:
                pass
        if rows:
            scores = pd.DataFrame(rows).sort_values("step_index").reset_index(drop=True)
            write_parquet_atomic(score_path, scores)
            write_parquet_atomic(event_path, aggregate_events(scores))
        meta["status"] = "complete" if completed else "failed_or_interrupted"
        meta["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["wall_seconds"] = time.time() - wall_start
        meta["runtime_counts"] = {
            "scores": len(rows), "events": len(set(row["event_id"] for row in rows)),
            "duplicates": duplicates, "malformed": malformed,
        }
        write_json_atomic(meta_path, meta)

    print(
        f"oracle_fed complete: scores={len(rows)} events={len(set(row['event_id'] for row in rows))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ProtocolError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
