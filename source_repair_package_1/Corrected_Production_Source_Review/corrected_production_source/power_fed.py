#!/usr/bin/env python3
"""Production HELICS power/measurement playback federate.

The real power solver (OpenDSS, pandapower, or Simscape) must first export the
truth contract documented in ``truth_contract.md``.  This federate never
creates labels and never reads communication patterns.  It publishes one
timestamped telemetry packet per physical channel and adds only the frozen
measurement-noise realization associated with ``--seed``.

Wire format sent to ``net_fed/in``::

    <channel_id><TAB>{"schema":"twin.telemetry.v1", ...}

The JSON repeats channel_id and includes step_index/event_id so the network
federate selects the impairment row from the packet, not from arrival order.
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


TELEMETRY_SCHEMA = "twin.telemetry.v1"


def finite_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return array


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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


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


def load_inputs(
    feeder_path: Path,
    truth_path: Path,
    *,
    dt: float,
    allow_linearized_telemetry: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    str,
]:
    feeder = np.load(feeder_path, allow_pickle=False)
    H = finite_array(feeder["H"], "feeder.H", 2)
    sigma2 = finite_array(feeder["sigma2"], "feeder.sigma2", 1)
    n_telemetry = int(feeder["n_telemetry"])
    if sigma2.shape != (H.shape[0],) or not 1 <= n_telemetry <= H.shape[0]:
        raise ValueError("feeder dimensions or n_telemetry are inconsistent")
    if np.any(sigma2 <= 0.0):
        raise ValueError("feeder.sigma2 must be positive")

    truth = np.load(truth_path, allow_pickle=False)
    x_true = finite_array(truth["x_true"], "truth.x_true", 2)
    if x_true.shape[1] != H.shape[1]:
        raise ValueError(
            f"truth.x_true has {x_true.shape[1]} states; feeder requires {H.shape[1]}"
        )
    z_true = None
    if "z_true" in truth.files:
        z_true = finite_array(truth["z_true"], "truth.z_true", 2)
        if z_true.shape != (x_true.shape[0], n_telemetry):
            raise ValueError(
                "truth.z_true must have shape (steps, feeder.n_telemetry)"
            )
    if z_true is None and not allow_linearized_telemetry:
        raise ValueError(
            "production truth is missing z_true. Export physical telemetry from "
            "the power-system solver for every step. The fallback H_telemetry @ "
            "x_true is permitted only for quarantined smoke/integration fixtures "
            "with --allow-linearized-telemetry and must never be used for paper results"
        )
    if "time" in truth.files:
        timestamps = finite_array(truth["time"], "truth.time", 1)
    else:
        timestamps = (np.arange(x_true.shape[0], dtype=float) + 1.0) * dt
    if timestamps.shape != (x_true.shape[0],):
        raise ValueError("truth.time must have one value per state row")
    expected = (np.arange(x_true.shape[0], dtype=float) + 1.0) * dt
    if not np.allclose(timestamps, expected, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "truth.time must equal dt, 2*dt, ...; resample the trajectory before playback"
        )
    telemetry_source = "truth.z_true" if z_true is not None else "linearized_smoke_fallback"
    return H, sigma2, n_telemetry, x_true, z_true, timestamps, telemetry_source


def resolved_steps(stop_time: float, dt: float, available: int) -> int:
    if stop_time <= 0.0:
        return available
    raw = stop_time / dt
    steps = int(round(raw))
    if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("--stop-time must be an integer multiple of --dt")
    if not 1 <= steps <= available:
        raise ValueError(f"--stop-time selects {steps} steps; available range is 1..{available}")
    return steps


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
    federate = h.helicsCreateMessageFederate(args.federate_name, info)
    endpoint = h.helicsFederateRegisterGlobalEndpoint(
        federate, args.output_endpoint, "json"
    )
    registered_name = h.helicsEndpointGetName(endpoint)
    if registered_name != args.output_endpoint:
        raise RuntimeError(
            f"HELICS registered output endpoint {registered_name!r}; "
            f"expected {args.output_endpoint!r}"
        )
    h.helicsEndpointSetDefaultDestination(endpoint, args.destination)
    return h, info, federate, endpoint


def send_packet(
    helics: Any,
    endpoint: Any,
    destination: str,
    channel_id: int,
    payload: dict[str, Any],
) -> int:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    envelope = f"{channel_id}\t{encoded}"
    if hasattr(helics, "helicsEndpointSendStringTo"):
        helics.helicsEndpointSendStringTo(endpoint, envelope, destination)
    else:
        helics.helicsEndpointSendBytesTo(
            endpoint, envelope.encode("utf-8"), destination
        )
    return len(envelope.encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production HELICS power/measurement playback federate"
    )
    parser.add_argument("--feeder", default="feeder.npz")
    parser.add_argument("--truth", default="truth.npz")
    parser.add_argument("--out-dir", default="runs/power")
    parser.add_argument("--federate-name", default="power_fed")
    parser.add_argument("--output-endpoint", default="power_fed/out")
    parser.add_argument("--destination", default="net_fed/in")
    parser.add_argument(
        "--broker",
        default=os.environ.get("HELICS_BROKER_ADDRESS", "tcp://broker:23404"),
    )
    parser.add_argument("--core-type", default="zmq")
    parser.add_argument("--core-init", default="")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--helics-time-delta", type=float, default=1.0e-6)
    parser.add_argument("--steps-per-event", type=int, default=12)
    parser.add_argument("--stop-time", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--measurement-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-linearized-telemetry",
        action="store_true",
        help=(
            "allow H_telemetry @ x_true only for quarantined smoke/integration "
            "fixtures; forbidden for scientific production runs"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dt <= 0.0 or args.helics_time_delta <= 0.0:
        raise ValueError("--dt and --helics-time-delta must be positive")
    if args.steps_per_event < 1 or args.seed < 0:
        raise ValueError("--steps-per-event must be positive and --seed nonnegative")

    feeder_path = Path(args.feeder)
    truth_path = Path(args.truth)
    for path in (feeder_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    H, sigma2, n_telemetry, x_true, z_true, timestamps, telemetry_source = load_inputs(
        feeder_path,
        truth_path,
        dt=args.dt,
        allow_linearized_telemetry=args.allow_linearized_telemetry,
    )
    steps = resolved_steps(args.stop_time, args.dt, x_true.shape[0])
    print(
        f"power_fed preflight: steps={steps} states={H.shape[1]} "
        f"telemetry={n_telemetry} telemetry_source={telemetry_source} "
        f"noise={args.measurement_noise} seed={args.seed}",
        flush=True,
    )
    if args.validate_only:
        print("PRODUCTION_POWER_VALIDATE_OK", flush=True)
        return 0

    output_directory = Path(args.out_dir)
    meta_path = output_directory / "meta.json"
    meta: dict[str, Any] = {
        "schema": "power.run.meta.v1",
        "status": "starting",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: package_version(name) for name in ("helics", "numpy")},
        "git_sha": git_sha(Path.cwd()),
        "inputs": {
            "feeder": {"path": str(feeder_path.resolve()), "sha256": sha256_file(feeder_path)},
            "truth": {"path": str(truth_path.resolve()), "sha256": sha256_file(truth_path)},
        },
        "resolved_arguments": vars(args),
        "dimensions": {
            "states": int(H.shape[1]),
            "telemetry": n_telemetry,
            "steps": steps,
        },
        "telemetry_source": telemetry_source,
        "seed_stream": {
            "role": "physical_measurement_noise",
            "seed": args.seed,
            "independent_from_network_seed": True,
        },
    }
    write_json_atomic(meta_path, meta)

    rng = np.random.default_rng(args.seed)
    helics = info = federate = None
    packets = bytes_sent = 0
    completed = False
    wall_start = time.time()
    try:
        helics, info, federate, endpoint = build_federate(args)
        helics.helicsFederateEnterExecutingMode(federate)
        meta["status"] = "running"
        write_json_atomic(meta_path, meta)
        for step_index in range(steps):
            requested = float(timestamps[step_index])
            granted = float(helics.helicsFederateRequestTime(federate, requested))
            if not math.isclose(granted, requested, rel_tol=0.0, abs_tol=1.0e-9):
                raise RuntimeError(f"power_fed requested {requested} but received {granted}")

            if z_true is None:
                values = H[:n_telemetry] @ x_true[step_index]
            else:
                values = z_true[step_index].copy()
            if args.measurement_noise:
                values = values + rng.normal(0.0, np.sqrt(sigma2[:n_telemetry]))
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"nonfinite telemetry at step {step_index}")

            event_id = step_index // args.steps_per_event
            for channel_id, value in enumerate(values):
                payload = {
                    "schema": TELEMETRY_SCHEMA,
                    "channel_id": channel_id,
                    "value": float(value),
                    "source_time": requested,
                    "sequence": step_index,
                    "step_index": step_index,
                    "event_id": event_id,
                }
                bytes_sent += send_packet(
                    helics, endpoint, args.destination, channel_id, payload
                )
                packets += 1
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
        meta["status"] = "complete" if completed else "failed_or_interrupted"
        meta["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["wall_seconds"] = time.time() - wall_start
        meta["runtime_counts"] = {"packets_sent": packets, "bytes_sent": bytes_sent}
        write_json_atomic(meta_path, meta)

    print(f"power_fed complete: packets={packets} bytes={bytes_sent}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
