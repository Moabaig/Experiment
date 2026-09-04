#!/usr/bin/env python3
"""Export deterministic, nonlinear OpenDSS truth for the DT experiment.

The exported state has the exact frozen feeder coordinates

    x = [theta(non-slack supernodes), voltage_magnitude(all supernodes)].

The 45-channel ``z_true`` array is measured from every solved OpenDSS state,
then expressed in the affine coordinates used by the frozen estimator:

    z_true(x) = h_OpenDSS(x) + H_telemetry x0 - h_OpenDSS(x0).

It is intentionally not computed as ``H_telemetry @ x``.  ``z_physical`` is
also saved so the raw nonlinear OpenDSS measurements remain auditable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import opendssdirect as dss


SCHEMA = "opendss.truth.v2"
SBASE_VA = 1_000_000.0
EXPECTED_STATES = 491
EXPECTED_TELEMETRY = 45
EXPECTED_MASTER_SHA256 = (
    "c92a69d9b218b1b2646ec7911783826229309038e72f16b848304c0457c0a54d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def compile_and_solve(master: Path) -> None:
    dss.Basic.ClearAll()
    dss.Text.Command(f'Redirect "{master}"')
    dss.Text.Command("Set MaxIterations=100")
    dss.Text.Command("Set MaxControlIter=100")
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError(f"OpenDSS base power flow did not converge: {master}")


def complex_vector(values: list[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size % 2:
        raise ValueError(f"invalid OpenDSS complex vector of length {raw.size}")
    return raw[0::2] + 1j * raw[1::2]


def voltage_bases(order: list[str]) -> np.ndarray:
    by_node: dict[str, float] = {}
    for bus in dss.Circuit.AllBusNames():
        # OpenDSS uses a zero-based bus index, so the first valid bus returns 0.
        dss.Circuit.SetActiveBus(bus)
        base = float(dss.Bus.kVBase()) * 1000.0
        if not math.isfinite(base) or base <= 0:
            raise RuntimeError(f"invalid voltage base at bus {bus!r}: {base}")
        for node in dss.Bus.Nodes():
            by_node[f"{bus}.{node}".upper()] = base
    missing = [node for node in order if node.upper() not in by_node]
    if missing:
        raise RuntimeError(f"missing voltage bases for nodes: {missing[:10]}")
    return np.asarray([by_node[node.upper()] for node in order], dtype=np.float64)


def system_y_pu(vbase: np.ndarray) -> np.ndarray:
    n = len(vbase)
    y = complex_vector(dss.Circuit.SystemY()).reshape(n, n)
    return y * (vbase[:, None] * vbase[None, :]) / SBASE_VA


def solved_vpu(vbase: np.ndarray) -> np.ndarray:
    return complex_vector(dss.Circuit.YNodeVArray()) / vbase


def state_from_vpu(
    vpu: np.ndarray, node2sn: np.ndarray, theta_idx: np.ndarray
) -> np.ndarray:
    supernodes = int(node2sn.max()) + 1
    representatives = np.full(supernodes, -1, dtype=np.int64)
    for node_index, supernode in enumerate(node2sn):
        if representatives[int(supernode)] < 0:
            representatives[int(supernode)] = node_index
    if (representatives < 0).any():
        raise RuntimeError("node2sn contains an empty supernode")
    angles = np.angle(vpu[representatives])
    magnitudes = np.abs(vpu[representatives])
    return np.concatenate((angles[theta_idx], magnitudes))


def telemetry_specs(order: list[str], slack_supernodes: np.ndarray, node2sn: np.ndarray):
    v_buses = {"150", "150R", "9R", "25R", "160R", "76", "300", "97", "67", "610"}
    pq_buses = {"76", "97", "300"}
    v_indices = [
        i for i, node in enumerate(order) if node.split(".")[0].upper() in v_buses
    ]
    slack = {int(x) for x in slack_supernodes}
    pq_indices = [
        i
        for i, node in enumerate(order)
        if node.split(".")[0].upper() in pq_buses and int(node2sn[i]) not in slack
    ]
    specs: list[tuple[str, int]] = [("Vmag", i) for i in v_indices]
    for index in pq_indices:
        specs.extend((("P", index), ("Q", index)))
    names = np.asarray([f"{kind}@{order[index]}" for kind, index in specs])
    return specs, names


def physical_telemetry(
    vpu: np.ndarray, ypu: np.ndarray, specs: list[tuple[str, int]]
) -> np.ndarray:
    injections = vpu * np.conj(ypu @ vpu)
    values = []
    for kind, index in specs:
        if kind == "Vmag":
            values.append(abs(vpu[index]))
        elif kind == "P":
            values.append(injections[index].real)
        elif kind == "Q":
            values.append(injections[index].imag)
        else:
            raise AssertionError(kind)
    return np.asarray(values, dtype=np.float64)


def analytic_telemetry_jacobian(
    vpu: np.ndarray,
    ypu: np.ndarray,
    node2sn: np.ndarray,
    theta_idx: np.ndarray,
    specs: list[tuple[str, int]],
) -> np.ndarray:
    """Rebuild the exact builder Jacobian as a mapping/order contract."""
    vm = np.abs(vpu)
    va = np.angle(vpu)
    g = ypu.real
    b = ypu.imag
    phase = va[:, None] - va[None, :]
    cosine = np.cos(phase)
    sine = np.sin(phase)
    vv = vm[:, None] * vm[None, :]
    p = (vv * (g * cosine + b * sine)).sum(axis=1)
    q = (vv * (g * sine - b * cosine)).sum(axis=1)

    dp_dt = vv * (g * sine - b * cosine)
    np.fill_diagonal(dp_dt, 0.0)
    np.fill_diagonal(dp_dt, -q - np.diag(b) * vm**2)
    dq_dt = -vv * (g * cosine + b * sine)
    np.fill_diagonal(dq_dt, 0.0)
    np.fill_diagonal(dq_dt, p - np.diag(g) * vm**2)
    dp_dv = vm[:, None] * (g * cosine + b * sine)
    np.fill_diagonal(dp_dv, 0.0)
    np.fill_diagonal(dp_dv, p / np.maximum(vm, 1e-9) + np.diag(g) * vm)
    dq_dv = vm[:, None] * (g * sine - b * cosine)
    np.fill_diagonal(dq_dv, 0.0)
    np.fill_diagonal(dq_dv, q / np.maximum(vm, 1e-9) - np.diag(b) * vm)

    supernodes = int(node2sn.max()) + 1
    nstate = len(theta_idx) + supernodes

    def merge(row: np.ndarray) -> np.ndarray:
        result = np.zeros(supernodes, dtype=np.float64)
        np.add.at(result, node2sn, row)
        return result

    rows = []
    for kind, index in specs:
        row = np.zeros(nstate, dtype=np.float64)
        if kind == "Vmag":
            row[len(theta_idx) + int(node2sn[index])] = 1.0
        else:
            derivative_theta, derivative_voltage = (
                (dp_dt, dp_dv) if kind == "P" else (dq_dt, dq_dv)
            )
            row[: len(theta_idx)] = merge(derivative_theta[index])[theta_idx]
            row[len(theta_idx) :] = merge(derivative_voltage[index])
        rows.append(row)
    return np.asarray(rows)


def collect_loads():
    names: list[str] = []
    buses: list[str] = []
    kw: list[float] = []
    kvar: list[float] = []
    if dss.Loads.First() == 0:
        raise RuntimeError("OpenDSS model has no loads")
    while True:
        names.append(str(dss.Loads.Name()))
        buses.append(str(dss.CktElement.BusNames()[0]).split(".")[0].upper())
        kw.append(float(dss.Loads.kW()))
        kvar.append(float(dss.Loads.kvar()))
        if dss.Loads.Next() == 0:
            break
    return names, buses, np.asarray(kw), np.asarray(kvar)


def feeder_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    excluded = {"line.sw7", "line.sw8"}
    if dss.PDElements.First() == 0:
        return graph
    while True:
        name = str(dss.CktElement.Name()).lower()
        buses = [str(x).split(".")[0].upper() for x in dss.CktElement.BusNames()]
        if name not in excluded and len(buses) >= 2 and buses[0] != buses[1]:
            graph.setdefault(buses[0], set()).add(buses[1])
            graph.setdefault(buses[1], set()).add(buses[0])
        if dss.PDElements.Next() == 0:
            break
    return graph


def load_groups(candidate_roots: list[str], load_buses: list[str]) -> dict[str, np.ndarray]:
    graph = feeder_graph()
    parent: dict[str, str | None] = {"150": None}
    queue: deque[str] = deque(["150"])
    while queue:
        bus = queue.popleft()
        for neighbor in sorted(graph.get(bus, ())):
            if neighbor not in parent:
                parent[neighbor] = bus
                queue.append(neighbor)
    children: dict[str, set[str]] = {}
    for node, ancestor in parent.items():
        if ancestor is not None:
            children.setdefault(ancestor, set()).add(node)

    groups: dict[str, np.ndarray] = {}
    for root in candidate_roots:
        key = root.upper()
        if key not in parent:
            continue
        descendants = {key}
        pending = [key]
        while pending:
            node = pending.pop()
            for child in children.get(node, ()):
                if child not in descendants:
                    descendants.add(child)
                    pending.append(child)
        mask = np.asarray([bus in descendants for bus in load_buses], dtype=bool)
        if int(mask.sum()) >= 2:
            groups[root] = mask
    if not groups:
        raise RuntimeError("none of the configured load-ramp roots contains two loads")
    return groups


def set_loads(names, base_kw, base_kvar, multipliers: np.ndarray) -> None:
    for name, kw, kvar, multiplier in zip(names, base_kw, base_kvar, multipliers):
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw * multiplier))
        dss.Loads.kvar(float(kvar * multiplier))


def add_tie(tie: dict[str, Any]) -> None:
    command = (
        f"New Line.{tie['name']} phases={int(tie['phases'])} "
        f"Bus1={tie['bus1']} Bus2={tie['bus2']} switch=yes "
        f"r1={float(tie['r1']):.12g} r0={float(tie['r0']):.12g} "
        f"x1={float(tie['x1']):.12g} x0={float(tie['x0']):.12g} "
        f"c1=0 c0=0 Length={float(tie['length']):.12g}"
    )
    dss.Text.Command(command)


def set_transformer_tap(name: str, value: float) -> None:
    dss.Transformers.Name(name)
    if str(dss.Transformers.Name()).lower() != name.lower():
        raise RuntimeError(f"unknown transformer {name!r}")
    dss.Transformers.Wdg(2)
    dss.Transformers.Tap(float(value))


def event_schedule(events: int, mix: dict[str, Any], seed: int) -> np.ndarray:
    families = ("nominal", "load_ramp", "parameter_change", "topology_change")
    probabilities = np.asarray([float(mix[name]) for name in families])
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("event_mix probabilities must be non-negative and sum to one")
    expected = probabilities * events
    counts = np.floor(expected).astype(int)
    for index in np.argsort(-(expected - counts))[: events - int(counts.sum())]:
        counts[index] += 1
    schedule = np.concatenate(
        [np.repeat(family, count) for family, count in zip(families, counts)]
    )
    np.random.default_rng(seed).shuffle(schedule)
    return schedule.astype("U32")


def load_weight(path: Path, nstate: int) -> np.ndarray:
    source = np.load(path, allow_pickle=False)
    try:
        if isinstance(source, np.lib.npyio.NpzFile):
            if "W" not in source.files:
                raise ValueError(f"weight source has no W array: {path}")
            weight = np.asarray(source["W"], dtype=np.float64)
        else:
            weight = np.asarray(source, dtype=np.float64)
    finally:
        if isinstance(source, np.lib.npyio.NpzFile):
            source.close()
    if weight.shape != (nstate, nstate):
        raise ValueError(f"W must have shape {(nstate, nstate)}, got {weight.shape}")
    if not np.isfinite(weight).all():
        raise ValueError("W contains non-finite values")
    symmetry_error = float(np.max(np.abs(weight - weight.T)))
    if symmetry_error > 1e-8:
        raise ValueError(f"W is not symmetric (max error {symmetry_error:g})")
    minimum_eigenvalue = float(np.linalg.eigvalsh(weight).min())
    if minimum_eigenvalue < -1e-8:
        raise ValueError(f"W is not positive semidefinite (min eigenvalue {minimum_eigenvalue:g})")
    return weight


def validate_design(design: dict[str, Any]) -> None:
    if design.get("schema") != "opendss.physical-design.v1":
        raise ValueError("design schema must be opendss.physical-design.v1")
    event_schedule(100, design["event_mix"], 0)
    background = design["background"]
    if not 0 < float(background["event_base_multiplier_min"]) <= float(
        background["event_base_multiplier_max"]
    ):
        raise ValueError("invalid background event multiplier bounds")
    if not 0 < float(background["multiplier_floor"]) < float(
        background["multiplier_ceiling"]
    ):
        raise ValueError("invalid load multiplier bounds")
    for section in ("load_ramp", "parameter_change", "topology_change"):
        if section not in design:
            raise ValueError(f"design is missing {section}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen-state, physical-telemetry OpenDSS truth"
    )
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--feeder", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--weight-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--events", type=int, default=1100)
    parser.add_argument("--steps-per-event", type=int, default=12)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.events <= 0 or args.steps_per_event <= 0 or args.dt <= 0:
        parser.error("events, steps-per-event, and dt must be positive")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is used")
    return args


def main() -> int:
    args = parse_arguments()
    master = args.master.resolve(strict=True)
    feeder_path = args.feeder.resolve(strict=True)
    design_path = args.design.resolve(strict=True)
    weight_path = args.weight_source.resolve(strict=True)
    model_root = (args.model_root or master.parent.parent).resolve(strict=True)
    if sha256_file(master) != EXPECTED_MASTER_SHA256:
        raise RuntimeError(
            "IEEE123Master.dss does not match the frozen production revision"
        )

    design = load_json(design_path)
    validate_design(design)
    feeder = np.load(feeder_path, allow_pickle=False)
    required = {
        "H_telemetry",
        "node_order",
        "node2sn",
        "theta_idx",
        "slack_supernodes",
    }
    missing = sorted(required - set(feeder.files))
    if missing:
        raise ValueError(f"feeder is missing required arrays: {missing}")
    frozen_order = [str(x) for x in feeder["node_order"]]
    node2sn = np.asarray(feeder["node2sn"], dtype=np.int64)
    theta_idx = np.asarray(feeder["theta_idx"], dtype=np.int64)
    slack_supernodes = np.asarray(feeder["slack_supernodes"], dtype=np.int64)
    frozen_h = np.asarray(feeder["H_telemetry"], dtype=np.float64)
    supernodes = int(node2sn.max()) + 1
    nstate = len(theta_idx) + supernodes
    if nstate != EXPECTED_STATES or frozen_h.shape != (EXPECTED_TELEMETRY, nstate):
        raise ValueError(
            f"expected frozen dimensions H_telemetry={(EXPECTED_TELEMETRY, EXPECTED_STATES)}, "
            f"got {frozen_h.shape} and state count {nstate}"
        )
    weight = load_weight(weight_path, nstate)

    compile_and_solve(master)
    order = list(dss.Circuit.YNodeOrder())
    if order != frozen_order:
        raise RuntimeError("OpenDSS YNodeOrder does not match frozen feeder.npz")
    vbase = voltage_bases(order)
    base_vpu = solved_vpu(vbase)
    base_ypu = system_y_pu(vbase)
    x0 = state_from_vpu(base_vpu, node2sn, theta_idx)
    specs, telemetry_names = telemetry_specs(order, slack_supernodes, node2sn)
    if len(specs) != EXPECTED_TELEMETRY:
        raise RuntimeError(f"expected 45 telemetry channels, got {len(specs)}")
    rebuilt_h = analytic_telemetry_jacobian(
        base_vpu, base_ypu, node2sn, theta_idx, specs
    )
    jacobian_error = float(np.max(np.abs(rebuilt_h - frozen_h)))
    # BLAS/OpenDSS builds can differ by a few ulps in the dense Y-bus algebra.
    if jacobian_error > 1e-8:
        raise RuntimeError(
            f"frozen telemetry Jacobian does not match model/mapping (max error {jacobian_error:g})"
        )
    base_physical = physical_telemetry(base_vpu, base_ypu, specs)
    telemetry_offset = frozen_h @ x0 - base_physical
    load_names, load_buses, base_kw, base_kvar = collect_loads()
    groups = load_groups(design["load_ramp"]["candidate_roots"], load_buses)

    if args.validate_only:
        print(
            "OPENDSS_EXPORTER_VALIDATE_OK",
            f"nodes={len(order)}",
            f"supernodes={supernodes}",
            f"states={nstate}",
            f"telemetry={len(specs)}",
            f"loads={len(load_names)}",
            f"load_groups={','.join(groups)}",
            f"jacobian_max_error={jacobian_error:.3g}",
            f"telemetry_offset_norm={np.linalg.norm(telemetry_offset):.9g}",
        )
        return 0

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite only deliberately): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    schedule = event_schedule(args.events, design["event_mix"], args.seed)
    total_steps = args.events * args.steps_per_event
    x_true = np.empty((total_steps, nstate), dtype=np.float32)
    z_true = np.empty((total_steps, len(specs)), dtype=np.float32)
    z_physical = np.empty_like(z_true)
    trajectory_id = np.empty(args.events, dtype="U96")
    event_mechanism = np.empty(args.events, dtype="U160")

    bg = design["background"]
    ramp_design = design["load_ramp"]
    tap_design = design["parameter_change"]
    tie_design = design["topology_change"]
    group_names = sorted(groups)

    for event_id, family in enumerate(schedule):
        event_rng = np.random.default_rng(np.random.SeedSequence([args.seed, event_id]))
        compile_and_solve(master)
        if list(dss.Circuit.YNodeOrder()) != frozen_order:
            raise RuntimeError(f"node order changed when event {event_id} was initialized")
        current_names, _, current_kw, current_kvar = collect_loads()
        if current_names != load_names:
            raise RuntimeError(f"load order changed when event {event_id} was initialized")
        dss.Text.Command("Set ControlMode=OFF")

        ramp_mask = np.zeros(len(load_names), dtype=bool)
        ramp_magnitude = 0.0
        ramp_direction = 0
        tap_name = ""
        tap_base = 0.0
        tap_delta = 0.0
        tie_name = ""
        if family == "load_ramp":
            group_name = str(event_rng.choice(group_names))
            ramp_mask = groups[group_name]
            ramp_magnitude = float(
                event_rng.uniform(
                    float(ramp_design["fraction_min"]),
                    float(ramp_design["fraction_max"]),
                )
            )
            ramp_direction = int(event_rng.choice(ramp_design["directions"]))
            event_mechanism[event_id] = (
                f"root={group_name};loads={int(ramp_mask.sum())};"
                f"terminal_fraction={ramp_direction * ramp_magnitude:.9g}"
            )
        elif family == "parameter_change":
            tap_name = str(event_rng.choice(tap_design["transformers"]))
            dss.Transformers.Name(tap_name)
            dss.Transformers.Wdg(2)
            tap_base = float(dss.Transformers.Tap())
            tap_steps = int(event_rng.choice(tap_design["tap_steps"]))
            tap_direction = int(event_rng.choice(tap_design["directions"]))
            tap_delta = tap_direction * tap_steps * float(tap_design["tap_step_pu"])
            event_mechanism[event_id] = (
                f"transformer={tap_name};base_tap={tap_base:.9g};"
                f"terminal_delta={tap_delta:.9g}"
            )
        elif family == "topology_change":
            tie = tie_design["ties"][int(event_rng.integers(len(tie_design["ties"])))]
            add_tie(tie)
            tie_name = str(tie["name"])
            if list(dss.Circuit.YNodeOrder()) != frozen_order:
                raise RuntimeError(f"topology tie {tie_name} changed frozen node order")
            event_mechanism[event_id] = (
                f"close_tie={tie_name};bus1={tie['bus1']};bus2={tie['bus2']}"
            )
        else:
            event_mechanism[event_id] = "none"

        base_level = float(
            event_rng.uniform(
                float(bg["event_base_multiplier_min"]),
                float(bg["event_base_multiplier_max"]),
            )
        )
        ou = np.zeros(len(load_names), dtype=np.float64)
        cloud = 0.0
        motor = np.zeros(len(load_names), dtype=np.float64)
        der_mask = event_rng.random(len(load_names)) < float(bg["der_load_fraction"])
        ou_a = math.exp(-args.dt / float(bg["ou_tau_seconds"]))
        ou_sd = float(bg["ou_sigma"]) * math.sqrt(1.0 - ou_a**2)
        cloud_a = math.exp(-args.dt / float(bg["cloud_tau_seconds"]))
        cloud_sd = float(bg["cloud_sigma"]) * math.sqrt(1.0 - cloud_a**2)

        for local_step in range(args.steps_per_event):
            fraction = (local_step + 1) / args.steps_per_event
            ou = ou_a * ou + ou_sd * event_rng.standard_normal(len(load_names))
            cloud = cloud_a * cloud + cloud_sd * float(event_rng.standard_normal())
            motor *= float(bg["motor_step_decay"])
            if event_rng.random() < float(bg["motor_step_probability_per_second"]) * args.dt:
                motor[int(event_rng.integers(len(load_names)))] += float(
                    bg["motor_step_fraction"]
                )
            multipliers = base_level * (1.0 + ou + cloud * der_mask + motor)
            if family == "load_ramp":
                multipliers[ramp_mask] *= 1.0 + ramp_direction * ramp_magnitude * fraction
            multipliers = np.clip(
                multipliers,
                float(bg["multiplier_floor"]),
                float(bg["multiplier_ceiling"]),
            )
            set_loads(load_names, current_kw, current_kvar, multipliers)
            if family == "parameter_change":
                tap = np.clip(
                    tap_base + tap_delta * fraction,
                    float(tap_design["tap_min_pu"]),
                    float(tap_design["tap_max_pu"]),
                )
                set_transformer_tap(tap_name, float(tap))
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                raise RuntimeError(
                    f"OpenDSS failed to converge at event={event_id}, step={local_step}, "
                    f"family={family}, mechanism={event_mechanism[event_id]}"
                )
            if list(dss.Circuit.YNodeOrder()) != frozen_order:
                raise RuntimeError(f"node order changed at event={event_id}, step={local_step}")
            vpu = solved_vpu(vbase)
            ypu = system_y_pu(vbase)
            state = state_from_vpu(vpu, node2sn, theta_idx)
            raw_measurement = physical_telemetry(vpu, ypu, specs)
            step_index = event_id * args.steps_per_event + local_step
            x_true[step_index] = state
            z_physical[step_index] = raw_measurement
            z_true[step_index] = raw_measurement + telemetry_offset

        trajectory_id[event_id] = f"{args.role}:seed{args.seed}:event{event_id:04d}"
        if (event_id + 1) % 25 == 0 or event_id + 1 == args.events:
            print(f"progress events={event_id + 1}/{args.events}", flush=True)

    if not np.isfinite(x_true).all() or not np.isfinite(z_true).all():
        raise RuntimeError("generated truth contains non-finite values")
    counts = {name: int(np.sum(schedule == name)) for name in np.unique(schedule)}
    meta = {
        "schema": SCHEMA,
        "artifact": "production physical truth trajectory",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": args.role,
        "physical_seed": args.seed,
        "solver": "OpenDSS via OpenDSSDirect.py",
        "model": "IEEE 123-node test feeder",
        "model_master": master.name,
        "master_sha256": sha256_file(master),
        "model_tree_root": str(model_root),
        "model_tree_sha256": sha256_tree(model_root),
        "feeder_sha256": sha256_file(feeder_path),
        "design_sha256": sha256_file(design_path),
        "weight_source_sha256": sha256_file(weight_path),
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "opendssdirect_version": importlib.metadata.version("OpenDSSDirect.py"),
        "dss_python_version": importlib.metadata.version("dss-python"),
        "dss_python_backend_version": importlib.metadata.version(
            "dss-python-backend"
        ),
        "events": args.events,
        "steps_per_event": args.steps_per_event,
        "steps": total_steps,
        "dt_seconds": args.dt,
        "nodes": len(order),
        "supernodes": supernodes,
        "states": nstate,
        "telemetry_channels": len(specs),
        "event_counts": counts,
        "state_definition": "[theta(non-slack supernodes), voltage_magnitude(all supernodes)] per unit/radians",
        "telemetry_definition": "z_true = h_OpenDSS(x) + H_telemetry@x0 - h_OpenDSS(x0)",
        "raw_telemetry_array": "z_physical",
        "telemetry_jacobian_max_error": jacobian_error,
        "label_independence": "contains no detector-derived labels",
        "event_independence": "each event recompiles and resolves the frozen base model",
        "control_policy": "base controls solved, then ControlMode=OFF within each 12-step event",
    }

    arrays = {
        "x_true": x_true,
        "z_true": z_true,
        "z_physical": z_physical,
        "time": np.arange(1, total_steps + 1, dtype=np.int64),
        "event_id": np.arange(total_steps, dtype=np.int64) // args.steps_per_event,
        "drift_family": schedule,
        "is_nominal": schedule == "nominal",
        "trajectory_id": trajectory_id,
        "event_mechanism": event_mechanism,
        "physical_seed": np.asarray(args.seed, dtype=np.int64),
        "W": weight,
        "node_order": np.asarray(order),
        "node2sn": node2sn,
        "theta_idx": theta_idx,
        "slack_supernodes": slack_supernodes,
        "telemetry_names": telemetry_names,
        "telemetry_offset": telemetry_offset,
        "x0": x0,
        "meta": np.asarray(json.dumps(meta, sort_keys=True)),
    }
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".npz", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        with np.load(temporary_path, allow_pickle=False) as check:
            if check["x_true"].shape != (total_steps, nstate):
                raise RuntimeError("atomic output verification failed for x_true")
            if check["z_true"].shape != (total_steps, EXPECTED_TELEMETRY):
                raise RuntimeError("atomic output verification failed for z_true")
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(
        "OPENDSS_TRUTH_EXPORT_OK",
        f"output={output}",
        f"steps={total_steps}",
        f"events={args.events}",
        f"states={nstate}",
        f"telemetry={len(specs)}",
        f"sha256={sha256_file(output)}",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OPENDSS_TRUTH_EXPORT_FAILED: {exc}", file=sys.stderr)
        raise
