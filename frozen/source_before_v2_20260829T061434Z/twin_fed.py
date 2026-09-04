#!/usr/bin/env python3
"""Production HELICS digital-twin federate for the v9 experiment.

This process is the estimator/trust federate.  It does not generate physical
states, communication outcomes, or oracle labels.  It receives timestamped
telemetry after ns-3/HELICS impairment, maintains the last valid sample for
each telemetry channel, constructs local pseudo-measurements from the previous
twin state, runs the common age-aware estimator, evaluates all detector scores
from that same estimate, and publishes the estimate/scores to the oracle.

Wire contract
-------------
The power federate sends the following text to ``net_fed/in``::

    <channel_id><TAB>{"schema":"twin.telemetry.v1",
                      "channel_id":<int>,
                      "value":<float>,
                      "source_time":<float>,
                      "sequence":<int>}

``net_fed`` removes the outer ``channel_id<TAB>`` routing envelope and forwards
the JSON payload to the global endpoint ``twin_fed/in``.  Keeping channel_id in
the JSON is mandatory because the current network federate forwards all
channels from one endpoint.

At each logical update, this federate sends ``twin.score.v1`` JSON to
``oracle_fed/in``.  It contains x_hat and detector scores but no oracle truth or
label.  The oracle federate alone creates labels.

Production safeguards
---------------------
* ``b <= b_min`` is starved, matching net_fed.cc exactly.
* Ages come from source timestamps, not configured latency.
* Out-of-order/duplicate packets cannot replace a newer cached sample.
* Residuals remain defined against a held estimate; they are never silently
  replaced by zero merely because the WLS solve was rejected.
* One inferential row is written per event, with maxima over the event window.
* Input hashes, versions, resolved arguments, and run status are written to
  meta.json.
* Normalizers/hybrid coefficients must be supplied by a frozen calibration
  file unless --allow-uncalibrated is explicitly selected for smoke testing.

Both exposure scalarizations use the stable row-scaled SVD formulation from the
verified metric.  This is deliberately preferred to an iterative smallest-
eigenvalue solve: the feeder is ill-conditioned enough that a converged
iterative pair can still miss the weakest near-zero mode.  The delta-confident
form uses the monotone envelopes and an exact Poisson-binomial loss quantile.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from trust_metric import MetricConfig, TrustMetric


TELEMETRY_SCHEMA = "twin.telemetry.v1"
SCORE_SCHEMA = "twin.score.v1"


class ProtocolError(ValueError):
    """A telemetry payload violates the declared wire contract."""


@dataclass(frozen=True)
class TelemetrySample:
    channel_id: int
    value: float
    source_time: float
    sequence: int
    arrival_time: float


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be finite")
    return result


def strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ProtocolError(f"{field} must be an integer")
    return result


def parse_telemetry_payload(
    payload: str | bytes,
    *,
    arrival_time: float,
    n_telemetry: int,
    future_tolerance: float,
) -> TelemetrySample:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("telemetry payload is not UTF-8") from exc
    else:
        text = str(payload)

    try:
        item = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"telemetry payload is not valid JSON: {exc}") from exc
    if not isinstance(item, dict):
        raise ProtocolError("telemetry payload must be a JSON object")
    if item.get("schema") != TELEMETRY_SCHEMA:
        raise ProtocolError(
            f"telemetry schema must be {TELEMETRY_SCHEMA!r}, "
            f"got {item.get('schema')!r}"
        )

    channel_id = strict_int(item.get("channel_id"), "channel_id")
    if not 0 <= channel_id < n_telemetry:
        raise ProtocolError(
            f"channel_id {channel_id} is outside [0,{n_telemetry - 1}]"
        )
    value = finite_float(item.get("value"), "value")
    source_time = finite_float(item.get("source_time"), "source_time")
    sequence = strict_int(item.get("sequence", 0), "sequence")
    if sequence < 0:
        raise ProtocolError("sequence must be nonnegative")
    if source_time < 0.0:
        raise ProtocolError("source_time must be nonnegative")
    if source_time > arrival_time + future_tolerance:
        raise ProtocolError(
            f"source_time {source_time:.12g} is later than arrival/granted time "
            f"{arrival_time:.12g}"
        )
    return TelemetrySample(
        channel_id=channel_id,
        value=value,
        source_time=source_time,
        sequence=sequence,
        arrival_time=float(arrival_time),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    value = result.stdout.strip()
    return value or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("calibration file must contain a JSON object")
    return data


def nested_number(data: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> float | None:
    for keys in paths:
        current: Any = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            number = finite_float(current, ".".join(keys))
            return number
    return None


def choose_parameter(
    explicit: float | None,
    calibration: dict[str, Any],
    paths: Iterable[tuple[str, ...]],
    *,
    name: str,
    fallback: float,
    allow_uncalibrated: bool,
    allow_zero: bool = False,
) -> float:
    value = explicit if explicit is not None else nested_number(calibration, paths)
    if value is None:
        if not allow_uncalibrated:
            raise ValueError(
                f"missing calibrated {name}; provide --calibration or --{name.replace('_', '-')}"
            )
        value = fallback
    value = finite_float(value, name)
    if value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def validate_arrays(
    H: np.ndarray,
    sigma2: np.ndarray,
    Q: np.ndarray,
    n_telemetry: int,
    P: np.ndarray,
    D: np.ndarray,
    B: np.ndarray,
    scenarios: pd.DataFrame,
) -> None:
    if H.ndim != 2:
        raise ValueError("H must be two-dimensional")
    m, n = H.shape
    if sigma2.shape != (m,) or Q.shape != (n, n):
        raise ValueError("feeder H/sigma2/Q shapes are incompatible")
    if P.shape != D.shape or P.shape != B.shape or P.ndim != 2 or P.shape[1] != m:
        raise ValueError("patterns P/D/B shapes are incompatible with H")
    if len(scenarios) != P.shape[0]:
        raise ValueError("scenarios.csv row count does not match patterns.npz")
    if not 0 < n_telemetry <= m:
        raise ValueError("n_telemetry is outside the measurement range")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(sigma2)):
        raise ValueError("H and sigma2 must be finite")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must be finite")
    if np.any(sigma2 <= 0.0):
        raise ValueError("all measurement variances must be positive")
    if np.any(~np.isfinite(P)) or np.any((P < 0.0) | (P > 1.0)):
        raise ValueError("all loss probabilities must be finite and in [0,1]")
    if np.any(~np.isfinite(D)) or np.any(D < 0.0):
        raise ValueError("configured ages must be finite and nonnegative")
    if np.any(~np.isfinite(B)) or np.any(B < 0.0):
        raise ValueError("bandwidth values must be finite and nonnegative")
    if not np.allclose(Q, Q.T, rtol=1e-9, atol=1e-12):
        raise ValueError("Q must be symmetric")
    minimum_q = float(np.linalg.eigvalsh(Q)[0])
    tolerance = 1e-10 * max(float(np.linalg.norm(Q, 2)), 1.0)
    if minimum_q < -tolerance:
        raise ValueError(f"Q is not positive semidefinite (lambda_min={minimum_q:.6g})")
    for column in ("arm", "regime"):
        if column not in scenarios.columns:
            raise ValueError(f"scenarios.csv is missing {column!r}")


class FastExposure:
    """Stable mean/delta information exposure for the production loop."""

    def __init__(self, metric: TrustMetric, *, exposure_form: str) -> None:
        self.metric = metric
        self.H = metric.H
        self.sigma2 = metric.s2
        self.cfg = metric.cfg
        self.m, self.n = self.H.shape
        self.exposure_form = exposure_form
        self.K_norm = np.einsum("ij,ij->i", self.H, self.H) / self.sigma2
        self.K_norm_sorted = np.sort(self.K_norm)[::-1]
        self._poisson_cache: dict[bytes, int] = {}
        self._delta_static_cache: dict[bytes, tuple[float, int, float]] = {}

    def _spectrum(self, weights: np.ndarray) -> np.ndarray:
        """Spectrum of sum_i weights_i K_i via the row-scaled H SVD.

        Forming G and calling eigvalsh produces small negative eigenvalues on
        this feeder at kappa ~ 1e9.  The singular-value route is slower than an
        iterative extremal solve but is stable in the near-unobservable cells
        that determine the exposure result.
        """
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
        row_scaled = self.H * np.sqrt(weights / self.sigma2)[:, None]
        singular_values = np.linalg.svd(row_scaled, compute_uv=False)
        spectrum = np.zeros(self.n, dtype=float)
        count = min(singular_values.size, self.n)
        spectrum[:count] = singular_values[:count] ** 2
        return np.sort(spectrum)

    def _poisson_binomial_quantile(self, probabilities: np.ndarray) -> int:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        key = clipped.tobytes()
        if key in self._poisson_cache:
            return self._poisson_cache[key]
        distribution = np.array([1.0], dtype=float)
        for probability in clipped:
            updated = np.zeros(distribution.size + 1, dtype=float)
            updated[:-1] += distribution * (1.0 - probability)
            updated[1:] += distribution * probability
            distribution = updated
        target = 1.0 - self.cfg.delta / 2.0
        quantile = int(np.searchsorted(np.cumsum(distribution), target, side="left"))
        quantile = min(quantile, clipped.size)
        self._poisson_cache[key] = quantile
        return quantile

    def _delta_static(self, p: np.ndarray) -> tuple[float, int, float]:
        """Cache the p-only Bernstein and loss-quantile constituents."""
        key = np.asarray(p, dtype=float).tobytes()
        if key in self._delta_static_cache:
            return self._delta_static_cache[key]
        phi = np.where(p <= 0.5, p * (1.0 - p), 0.25)
        vbar = float(self._spectrum(phi * self.K_norm)[-1])
        stochastic_norms = self.K_norm[p > 0.0]
        lbar = float(stochastic_norms.max()) if stochastic_norms.size else 0.0
        logarithm = math.log(4.0 * self.n / self.cfg.delta)
        floor_deflation = math.sqrt(2.0 * vbar * logarithm) + (
            2.0 * lbar / 3.0
        ) * logarithm
        loss_quantile = self._poisson_binomial_quantile(p)
        lambda_loss = float(self.K_norm_sorted[:loss_quantile].sum())
        result = (floor_deflation, loss_quantile, lambda_loss)
        self._delta_static_cache[key] = result
        return result

    def evaluate(self, p: np.ndarray, gamma: np.ndarray) -> dict[str, float]:
        p = np.asarray(p, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        mean_spectrum = self._spectrum((1.0 - p) * gamma)
        lam_mean = float(mean_spectrum[0])
        use_spectrum = mean_spectrum
        floor_kind = "mean"
        floor_deflation = 0.0
        loss_quantile = 0

        if self.exposure_form == "delta":
            floor_deflation, loss_quantile, lambda_loss = self._delta_static(p)
            gamma_spectrum = self._spectrum(gamma)
            floor_bern = np.maximum(mean_spectrum - floor_deflation, 0.0)
            floor_quantile = np.maximum(gamma_spectrum - lambda_loss, 0.0)
            use_spectrum = np.maximum(floor_bern, floor_quantile)
            floor_kind = (
                "bernstein"
                if float(floor_bern[0]) >= float(floor_quantile[0])
                else "loss_quantile"
            )

        lam_use = float(use_spectrum[0])
        u_lmax_mean = self.cfg.omega**2 * self.metric.nu_l * max(
            1.0 / (lam_mean + self.cfg.eps) - self.metric.L0,
            0.0,
        )
        u_lmax = self.cfg.omega**2 * self.metric.nu_l * max(
            1.0 / (lam_use + self.cfg.eps) - self.metric.L0,
            0.0,
        )
        trace_mean = float(np.sum(1.0 / (mean_spectrum + self.cfg.eps)))
        trace_inverse = float(np.sum(1.0 / (use_spectrum + self.cfg.eps)))
        u_trace_mean = self.cfg.omega**2 * self.metric.nu_t * max(
            trace_mean - self.metric.T0,
            0.0,
        )
        u_trace = self.cfg.omega**2 * self.metric.nu_t * max(
            trace_inverse - self.metric.T0,
            0.0,
        )

        return {
            "lam_min": lam_use,
            "lam_min_mean": lam_mean,
            "trace_inverse": trace_inverse,
            "u_lmax": u_lmax,
            "u_trace": u_trace,
            "u_lmax_mean": u_lmax_mean,
            "u_trace_mean": u_trace_mean,
            "floor_kind": floor_kind,
            "floor_deflation": floor_deflation,
            "loss_quantile": loss_quantile,
        }


class ProductionTwin:
    def __init__(
        self,
        *,
        H: np.ndarray,
        sigma2: np.ndarray,
        Q: np.ndarray,
        n_telemetry: int,
        P: np.ndarray,
        B: np.ndarray,
        scenarios: pd.DataFrame,
        metric: TrustMetric,
        exposure: FastExposure,
        b_min: float,
        beta1: float,
        beta2: float,
        hold_factor: float,
        lnr_enabled: bool,
        initial_state: np.ndarray,
    ) -> None:
        self.H = H
        self.sigma2 = sigma2
        self.Q = Q
        self.n_telemetry = n_telemetry
        self.P = P
        self.B = B
        self.scenarios = scenarios
        self.metric = metric
        self.exposure = exposure
        self.b_min = b_min
        self.beta1 = beta1
        self.beta2 = beta2
        self.hold_factor = hold_factor
        self.lnr_enabled = lnr_enabled
        self.m, self.n = H.shape
        self.x_previous = np.asarray(initial_state, dtype=float).copy()
        self.samples: list[TelemetrySample | None] = [None] * n_telemetry
        self.received_messages = 0
        self.accepted_messages = 0
        self.stale_messages = 0
        self.malformed_messages = 0
        self.new_since_update = 0

    def accept_sample(self, sample: TelemetrySample) -> bool:
        self.received_messages += 1
        previous = self.samples[sample.channel_id]
        if previous is not None:
            old_key = (previous.source_time, previous.sequence)
            new_key = (sample.source_time, sample.sequence)
            if new_key <= old_key:
                self.stale_messages += 1
                return False
        self.samples[sample.channel_id] = sample
        self.accepted_messages += 1
        self.new_since_update += 1
        return True

    def note_malformed(self) -> None:
        self.received_messages += 1
        self.malformed_messages += 1

    def _measurements(
        self,
        *,
        logical_time: float,
        event_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(self.P[event_id], dtype=float).copy()
        bandwidth = np.asarray(self.B[event_id], dtype=float).copy()
        feasible = bandwidth > self.b_min
        # TrustMetric uses b >= cfg.b_min. Mapping every starved value to zero
        # preserves the net_fed rule b <= bMin -> starved, including equality.
        bandwidth_effective = np.where(feasible, bandwidth, 0.0)

        z = np.zeros(self.m, dtype=float)
        age = np.zeros(self.m, dtype=float)
        rx = np.zeros(self.m, dtype=bool)

        for channel, sample in enumerate(self.samples):
            if sample is None:
                age[channel] = logical_time
                continue
            z[channel] = sample.value
            age[channel] = max(logical_time - sample.source_time, 0.0)
            rx[channel] = bool(feasible[channel])

        # Local pseudo-measurements are generated from the prior twin state.
        # They are fresh, never traverse the network, and use their feeder
        # variances in the common estimator.
        if self.n_telemetry < self.m:
            z[self.n_telemetry :] = self.H[self.n_telemetry :] @ self.x_previous
            age[self.n_telemetry :] = 0.0
            rx[self.n_telemetry :] = True
            p[self.n_telemetry :] = 0.0
            bandwidth_effective[self.n_telemetry :] = max(self.b_min + 1.0, 1.0e12)

        return z, age, rx, p, bandwidth_effective

    def _estimate(
        self,
        z: np.ndarray,
        rx: np.ndarray,
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        candidate, solved_exactly = self.metric.estimate(z, rx, gamma)
        scale = max(
            float(np.linalg.norm(self.x_previous)),
            math.sqrt(self.n) * 0.01,
            1e-12,
        )
        held = (
            not solved_exactly
            or not np.all(np.isfinite(candidate))
            or float(np.linalg.norm(candidate)) > self.hold_factor * scale
        )
        if held:
            return self.x_previous.copy(), True, solved_exactly
        self.x_previous = np.asarray(candidate, dtype=float).copy()
        return self.x_previous.copy(), False, solved_exactly

    def _residual_statistics(
        self,
        z: np.ndarray,
        rx: np.ndarray,
        gamma: np.ndarray,
        x_hat: np.ndarray,
    ) -> dict[str, float]:
        idx = np.flatnonzero(rx & (gamma > 0.0))
        if idx.size == 0:
            return {"r": 0.0, "chi2": 0.0, "huber": 0.0, "lnr": float("nan")}

        residual = z[idx] - self.H[idx] @ x_hat
        standardized = residual / np.sqrt(
            self.sigma2[idx] / np.maximum(gamma[idx], 1e-300)
        )
        chi2 = float(np.sum(standardized**2) / self.m)
        absolute = np.abs(standardized)
        huber_k = 1.345
        huber = float(
            np.sum(
                np.where(
                    absolute <= huber_k,
                    0.5 * absolute**2,
                    huber_k * (absolute - 0.5 * huber_k),
                )
            )
            / self.m
        )

        lnr = float("nan")
        if self.lnr_enabled:
            Hr = self.H[idx]
            weights = gamma[idx] / self.sigma2[idx]
            information = Hr.T @ (Hr * weights[:, None])
            covariance = np.linalg.pinv(information, hermitian=True)
            leverage = np.einsum("ij,jk,ik->i", Hr, covariance, Hr)
            omega = np.maximum(
                self.sigma2[idx] / np.maximum(gamma[idx], 1e-300) - leverage,
                1e-30,
            )
            lnr = float(np.max(np.abs(residual) / np.sqrt(omega)))

        return {"r": chi2, "chi2": chi2, "huber": huber, "lnr": lnr}

    def update(
        self,
        *,
        logical_time: float,
        step_index: int,
        steps_per_event: int,
    ) -> tuple[dict[str, Any], np.ndarray]:
        event_id = step_index // steps_per_event
        z, age, rx, p, bandwidth = self._measurements(
            logical_time=logical_time,
            event_id=event_id,
        )
        gamma = self.metric.g_of(age) * (bandwidth >= self.metric.cfg.b_min)
        x_hat, held, solved_exactly = self._estimate(z, rx, gamma)
        residual = self._residual_statistics(z, rx, gamma, x_hat)
        exposure = self.exposure.evaluate(p, gamma)

        r = residual["r"]
        b1 = float(np.mean(rx[: self.n_telemetry]))
        b2 = float(np.mean(age[: self.n_telemetry]))
        s_lmax = r / self.metric.cfg.r0 + exposure["u_lmax"] / self.metric.cfg.u0_lmax
        s_trace = float("nan")
        if math.isfinite(exposure["u_trace"]):
            s_trace = r / self.metric.cfg.r0 + exposure["u_trace"] / self.metric.cfg.u0_trace
        active_s = s_lmax if self.metric.cfg.scalarization == "lmax" else s_trace
        if not math.isfinite(active_s):
            raise RuntimeError("active trust score is nonfinite")

        scenario = self.scenarios.iloc[event_id]
        row: dict[str, Any] = {
            "time": float(logical_time),
            "step_index": int(step_index),
            "event_id": int(event_id),
            "step_in_event": int(step_index % steps_per_event),
            "pattern_id": int(event_id),
            "arm": None if pd.isna(scenario.get("arm")) else str(scenario.get("arm")),
            "regime": None
            if pd.isna(scenario.get("regime"))
            else str(scenario.get("regime")),
            "stratum": None
            if pd.isna(scenario.get("stratum"))
            else int(scenario.get("stratum")),
            "r": r,
            "chi2": residual["chi2"],
            "huber": residual["huber"],
            "lnr": residual["lnr"],
            "u_lmax": exposure["u_lmax"],
            "u_trace": exposure["u_trace"],
            "u_lmax_mean": exposure["u_lmax_mean"],
            "u_trace_mean": exposure["u_trace_mean"],
            "s_lmax": s_lmax,
            "s_trace": s_trace,
            "s": active_s,
            "T": float(math.exp(-min(active_s, 745.0))),
            "alarm": bool(active_s > math.log(1.0 / self.metric.cfg.T_th)),
            "sB1": r / self.metric.cfg.r0 + self.beta1 * (1.0 - b1),
            "sB2": r / self.metric.cfg.r0 + self.beta2 * b2,
            "s_gated_lmax": b1 * r / self.metric.cfg.r0
            + exposure["u_lmax"] / self.metric.cfg.u0_lmax,
            "s_gated_trace": float("nan")
            if not math.isfinite(exposure["u_trace"])
            else b1 * r / self.metric.cfg.r0
            + exposure["u_trace"] / self.metric.cfg.u0_trace,
            "b1": b1,
            "b2": b2,
            "n_rx": int(rx.sum()),
            "n_rx_telemetry": int(rx[: self.n_telemetry].sum()),
            "cached_telemetry": int(sum(sample is not None for sample in self.samples)),
            "new_telemetry": int(self.new_since_update),
            "held": bool(held),
            "solve_exact": bool(solved_exactly),
            "residual_available": bool(np.any(rx & (gamma > 0.0))),
            "mean_age_telemetry": b2,
            "max_age_telemetry": float(np.max(age[: self.n_telemetry])),
            "lam_min": exposure["lam_min"],
            "lam_min_mean": exposure["lam_min_mean"],
            "trace_inverse": exposure["trace_inverse"],
            "floor_kind": exposure["floor_kind"],
            "floor_deflation": exposure["floor_deflation"],
            "loss_quantile": int(exposure["loss_quantile"]),
        }
        self.new_since_update = 0
        return row, x_hat


MAX_EVENT_COLUMNS = (
    "s",
    "s_lmax",
    "s_trace",
    "r",
    "chi2",
    "huber",
    "lnr",
    "u_lmax",
    "u_trace",
    "u_lmax_mean",
    "u_trace_mean",
    "sB1",
    "sB2",
    "s_gated_lmax",
    "s_gated_trace",
    "b2",
    "mean_age_telemetry",
    "max_age_telemetry",
)


def aggregate_events(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, block in scores.groupby("event_id", sort=True):
        first = block.iloc[0]
        row: dict[str, Any] = {
            "event_id": int(first["event_id"]),
            "pattern_id": int(first["pattern_id"]),
            "arm": first["arm"],
            "regime": first["regime"],
            "stratum": first["stratum"],
            "T": float(block["T"].min()),
            "alarm": bool(block["alarm"].any()),
            "b1": float(block["b1"].min()),
            "n_rx": int(block["n_rx"].min()),
            "n_rx_telemetry": int(block["n_rx_telemetry"].min()),
            "held": float(block["held"].mean()),
            "held_any": bool(block["held"].any()),
            "residual_available": bool(block["residual_available"].any()),
            "steps": int(len(block)),
        }
        for column in MAX_EVENT_COLUMNS:
            values = pd.to_numeric(block[column], errors="coerce").to_numpy(float)
            row[column] = float(np.nanmax(values)) if np.any(np.isfinite(values)) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def first_attribute(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"none of {names!r} exists in the HELICS Python module")


def message_text(helics: Any, message: Any) -> str:
    if hasattr(helics, "helicsMessageIsValid") and not bool(
        helics.helicsMessageIsValid(message)
    ):
        raise ProtocolError("received an invalid HELICS telemetry message")
    if hasattr(helics, "helicsMessageGetBytes"):
        data = bytes(helics.helicsMessageGetBytes(message))
    else:
        value = helics.helicsMessageGetString(message)
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    if not data:
        raise ProtocolError("received an empty HELICS telemetry payload")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("HELICS telemetry payload is not UTF-8") from exc


def send_text(helics: Any, endpoint: Any, payload: str, destination: str) -> None:
    if hasattr(helics, "helicsEndpointSendStringTo"):
        helics.helicsEndpointSendStringTo(endpoint, payload, destination)
        return
    helics.helicsEndpointSendBytesTo(endpoint, payload.encode("utf-8"), destination)


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
    core_init = args.core_init.strip()
    if not core_init:
        core_init = f"--federates=1 --broker_address={args.broker}"
    h.helicsFederateInfoSetCoreInitString(info, core_init)
    if hasattr(h, "helicsFederateInfoSetBroker"):
        h.helicsFederateInfoSetBroker(info, args.broker)
    time_delta = first_attribute(
        h,
        "HELICS_PROPERTY_TIME_DELTA",
        "helics_property_time_delta",
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
    input_endpoint = h.helicsFederateRegisterGlobalEndpoint(
        federate,
        args.input_endpoint,
        "json",
    )
    output_endpoint = h.helicsFederateRegisterGlobalEndpoint(
        federate,
        args.output_endpoint,
        "json",
    )
    input_name = h.helicsEndpointGetName(input_endpoint)
    output_name = h.helicsEndpointGetName(output_endpoint)
    if input_name != args.input_endpoint or output_name != args.output_endpoint:
        raise RuntimeError(
            "HELICS endpoint registration mismatch: "
            f"input={input_name!r}, output={output_name!r}"
        )
    h.helicsEndpointSetDefaultDestination(
        output_endpoint, args.output_destination
    )
    return h, info, federate, (input_endpoint, output_endpoint)


def drain_messages(
    helics: Any,
    endpoint: Any,
    twin: ProductionTwin,
    *,
    granted_time: float,
    future_tolerance: float,
    strict_protocol: bool,
) -> int:
    accepted = 0
    while bool(helics.helicsEndpointHasMessage(endpoint)):
        # PyHELICS owns this HelicsMessage wrapper and installs its
        # helicsMessageFree finalizer; never free it a second time.
        message = helics.helicsEndpointGetMessage(endpoint)
        if message is None:
            break
        try:
            sample = parse_telemetry_payload(
                message_text(helics, message),
                arrival_time=granted_time,
                n_telemetry=twin.n_telemetry,
                future_tolerance=future_tolerance,
            )
            accepted += int(twin.accept_sample(sample))
        except ProtocolError as exc:
            twin.note_malformed()
            if strict_protocol:
                raise
            print(f"[WARN] rejected telemetry at t={granted_time:.12g}: {exc}", flush=True)
    return accepted


def publish_score(
    helics: Any,
    endpoint: Any,
    destination: str,
    row: dict[str, Any],
    x_hat: np.ndarray,
    *,
    publish_state: bool,
) -> None:
    if not destination:
        return
    payload: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "time": row["time"],
        "step_index": row["step_index"],
        "event_id": row["event_id"],
        "pattern_id": row["pattern_id"],
        "arm": row["arm"],
        "regime": row["regime"],
        "stratum": row["stratum"],
        "T": row["T"],
        "s": row["s"],
        "s_lmax": row["s_lmax"],
        "s_trace": row["s_trace"],
        "r": row["r"],
        "chi2": row["chi2"],
        "huber": row["huber"],
        "lnr": row["lnr"],
        "u_lmax": row["u_lmax"],
        "u_trace": row["u_trace"],
        "u_lmax_mean": row["u_lmax_mean"],
        "u_trace_mean": row["u_trace_mean"],
        "sB1": row["sB1"],
        "sB2": row["sB2"],
        "s_gated_lmax": row["s_gated_lmax"],
        "s_gated_trace": row["s_gated_trace"],
        "b1": row["b1"],
        "b2": row["b2"],
        "n_rx": row["n_rx"],
        "n_rx_telemetry": row["n_rx_telemetry"],
        "held": row["held"],
        "solve_exact": row["solve_exact"],
        "floor_kind": row["floor_kind"],
        "loss_quantile": row["loss_quantile"],
    }
    if publish_state:
        payload["x_hat"] = x_hat.tolist()
    encoded = json.dumps(json_safe(payload), separators=(",", ":"))
    send_text(helics, endpoint, encoded, destination)


def resolved_stop_time(args: argparse.Namespace, pattern_count: int) -> tuple[float, int]:
    full_steps = pattern_count * args.steps_per_event
    if args.stop_time <= 0.0:
        return full_steps * args.dt, full_steps
    steps_float = args.stop_time / args.dt
    steps = int(round(steps_float))
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--stop-time must be an integer multiple of --dt")
    if steps < 1 or steps > full_steps:
        raise ValueError(
            f"--stop-time selects {steps} updates; allowed range is 1..{full_steps}"
        )
    return float(args.stop_time), steps


def staged_request_time(
    step_index: int,
    *,
    dt: float,
    helics_time_delta: float,
    stage: int,
) -> float:
    """Map a zero-based logical step to its acyclic HELICS microstep."""
    if step_index < 0 or dt <= 0.0 or helics_time_delta <= 0.0 or stage < 0:
        raise ValueError("invalid staged HELICS time")
    return (step_index + 1) * dt + stage * helics_time_delta


def make_meta(
    args: argparse.Namespace,
    *,
    files: dict[str, Path],
    H: np.ndarray,
    n_telemetry: int,
    pattern_count: int,
    stop_time: float,
    total_steps: int,
    calibration: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema": "twin.run.meta.v1",
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("helics", "numpy", "pandas", "pyarrow", "scipy")
        },
        "git_sha": git_sha(Path.cwd()),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "dimensions": {
            "measurements": int(H.shape[0]),
            "states": int(H.shape[1]),
            "telemetry": int(n_telemetry),
            "patterns": int(pattern_count),
        },
        "timing": {
            "dt": args.dt,
            "helics_time_delta": args.helics_time_delta,
            "stage_offset": 2.0 * args.helics_time_delta,
            "steps_per_event": args.steps_per_event,
            "stop_time": stop_time,
            "total_steps": total_steps,
            "tail_time": args.tail_time,
        },
        "wire_contract": {
            "telemetry_schema": TELEMETRY_SCHEMA,
            "score_schema": SCORE_SCHEMA,
            "input_endpoint": args.input_endpoint,
            "output_endpoint": args.output_endpoint,
            "output_destination": args.output_destination,
        },
        "resolved_arguments": vars(args),
        "calibration_file_content": calibration,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production HELICS digital-twin federate for v9"
    )
    parser.add_argument("--feeder", default="feeder.npz")
    parser.add_argument("--patterns", default="patterns.npz")
    parser.add_argument("--scenarios", default="scenarios.csv")
    parser.add_argument("--calibration")
    parser.add_argument("--initial-state")
    parser.add_argument("--out-dir", default="runs/twin")

    parser.add_argument("--federate-name", default="twin_fed")
    parser.add_argument("--input-endpoint", default="twin_fed/in")
    parser.add_argument("--output-endpoint", default="twin_fed/out")
    parser.add_argument("--output-destination", default="oracle_fed/in")
    parser.add_argument("--upstream-federate", default="net_fed")
    parser.add_argument(
        "--broker",
        default=os.environ.get("HELICS_BROKER_ADDRESS", "tcp://broker:23404"),
    )
    parser.add_argument("--core-type", default="zmq")
    parser.add_argument("--core-init", default="")

    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--helics-time-delta",
        type=float,
        default=1.0e-6,
        help="HELICS timing resolution; independent of the logical update interval",
    )
    parser.add_argument("--steps-per-event", type=int, default=12)
    parser.add_argument(
        "--stop-time",
        type=float,
        default=0.0,
        help="0 derives the full horizon from patterns x steps/event",
    )
    parser.add_argument("--tail-time", type=float, default=30.0)
    parser.add_argument("--b-min", type=float, default=1.0)
    parser.add_argument("--future-time-tolerance", type=float, default=1e-9)

    parser.add_argument("--scalarization", choices=("lmax", "trace"), default="lmax")
    parser.add_argument("--exposure-form", choices=("mean", "delta"), default="mean")
    parser.add_argument(
        "--T-th",
        type=float,
        default=None,
        help="frozen trust threshold; read from calibration when omitted",
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--u0-lmax", type=float)
    parser.add_argument("--u0-trace", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--hold-factor", type=float, default=50.0)
    parser.add_argument("--lnr", action="store_true")

    parser.add_argument(
        "--strict-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--publish-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every-events", type=int, default=50)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def check_args(args: argparse.Namespace) -> None:
    if (
        args.dt <= 0.0
        or args.helics_time_delta <= 0.0
        or args.steps_per_event < 1
        or not args.upstream_federate
    ):
        raise ValueError(
            "--dt, --helics-time-delta, and --steps-per-event must be positive"
        )
    if args.tail_time < 0.0 or args.b_min < 0.0:
        raise ValueError("--tail-time and --b-min must be nonnegative")
    if args.future_time_tolerance < 0.0:
        raise ValueError("--future-time-tolerance must be nonnegative")
    if args.T_th is not None and not 0.0 < args.T_th < 1.0:
        raise ValueError("--T-th must lie in (0,1)")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("--delta must lie in (0,1)")
    if args.hold_factor <= 1.0:
        raise ValueError("--hold-factor must exceed 1")
    if args.checkpoint_every_events < 0:
        raise ValueError("--checkpoint-every-events must be nonnegative")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_args(args)

    feeder_path = Path(args.feeder)
    patterns_path = Path(args.patterns)
    scenarios_path = Path(args.scenarios)
    calibration_path = Path(args.calibration) if args.calibration else None
    for path in (feeder_path, patterns_path, scenarios_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if calibration_path is not None and not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    feeder = np.load(feeder_path, allow_pickle=False)
    H = np.asarray(feeder["H"], dtype=float)
    sigma2 = np.asarray(feeder["sigma2"], dtype=float)
    Q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])

    patterns = np.load(patterns_path, allow_pickle=False)
    P = np.asarray(patterns["P"], dtype=float)
    D_configured = np.asarray(patterns["D"], dtype=float)
    B = np.asarray(patterns["B"], dtype=float)
    if "n_telemetry" in patterns.files and int(patterns["n_telemetry"]) != n_telemetry:
        raise ValueError("feeder and patterns disagree on n_telemetry")
    scenarios = pd.read_csv(scenarios_path)
    validate_arrays(H, sigma2, Q, n_telemetry, P, D_configured, B, scenarios)

    calibration = load_calibration(calibration_path)
    r0 = choose_parameter(
        args.r0,
        calibration,
        (("normalizers", "r0"), ("r0",)),
        name="r0",
        fallback=1.0,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    u0_lmax = choose_parameter(
        args.u0_lmax,
        calibration,
        (("normalizers", "u0_lmax"), ("u0_lmax",)),
        name="u0_lmax",
        fallback=1.0,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    u0_trace = choose_parameter(
        args.u0_trace,
        calibration,
        (("normalizers", "u0_trace"), ("u0_trace",)),
        name="u0_trace",
        fallback=1.0,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    beta1 = choose_parameter(
        args.beta1,
        calibration,
        (("hybrid_calibration", "beta1"), ("beta1",)),
        name="beta1",
        fallback=1.0,
        allow_uncalibrated=args.allow_uncalibrated,
        allow_zero=True,
    )
    beta2 = choose_parameter(
        args.beta2,
        calibration,
        (("hybrid_calibration", "beta2"), ("beta2",)),
        name="beta2",
        fallback=1.0,
        allow_uncalibrated=args.allow_uncalibrated,
        allow_zero=True,
    )
    T_th = choose_parameter(
        args.T_th,
        calibration,
        (("threshold", "T_th"), ("T_th",)),
        name="T_th",
        fallback=0.70,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    if not 0.0 < T_th < 1.0:
        raise ValueError("T_th must lie in (0,1)")
    # Persist the fully resolved value in meta.json rather than the original
    # command-line None.
    args.T_th = T_th

    config = MetricConfig(
        scalarization=args.scalarization,
        exposure_form=args.exposure_form,
        r0=r0,
        u0_lmax=u0_lmax,
        u0_trace=u0_trace,
        T_th=T_th,
        delta=args.delta,
        b_min=args.b_min,
        n_telemetry=n_telemetry,
    )
    metric = TrustMetric(H, sigma2, Q, config)
    metric.check_feasibility(verbose=True)
    exposure = FastExposure(
        metric,
        exposure_form=args.exposure_form,
    )

    if args.initial_state:
        initial_state = np.asarray(np.load(args.initial_state), dtype=float)
        if initial_state.shape != (H.shape[1],) or not np.all(np.isfinite(initial_state)):
            raise ValueError("--initial-state must be a finite vector of length n")
    else:
        initial_state = np.zeros(H.shape[1], dtype=float)

    stop_time, total_steps = resolved_stop_time(args, len(P))
    print(
        f"twin_fed production preflight: m={H.shape[0]} n={H.shape[1]} "
        f"telemetry={n_telemetry} patterns={len(P)} steps={total_steps} "
        f"stop={stop_time:g}s b_min={args.b_min:g} "
        f"stage_offset={2.0 * args.helics_time_delta:g}s",
        flush=True,
    )
    print(
        f"metric: scalarization={args.scalarization} exposure={args.exposure_form} "
        f"spectrum_solver=row-scaled-SVD r0={r0:.6g} "
        f"u0_lmax={u0_lmax:.6g} u0_trace={u0_trace:.6g}",
        flush=True,
    )
    if args.validate_only:
        print("PRODUCTION_TWIN_VALIDATE_OK", flush=True)
        return 0

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    score_path = output_directory / "scores.parquet"
    event_path = output_directory / "scores_events.parquet"
    meta_path = output_directory / "meta.json"
    files = {
        "feeder": feeder_path,
        "patterns": patterns_path,
        "scenarios": scenarios_path,
    }
    if calibration_path is not None:
        files["calibration"] = calibration_path
    meta = make_meta(
        args,
        files=files,
        H=H,
        n_telemetry=n_telemetry,
        pattern_count=len(P),
        stop_time=stop_time,
        total_steps=total_steps,
        calibration=calibration,
        status="starting",
    )
    write_json_atomic(meta_path, meta)

    twin = ProductionTwin(
        H=H,
        sigma2=sigma2,
        Q=Q,
        n_telemetry=n_telemetry,
        P=P,
        B=B,
        scenarios=scenarios,
        metric=metric,
        exposure=exposure,
        b_min=args.b_min,
        beta1=beta1,
        beta2=beta2,
        hold_factor=args.hold_factor,
        lnr_enabled=args.lnr,
        initial_state=initial_state,
    )

    helics = info = federate = None
    rows: list[dict[str, Any]] = []
    wall_start = time.time()
    completed = False
    try:
        helics, info, federate, endpoints = build_federate(args)
        input_endpoint, output_endpoint = endpoints
        meta["status"] = "connecting"
        write_json_atomic(meta_path, meta)
        helics.helicsFederateEnterExecutingMode(federate)
        print(
            f"HELICS connected: {args.federate_name} input={args.input_endpoint} "
            f"output={args.output_endpoint}->{args.output_destination}",
            flush=True,
        )
        meta["status"] = "running"
        write_json_atomic(meta_path, meta)

        requested_time = staged_request_time(
            0,
            dt=args.dt,
            helics_time_delta=args.helics_time_delta,
            stage=2,
        )
        step_index = 0
        while step_index < total_steps:
            granted_time = float(
                helics.helicsFederateRequestTime(federate, requested_time)
            )
            if granted_time > requested_time + args.future_time_tolerance:
                raise RuntimeError(
                    f"HELICS granted {granted_time} beyond requested {requested_time}"
                )
            drain_messages(
                helics,
                input_endpoint,
                twin,
                granted_time=granted_time,
                future_tolerance=args.future_time_tolerance,
                strict_protocol=args.strict_protocol,
            )
            # Future-dated network delivery can create intermediate grants.
            # Cache those messages and request the same logical update again.
            if granted_time + args.future_time_tolerance < requested_time:
                continue

            row, x_hat = twin.update(
                logical_time=(step_index + 1) * args.dt,
                step_index=step_index,
                steps_per_event=args.steps_per_event,
            )
            rows.append(row)
            publish_score(
                helics,
                output_endpoint,
                args.output_destination,
                row,
                x_hat,
                publish_state=args.publish_state,
            )

            step_index += 1
            requested_time = staged_request_time(
                step_index,
                dt=args.dt,
                helics_time_delta=args.helics_time_delta,
                stage=2,
            )
            if (
                args.checkpoint_every_events > 0
                and step_index % (
                    args.checkpoint_every_events * args.steps_per_event
                )
                == 0
            ):
                partial = pd.DataFrame(rows)
                write_parquet_atomic(output_directory / "scores.partial.parquet", partial)
                write_parquet_atomic(
                    output_directory / "scores_events.partial.parquet",
                    aggregate_events(partial),
                )
                print(
                    f"checkpoint: steps={step_index}/{total_steps} "
                    f"events={step_index // args.steps_per_event}",
                    flush=True,
                )

        if args.tail_time > 0.0:
            tail_target = (
                stop_time + args.tail_time + 2.0 * args.helics_time_delta
            )
            while True:
                granted_time = float(
                    helics.helicsFederateRequestTime(federate, tail_target)
                )
                drain_messages(
                    helics,
                    input_endpoint,
                    twin,
                    granted_time=granted_time,
                    future_tolerance=args.future_time_tolerance,
                    strict_protocol=args.strict_protocol,
                )
                if granted_time + args.future_time_tolerance >= tail_target:
                    break
        completed = True
    except KeyboardInterrupt:
        print("twin_fed interrupted", file=sys.stderr, flush=True)
        raise
    finally:
        if federate is not None and helics is not None:
            try:
                disconnect_federate(helics, federate)
            except Exception as exc:  # best-effort shutdown after a run error
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
            scores = pd.DataFrame(rows)
            write_parquet_atomic(score_path, scores)
            write_parquet_atomic(event_path, aggregate_events(scores))
        meta["status"] = "complete" if completed else "failed_or_interrupted"
        meta["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        meta["wall_seconds"] = time.time() - wall_start
        meta["runtime_counts"] = {
            "steps": len(rows),
            "events": int(math.ceil(len(rows) / args.steps_per_event)) if rows else 0,
            "messages_received": twin.received_messages,
            "messages_accepted": twin.accepted_messages,
            "messages_stale_or_duplicate": twin.stale_messages,
            "messages_malformed": twin.malformed_messages,
            "cached_telemetry_final": int(
                sum(sample is not None for sample in twin.samples)
            ),
        }
        write_json_atomic(meta_path, meta)

    print(
        f"twin_fed complete: steps={len(rows)} "
        f"events={len(aggregate_events(pd.DataFrame(rows)))} "
        f"accepted={twin.accepted_messages} stale={twin.stale_messages} "
        f"malformed={twin.malformed_messages}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ProtocolError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
