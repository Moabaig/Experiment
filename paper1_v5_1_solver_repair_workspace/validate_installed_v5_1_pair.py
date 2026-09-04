from __future__ import annotations

import ast
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace")

EXPECTED_TRUST = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)
EXPECTED_TWIN = (
    "39e6729af233032ab9c58851c9682252"
    "f02d36eed739eb2ec769e165659da34c"
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

require(
    sha256(ROOT / "trust_metric.py") == EXPECTED_TRUST,
    "Installed trust_metric.py hash mismatch.",
)
require(
    sha256(ROOT / "twin_fed.py") == EXPECTED_TWIN,
    "Installed twin_fed.py hash mismatch.",
)

sys.path.insert(0, str(ROOT))

from trust_metric import MetricConfig, TrustMetric
import twin_fed

feeder = np.load(ROOT / "feeder.npz")
H = np.asarray(feeder["H"], dtype=float)
sigma2 = np.asarray(feeder["sigma2"], dtype=float)
Q = np.asarray(feeder["Q"], dtype=float)

m, n = H.shape
n_telemetry = 45

cfg = MetricConfig(n_telemetry=n_telemetry)
metric = TrustMetric(H, sigma2, Q, cfg)

x = np.linspace(0.9, 1.1, n)
x *= 30.8 / np.linalg.norm(x)

z = H @ x
rx_full = np.ones(m, dtype=bool)
gamma = np.ones(m, dtype=float)

candidate, reliable = metric.estimate(
    z,
    rx_full,
    gamma,
)

candidate = np.asarray(candidate, dtype=float)
relative_error = float(
    np.linalg.norm(candidate - x) / np.linalg.norm(x)
)

require(reliable, "Full production design was not reliable.")
require(
    relative_error < 1e-7,
    f"Full-design recovery error is too large: {relative_error}",
)
require(
    metric.last_estimator_solver == "weighted_lstsq_svd",
    "Installed solver identity is incorrect.",
)
require(
    metric.last_estimator_rank == n,
    "Full production design did not reach full rank.",
)

rx_pseudo = np.zeros(m, dtype=bool)
rx_pseudo[n_telemetry:] = True

_, pseudo_reliable = metric.estimate(
    z,
    rx_pseudo,
    gamma,
)
pseudo_rank = int(metric.last_estimator_rank)

require(
    not pseudo_reliable,
    "Pseudo-only design was incorrectly classified reliable.",
)
require(
    metric.last_estimator_rank < n,
    "Pseudo-only design was unexpectedly full rank.",
)

# Re-establish full-design diagnostics before exercising the twin.
metric.estimate(z, rx_full, gamma)

twin = object.__new__(twin_fed.ProductionTwin)
twin.n = n
twin.n_telemetry = n_telemetry
twin.hold_factor = 50.0
twin.x_previous = np.zeros(n)
twin.has_valid_estimate = False
twin.metric = metric

estimate, held, reliable = twin._estimate(
    z,
    rx_full,
    gamma,
)

expected_limit = (
    twin.hold_factor
    * math.sqrt(n)
    * cfg.omega
)

require(reliable, "Twin bootstrap solver was not reliable.")
require(not held, "Reliable twin bootstrap was held.")
require(
    twin.last_hold_reason == "bootstrap_accept",
    "Twin bootstrap reason is incorrect.",
)
require(
    twin.last_jump_guard_policy == "fixed_model_increment",
    "Twin guard policy is incorrect.",
)
require(
    math.isclose(
        twin.last_jump_limit,
        expected_limit,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ),
    "Twin fixed jump limit is incorrect.",
)

increment = np.full(n, 0.1 / math.sqrt(n))
x_next = x + increment
z_next = H @ x_next

_, second_held, second_reliable = twin._estimate(
    z_next,
    rx_full,
    gamma,
)

require(second_reliable, "Second full-design solve was unreliable.")
require(not second_held, "Valid small state increment was held.")
require(
    twin.last_hold_reason == "accepted",
    "Second state update was not marked accepted.",
)

source = (ROOT / "twin_fed.py").read_text(
    encoding="utf-8-sig"
)
tree = ast.parse(source)

required_tokens = (
    '"estimator_reliable"',
    '"estimator_solver"',
    '"estimator_rank"',
    '"estimator_condition"',
    '"estimator_residual_norm"',
    '"jump_guard_policy"',
    '"model_increment_scale"',
    '"fixed_model_increment"',
)

for token in required_tokens:
    require(
        token in source,
        f"Installed twin logging token is missing: {token}",
    )

require(
    "self.hold_factor * state_scale" not in source,
    "Obsolete self-scaling guard remains installed.",
)

print("PAPER1_V5_1_INSTALLED_PAIR_VALIDATION_OK")
print("FULL_DESIGN_RELIABLE=True")
print("FULL_DESIGN_RANK=", n)
print("FULL_DESIGN_RELATIVE_ERROR=", relative_error)
print("PSEUDO_ONLY_RELIABLE=False")
print("PSEUDO_ONLY_RANK=", pseudo_rank)
print("BOOTSTRAP_ACCEPTED=True")
print("SECOND_SMALL_INCREMENT_ACCEPTED=True")
print("JUMP_GUARD_POLICY=fixed_model_increment")
print("FIXED_JUMP_LIMIT=", expected_limit)
print("SOLVER=weighted_lstsq_svd")
print("NORMAL_EQUATIONS_USED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
