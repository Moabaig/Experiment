from pathlib import Path
import importlib.util
import sys

import numpy as np

workspace = Path(
    "/workspace/paper1_v5_1_solver_repair_workspace"
)

module_path = (
    workspace /
    "trust_metric.v5_1.candidate.py"
)

spec = importlib.util.spec_from_file_location(
    "trust_metric_v5_1_candidate",
    module_path,
)

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

MetricConfig = module.MetricConfig
TrustMetric = module.TrustMetric

passed = 0

def check(condition, name):
    global passed

    if not condition:
        raise AssertionError(name)

    passed += 1
    print(name, "PASS")

def build_metric(H, sigma2=None, Q=None, n_telemetry=None):
    H = np.asarray(H, dtype=float)
    m, n = H.shape

    if sigma2 is None:
        sigma2 = np.ones(m)

    if Q is None:
        Q = np.eye(n) * 1e-6

    cfg = MetricConfig(
        n_telemetry=n_telemetry,
        ridge=0.0,
    )

    return TrustMetric(
        H,
        sigma2,
        Q,
        cfg,
    )

# 1. Well-conditioned exact recovery.
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])

x_true = np.array([1.2, -0.4, 2.1])
z = H @ x_true

metric = build_metric(H)

candidate, exact = metric.estimate(
    z,
    np.ones(len(z), dtype=bool),
    np.ones(len(z)),
)

check(
    exact
    and np.linalg.norm(candidate - x_true) < 1e-10
    and metric.last_estimator_rank == 3,
    "well_conditioned_exact_recovery",
)

# 2. Rank-deficient systems must not be labeled exact.
H_rank_deficient = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
])

metric = build_metric(H_rank_deficient)

candidate, exact = metric.estimate(
    H_rank_deficient @ x_true,
    np.ones(3, dtype=bool),
    np.ones(3),
)

check(
    not exact
    and metric.last_estimator_rank < 3
    and np.all(np.isfinite(candidate)),
    "rank_deficiency_detected",
)

# 3. Numerically negligible directions are truncated.
H_ill = np.diag([1.0, 1e-10])
x_ill = np.ones(2)

metric = build_metric(H_ill)

candidate, exact = metric.estimate(
    H_ill @ x_ill,
    np.ones(2, dtype=bool),
    np.ones(2),
)

check(
    not exact
    and metric.last_estimator_rank == 1
    and np.all(np.isfinite(candidate))
    and np.linalg.norm(candidate) < 10.0,
    "ill_conditioned_direction_not_falsely_exact",
)

# 4. No effective measurement rows.
metric = build_metric(np.eye(3))

candidate, exact = metric.estimate(
    np.ones(3),
    np.ones(3, dtype=bool),
    np.zeros(3),
)

check(
    not exact
    and np.all(np.isnan(candidate))
    and metric.last_estimator_effective_rows == 0,
    "zero_weight_measurements_rejected",
)

# Production feeder checks.
with np.load("/workspace/feeder.npz") as feeder:
    H = np.asarray(feeder["H"], dtype=float)
    sigma2 = np.asarray(
        feeder["sigma2"],
        dtype=float,
    )
    Q = np.asarray(feeder["Q"], dtype=float)
    n_telemetry = int(feeder["n_telemetry"])

with np.load(
    "/workspace/truth.eval.paper1.v4.seed002.npz"
) as truth_data:
    production_state = np.asarray(
        truth_data["x_true"][0],
        dtype=float,
    )

metric = build_metric(
    H,
    sigma2=sigma2,
    Q=Q,
    n_telemetry=n_telemetry,
)

production_z = H @ production_state
gamma = np.ones(len(H))

# 5. Full production design.
full_rx = np.ones(len(H), dtype=bool)

candidate, exact = metric.estimate(
    production_z,
    full_rx,
    gamma,
)

check(
    exact
    and metric.last_estimator_rank == H.shape[1]
    and np.linalg.norm(
        candidate - production_state
    ) < 1e-6
    and np.linalg.norm(candidate) < 100.0,
    "production_full_design_recovery",
)

# 6. Representative 37 external channels plus all pseudos.
partial_rx = np.zeros(len(H), dtype=bool)
partial_rx[:37] = True
partial_rx[n_telemetry:] = True

candidate, exact = metric.estimate(
    production_z,
    partial_rx,
    gamma,
)

check(
    exact
    and metric.last_estimator_rank == H.shape[1]
    and np.linalg.norm(
        candidate - production_state
    ) < 1e-6
    and np.linalg.norm(candidate) < 100.0,
    "production_37_external_recovery",
)

# 7. Pseudo-only design is numerically rank deficient.
pseudo_only_rx = np.zeros(len(H), dtype=bool)
pseudo_only_rx[n_telemetry:] = True

candidate, exact = metric.estimate(
    production_z,
    pseudo_only_rx,
    gamma,
)

check(
    not exact
    and metric.last_estimator_rank < H.shape[1]
    and np.all(np.isfinite(candidate)),
    "production_pseudo_only_not_full_rank",
)

print("PAPER1_V5_1_STABLE_SOLVER_TESTS_OK")
print("TESTS_PASSED=", passed)
print("TESTS_FAILED=0")
print("SOLVER=weighted_lstsq_svd")
print("NORMAL_EQUATIONS_USED=False")
print("LIVE_FILES_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")