from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace")
WORKSPACE = ROOT / "paper1_v5_1_solver_repair_workspace"
TRUST = WORKSPACE / "trust_metric.v5_1.candidate.py"
TWIN = WORKSPACE / "twin_fed.v5_1.candidate.py"

EXPECTED_LIVE_TRUST = (
    "be0d1dc0c5f8924a7794b9923d9ee3bb"
    "373da08266c65ff16a968cbe2c3e1ab4"
)
EXPECTED_LIVE_TWIN = (
    "009bdcc85fec147d0885bec97f0c762de"
    "924626a5f3b8148516ccb56a8d822cf"
)
EXPECTED_TRUST_CANDIDATE = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

require(sha256(ROOT / "trust_metric.py") == EXPECTED_LIVE_TRUST,
        "Live trust_metric.py changed unexpectedly.")
require(sha256(ROOT / "twin_fed.py") == EXPECTED_LIVE_TWIN,
        "Live twin_fed.py changed unexpectedly.")
require(sha256(TRUST) == EXPECTED_TRUST_CANDIDATE,
        "V5.1 solver candidate hash changed unexpectedly.")
require(sha256(TWIN) == EXPECTED_LIVE_TWIN,
        "Initial V5.1 twin candidate is not the installed V5 twin.")

module_name = "paper1_v5_1_trust_candidate"
spec = importlib.util.spec_from_file_location(module_name, TRUST)
require(spec is not None and spec.loader is not None,
        "Could not construct candidate import specification.")

module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

feeder = np.load(ROOT / "feeder.npz")
H = np.asarray(feeder["H"], dtype=float)
sigma2 = np.asarray(feeder["sigma2"], dtype=float)
Q = np.asarray(feeder["Q"], dtype=float)

m, n = H.shape
n_telemetry = 45

cfg = module.MetricConfig(n_telemetry=n_telemetry)
metric = module.TrustMetric(H, sigma2, Q, cfg)

truth_candidates = [
    ROOT / "truth.eval.paper1.v4.seed002.npz",
    ROOT / "truth.eval.paper1.v5.seed002.npz",
]
truth_path = next((p for p in truth_candidates if p.exists()), None)
require(truth_path is not None, "Seed-002 truth file was not found.")

truth_data = np.load(truth_path)
require("x_true" in truth_data.files, "Truth file has no x_true array.")
truth = np.asarray(truth_data["x_true"], dtype=float).reshape(-1, n)

rng = np.random.default_rng(51002)
rcond = float(np.sqrt(np.finfo(float).eps))
records = []

def exercise_case(
    label: str,
    x: np.ndarray,
    external_count: int,
    noisy: bool,
) -> None:
    rx = np.zeros(m, dtype=bool)
    rx[n_telemetry:] = True

    if external_count:
        selected = rng.choice(
            n_telemetry,
            size=external_count,
            replace=False,
        )
        rx[selected] = True

    gamma = np.ones(m, dtype=float)
    gamma[:n_telemetry] = 0.0

    external_indices = np.flatnonzero(rx[:n_telemetry])
    gamma[external_indices] = 10.0 ** rng.uniform(
        -2.0,
        0.0,
        size=len(external_indices),
    )

    z = H @ x
    if noisy:
        z = z + rng.normal(size=m) * np.sqrt(sigma2)

    candidate, reliable = metric.estimate(z, rx, gamma)
    candidate = np.asarray(candidate, dtype=float)

    effective = rx & np.isfinite(gamma) & (gamma > 0.0)
    weighted_H = (
        H[effective]
        * np.sqrt(gamma[effective] / sigma2[effective])[:, None]
    )

    singular = np.linalg.svd(weighted_H, compute_uv=False)
    threshold = rcond * singular[0] if singular.size else np.inf
    rank = int(np.sum(singular > threshold))
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0.0
        else float("inf")
    )
    expected_reliable = rank == n and condition <= 1.0 / rcond

    candidate_norm = float(np.linalg.norm(candidate))
    truth_norm = float(np.linalg.norm(x))
    relative_error = float(
        np.linalg.norm(candidate - x) / max(truth_norm, 1e-12)
    )
    norm_ratio = candidate_norm / max(truth_norm, 1e-12)

    require(np.all(np.isfinite(candidate)),
            f"{label}: candidate is nonfinite.")
    require(bool(reliable) == expected_reliable,
            f"{label}: reliability classification mismatch.")

    if reliable and not noisy:
        require(relative_error < 1e-7,
                f"{label}: noiseless recovery error is too large.")
        require(0.5 < norm_ratio < 2.0,
                f"{label}: noiseless candidate has invalid scale.")

    if reliable and noisy:
        require(norm_ratio < 5.0,
                f"{label}: noisy candidate scale is excessive.")

    records.append({
        "label": label,
        "external_count": external_count,
        "noisy": noisy,
        "effective_rows": int(effective.sum()),
        "rank": rank,
        "condition": condition,
        "reliable": bool(reliable),
        "candidate_norm": candidate_norm,
        "truth_norm": truth_norm,
        "candidate_to_truth_norm_ratio": norm_ratio,
        "relative_recovery_error": relative_error,
    })

for row_index, external_count in enumerate((0, 21, 29, 37, 45)):
    exercise_case(
        f"noiseless_external_{external_count:02d}",
        truth[row_index],
        external_count,
        False,
    )

for row_index, external_count in enumerate((29, 37, 45), start=10):
    exercise_case(
        f"noisy_external_{external_count:02d}",
        truth[row_index],
        external_count,
        True,
    )

require(any(r["reliable"] for r in records),
        "No stress case was classified reliable.")
require(any(not r["reliable"] for r in records),
        "No rank-deficient case was identified.")
require(
    not next(
        r for r in records
        if r["label"] == "noiseless_external_00"
    )["reliable"],
    "Pseudo-only production design was incorrectly classified reliable.",
)

report = {
    "schema": "paper1.v5_1.solver.production_stress.v1",
    "performance_outcomes_inspected": False,
    "solver": "weighted_lstsq_svd",
    "rcond": rcond,
    "feeder_rows": m,
    "state_dimension": n,
    "telemetry_rows": n_telemetry,
    "truth_source": str(truth_path),
    "cases": records,
}

report_path = WORKSPACE / "v5_1_solver_production_stress.json"
report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

def source_lines(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    return text, text.splitlines(), ast.parse(text)

def named_method(tree, class_name, method_name):
    matches = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            matches.extend(
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
            )
    require(len(matches) == 1,
            f"Expected one {class_name}.{method_name}; found {len(matches)}.")
    return matches[0]

def named_function(tree, function_name):
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    require(len(matches) == 1,
            f"Expected one {function_name}; found {len(matches)}.")
    return matches[0]

def named_assignment(tree, variable_name):
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == variable_name
                for target in node.targets
            ):
                matches.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == variable_name
        ):
            matches.append(node)
    require(len(matches) == 1,
            f"Expected one assignment to {variable_name}; found {len(matches)}.")
    return matches[0]

def named_dict_assignment(function_node, variable_name):
    matches = []
    for node in ast.walk(function_node):
        target = None
        value = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value

        if (
            isinstance(target, ast.Name)
            and target.id == variable_name
            and isinstance(value, ast.Dict)
        ):
            matches.append(node)

    require(
        len(matches) == 1,
        f"Expected one {variable_name} dictionary in "
        f"{function_node.name}; found {len(matches)}.",
    )
    return matches[0]

def render_section(title, path, lines, node):
    start = int(node.lineno)
    end = int(node.end_lineno)
    body = [f"=== {title} ===", f"FILE={path}", f"LINES={start}-{end}"]
    body.extend(
        f"{number:05d}: {lines[number - 1]}"
        for number in range(start, end + 1)
    )
    body.append("")
    return body

_, trust_lines, trust_tree = source_lines(TRUST)
_, twin_lines, twin_tree = source_lines(TWIN)

trust_estimate = named_method(trust_tree, "TrustMetric", "estimate")
twin_estimate = named_method(twin_tree, "ProductionTwin", "_estimate")
twin_update = named_method(twin_tree, "ProductionTwin", "update")
step_row = named_dict_assignment(twin_update, "row")
publish = named_function(twin_tree, "publish_score")
publish_payload = named_dict_assignment(publish, "payload")
maximum_columns = named_assignment(twin_tree, "MAX_EVENT_COLUMNS")
aggregation = named_function(twin_tree, "aggregate_events")

context_lines = []
for title, path, lines, node in (
    ("TRUSTMETRIC_ESTIMATE", TRUST, trust_lines, trust_estimate),
    ("PRODUCTIONTWIN_ESTIMATE", TWIN, twin_lines, twin_estimate),
    ("STEP_ROW_DICTIONARY", TWIN, twin_lines, step_row),
    ("PUBLICATION_PAYLOAD", TWIN, twin_lines, publish_payload),
    ("MAX_EVENT_COLUMNS", TWIN, twin_lines, maximum_columns),
    ("EVENT_AGGREGATION", TWIN, twin_lines, aggregation),
):
    context_lines.extend(render_section(title, path, lines, node))

context_path = WORKSPACE / "v5_1_guard_logging_context.txt"
context_path.write_text(
    "\n".join(context_lines),
    encoding="utf-8",
)

print("PAPER1_V5_1_SOLVER_PRODUCTION_STRESS_OK")
print("CASES=", len(records))
print("RELIABLE_CASES=", sum(r["reliable"] for r in records))
print("INEXACT_CASES=", sum(not r["reliable"] for r in records))
print("PSEUDO_ONLY_RELIABLE=False")
print("MAXIMUM_RELIABLE_NORM_RATIO=", max(
    r["candidate_to_truth_norm_ratio"]
    for r in records if r["reliable"]
))
print("STRESS_REPORT_SHA256=", sha256(report_path).upper())
print("CONTEXT_SHA256=", sha256(context_path).upper())
print("LIVE_TRUST_METRIC_MODIFIED=False")
print("LIVE_TWIN_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_1_GUARD_LOGGING_CONTEXT_READY")