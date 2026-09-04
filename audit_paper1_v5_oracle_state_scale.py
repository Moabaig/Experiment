from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root = Path("/workspace")
run_id = "paper1_v5mv_s002_bw04_oracle"

scores_path = (
    root / "runs" / run_id /
    "twin" / "scores.parquet"
)
truth_path = (
    root /
    "truth.eval.paper1.v4.seed002.npz"
)
source_path = root / "twin_fed.py"

report_path = (
    root /
    "paper1_v5_oracle_state_scale_audit.json"
)
context_path = (
    root /
    "paper1_v5_estimate_method_context.txt"
)

columns = [
    "step_index",
    "event_id",
    "hold_reason",
    "held",
    "solve_exact",
    "state_update_accepted_step",
    "bootstrap_accept_step",
    "candidate_norm",
    "previous_norm",
    "jump_norm",
    "jump_limit",
    "external_received_count",
    "external_support_fraction",
    "pseudo_only_step",
]

schema = set(
    pq.ParquetFile(scores_path).schema_arrow.names
)

missing = sorted(set(columns) - schema)

if missing:
    raise RuntimeError(
        f"missing diagnostic columns: {missing}"
    )

steps = pd.read_parquet(
    scores_path,
    columns=columns,
)

if len(steps) != 13200:
    raise RuntimeError(
        f"unexpected step count: {len(steps)}"
    )

def numeric(name):
    return pd.to_numeric(
        steps[name],
        errors="coerce",
    ).to_numpy(dtype=float)

def boolean(name):
    series = steps[name]

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isin(values, [0.0, 1.0]))
    ):
        raise RuntimeError(
            f"{name} is not Boolean"
        )

    return values > 0.5

def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }

    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }

candidate = numeric("candidate_norm")
previous = numeric("previous_norm")
jump = numeric("jump_norm")
limit = numeric("jump_limit")
external_count = numeric(
    "external_received_count"
)
external_fraction = numeric(
    "external_support_fraction"
)
step_index = numeric("step_index").astype(int)
event_id = numeric("event_id").astype(int)

accepted = boolean(
    "state_update_accepted_step"
)
bootstrap = boolean(
    "bootstrap_accept_step"
)
held = boolean("held")
exact = boolean("solve_exact")
pseudo_only = boolean("pseudo_only_step")

reasons = (
    steps["hold_reason"]
    .fillna("<missing>")
    .astype(str)
    .to_numpy()
)

jump_guard = reasons == "jump_guard"
solve_inexact = reasons == "solve_inexact"

with np.load(truth_path) as truth_data:
    if "x_true" not in truth_data.files:
        raise RuntimeError(
            "truth archive lacks x_true"
        )

    truth = np.asarray(
        truth_data["x_true"],
        dtype=float,
    )

if truth.shape != (13200, 491):
    raise RuntimeError(
        f"unexpected x_true shape: {truth.shape}"
    )

truth_norm = np.linalg.norm(
    truth,
    axis=1,
)

finite_candidate_ratio = (
    np.isfinite(candidate)
    & np.isfinite(truth_norm)
    & (truth_norm > 0)
)

candidate_truth_ratio = np.full(
    len(steps),
    np.nan,
    dtype=float,
)

candidate_truth_ratio[finite_candidate_ratio] = (
    candidate[finite_candidate_ratio]
    / truth_norm[finite_candidate_ratio]
)

finite_jump_ratio = (
    jump_guard
    & np.isfinite(jump)
    & np.isfinite(limit)
    & (limit > 0)
)

jump_limit_ratio = (
    jump[finite_jump_ratio]
    / limit[finite_jump_ratio]
)

# Identify consecutive jump-guard clusters.
positions = np.flatnonzero(jump_guard)
clusters = []

if len(positions):
    start = positions[0]
    end = positions[0]

    for position in positions[1:]:
        if position == end + 1:
            end = position
        else:
            clusters.append((start, end))
            start = position
            end = position

    clusters.append((start, end))

cluster_records = []

for start, end in clusters:
    mask = np.arange(len(steps))
    mask = (mask >= start) & (mask <= end)

    ratios = np.full(
        int(np.sum(mask)),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(jump[mask])
        & np.isfinite(limit[mask])
        & (limit[mask] > 0)
    )

    ratios[valid] = (
        jump[mask][valid]
        / limit[mask][valid]
    )

    cluster_records.append({
        "start_step_index": int(
            step_index[start]
        ),
        "end_step_index": int(
            step_index[end]
        ),
        "length": int(end - start + 1),
        "start_event_id": int(
            event_id[start]
        ),
        "end_event_id": int(
            event_id[end]
        ),
        "candidate_norm_max": float(
            np.nanmax(candidate[mask])
        ),
        "previous_norm_max": float(
            np.nanmax(previous[mask])
        ),
        "jump_norm_max": float(
            np.nanmax(jump[mask])
        ),
        "jump_limit_min": float(
            np.nanmin(limit[mask])
        ),
        "jump_to_limit_ratio_max": float(
            np.nanmax(ratios)
        ),
        "external_received_min": float(
            np.nanmin(external_count[mask])
        ),
        "external_support_fraction_min": float(
            np.nanmin(external_fraction[mask])
        ),
    })

cluster_records.sort(
    key=lambda item: item["length"],
    reverse=True,
)

bootstrap_positions = np.flatnonzero(bootstrap)

if len(bootstrap_positions) != 1:
    raise RuntimeError(
        "expected exactly one bootstrap step"
    )

bootstrap_position = int(
    bootstrap_positions[0]
)

# Extract the exact installed _estimate implementation.
source_text = source_path.read_text(
    encoding="utf-8-sig"
)
source_lines = source_text.splitlines()
tree = ast.parse(source_text)

estimate_node = None

for node in tree.body:
    if (
        isinstance(node, ast.ClassDef)
        and node.name == "ProductionTwin"
    ):
        for child in node.body:
            if (
                isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and child.name == "_estimate"
            ):
                estimate_node = child
                break

if estimate_node is None:
    raise RuntimeError(
        "ProductionTwin._estimate was not found"
    )

start_line = max(1, estimate_node.lineno - 5)
end_line = min(
    len(source_lines),
    estimate_node.end_lineno + 5,
)

context = "\n".join(
    f"{number:05d}: {source_lines[number - 1]}"
    for number in range(start_line, end_line + 1)
)

context_path.write_text(
    context + "\n",
    encoding="utf-8",
)

source_hash = hashlib.sha256(
    source_path.read_bytes()
).hexdigest()

report = {
    "schema":
        "paper1.v5.oracle.state.scale.audit.v1",
    "run_id": run_id,
    "estimator_sha256": source_hash,
    "performance_outcomes_inspected": False,
    "steps": int(len(steps)),
    "accepted_count": int(np.sum(accepted)),
    "held_count": int(np.sum(held)),
    "solve_exact_count": int(np.sum(exact)),
    "solve_inexact_count": int(
        np.sum(solve_inexact)
    ),
    "jump_guard_count": int(
        np.sum(jump_guard)
    ),
    "pseudo_only_count": int(
        np.sum(pseudo_only)
    ),
    "truth_norm_statistics":
        stats(truth_norm),
    "candidate_norm_all_statistics":
        stats(candidate),
    "candidate_norm_accepted_statistics":
        stats(candidate[accepted]),
    "candidate_norm_jump_guard_statistics":
        stats(candidate[jump_guard]),
    "previous_norm_jump_guard_statistics":
        stats(previous[jump_guard]),
    "candidate_to_truth_ratio_all_statistics":
        stats(candidate_truth_ratio),
    "candidate_to_truth_ratio_accepted_statistics":
        stats(candidate_truth_ratio[accepted]),
    "candidate_to_truth_ratio_jump_guard_statistics":
        stats(candidate_truth_ratio[jump_guard]),
    "jump_norm_guard_statistics":
        stats(jump[jump_guard]),
    "jump_limit_guard_statistics":
        stats(limit[jump_guard]),
    "jump_to_limit_ratio_statistics":
        stats(jump_limit_ratio),
    "bootstrap": {
        "step_index": int(
            step_index[bootstrap_position]
        ),
        "event_id": int(
            event_id[bootstrap_position]
        ),
        "candidate_norm": float(
            candidate[bootstrap_position]
        ),
        "truth_norm": float(
            truth_norm[bootstrap_position]
        ),
        "candidate_to_truth_ratio": float(
            candidate_truth_ratio[
                bootstrap_position
            ]
        ),
        "external_received_count": float(
            external_count[bootstrap_position]
        ),
        "external_support_fraction": float(
            external_fraction[
                bootstrap_position
            ]
        ),
    },
    "jump_guard_cluster_count": int(
        len(cluster_records)
    ),
    "longest_jump_guard_clusters":
        cluster_records[:10],
    "estimate_method_lines": {
        "start": int(start_line),
        "end": int(end_line),
    },
    "status": "INVESTIGATION_REQUIRED",
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_ORACLE_STATE_SCALE_AUDIT_OK")
print(
    "TRUTH_NORM_STATISTICS=",
    json.dumps(
        report["truth_norm_statistics"],
        sort_keys=True,
    ),
)
print(
    "ACCEPTED_CANDIDATE_NORM_STATISTICS=",
    json.dumps(
        report[
            "candidate_norm_accepted_statistics"
        ],
        sort_keys=True,
    ),
)
print(
    "ACCEPTED_CANDIDATE_TO_TRUTH_RATIO=",
    json.dumps(
        report[
            "candidate_to_truth_ratio_accepted_statistics"
        ],
        sort_keys=True,
    ),
)
print(
    "BOOTSTRAP=",
    json.dumps(
        report["bootstrap"],
        sort_keys=True,
    ),
)
print(
    "JUMP_GUARD_CLUSTER_COUNT=",
    report["jump_guard_cluster_count"],
)
print(
    "LONGEST_JUMP_GUARD_CLUSTERS=",
    json.dumps(
        report["longest_jump_guard_clusters"],
        sort_keys=True,
    ),
)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print()
print("ESTIMATE_METHOD_CONTEXT")
print(context)
print()
print("REPORT=", report_path)
print("CONTEXT=", context_path)