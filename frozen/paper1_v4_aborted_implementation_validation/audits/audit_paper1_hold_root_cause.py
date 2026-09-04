from pathlib import Path
import math

import numpy as np
import pandas as pd

root = Path("/workspace")
seeds = range(2, 6)

levels = [
    "bw00_floor",
    "bw01_10kbps",
    "bw02_100kbps",
    "bw03_1mbps",
    "bw04_oracle",
]

n = 491
hold_factor = 50.0
initial_scale_floor = math.sqrt(n) * 0.01
initial_candidate_limit = hold_factor * initial_scale_floor

print("PAPER1_HOLD_ROOT_CAUSE_AUDIT")
print("STATE_DIMENSION=", n)
print("HOLD_FACTOR=", hold_factor)
print("INITIAL_SCALE_FLOOR=", initial_scale_floor)
print("INITIAL_CANDIDATE_NORM_LIMIT=", initial_candidate_limit)
print()

print(
    "seed level held_mean solve_exact_mean "
    "held_and_exact_fraction held_and_inexact_fraction"
)

for seed in seeds:
    for level in levels:
        run_id = f"paper1_s{seed:03d}_{level}"

        step_path = (
            root / "runs" / run_id /
            "twin" / "scores.parquet"
        )

        frame = pd.read_parquet(step_path)

        held = pd.to_numeric(
            frame["held"],
            errors="coerce",
        ).astype(float)

        if "solve_exact" not in frame.columns:
            print(
                f"{run_id}: solve_exact is unavailable; "
                f"related columns="
                f"{[c for c in frame.columns if 'solve' in c.lower()]}"
            )
            continue

        exact = pd.to_numeric(
            frame["solve_exact"],
            errors="coerce",
        ).astype(float)

        held_bool = held > 0.5
        exact_bool = exact > 0.5

        print(
            seed,
            level,
            f"{held.mean():.9f}",
            f"{exact.mean():.9f}",
            f"{np.mean(held_bool & exact_bool):.9f}",
            f"{np.mean(held_bool & ~exact_bool):.9f}",
        )

print()
print("TRUTH_STATE_NORM_AUDIT")

for seed in seeds:
    truth_path = (
        root /
        f"truth.eval.paper1.v4.seed{seed:03d}.npz"
    )

    data = np.load(truth_path)

    state_candidates = []

    for key in data.files:
        array = np.asarray(data[key])

        if (
            array.ndim >= 2
            and array.shape[-1] == n
            and np.issubdtype(array.dtype, np.number)
        ):
            state_candidates.append((key, array))

    if not state_candidates:
        print(
            f"seed={seed} no 491-state array identified; "
            f"keys={data.files}"
        )
        continue

    for key, array in state_candidates:
        matrix = array.reshape(-1, n).astype(float)
        norms = np.linalg.norm(matrix, axis=1)

        print(
            f"seed={seed}",
            f"key={key}",
            f"rows={len(norms)}",
            f"initial_norm={norms[0]:.9f}",
            f"minimum_norm={np.min(norms):.9f}",
            f"mean_norm={np.mean(norms):.9f}",
            f"maximum_norm={np.max(norms):.9f}",
            f"exceeds_initial_limit_fraction="
            f"{np.mean(norms > initial_candidate_limit):.9f}",
        )

print()
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_HOLD_ROOT_CAUSE_AUDIT_OK")