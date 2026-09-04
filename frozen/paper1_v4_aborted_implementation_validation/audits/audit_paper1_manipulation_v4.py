from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

root = Path("/workspace")
output = root / "paper1_manipulation_audit_v4"
output.mkdir(exist_ok=True)

seeds = range(2, 7)

levels = [
    "bw00_floor",
    "bw01_10kbps",
    "bw02_100kbps",
    "bw03_1mbps",
    "bw04_oracle",
]

core_columns = [
    "b1",
    "b2",
    "n_rx",
    "u_lmax",
    "u_trace",
    "observable",
    "held",
]

frames = {}
summary_rows = []

network_pattern = re.compile(
    r"net_fed complete:\s*"
    r"received=(\d+)\s+"
    r"delivered=(\d+)\s+"
    r"dropped_random=(\d+)\s+"
    r"dropped_starved=(\d+)\s+"
    r"dropped_queue=(\d+)"
)

twin_pattern = re.compile(
    r"twin_fed complete:\s*"
    r"steps=(\d+)\s+"
    r"events=(\d+)\s+"
    r"accepted=(\d+)\s+"
    r"stale=(\d+)\s+"
    r"malformed=(\d+)"
)

def numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }

for seed in seeds:
    for level in levels:
        run_id = f"paper1_s{seed:03d}_{level}"
        event_path = (
            root
            / "runs"
            / run_id
            / "twin"
            / "scores_events.parquet"
        )

        if not event_path.exists():
            raise FileNotFoundError(
                f"Missing event results: {event_path}"
            )

        frame = pd.read_parquet(event_path)

        if len(frame) != 1100:
            raise ValueError(
                f"{run_id}: expected 1100 events, found {len(frame)}"
            )

        if "event_id" not in frame.columns:
            frame = frame.copy()
            frame["event_id"] = np.arange(len(frame))

        if frame["event_id"].duplicated().any():
            raise ValueError(
                f"{run_id}: duplicate event identifiers"
            )

        frames[(seed, level)] = frame.copy()

        row = {
            "seed": seed,
            "level": level,
            "run_id": run_id,
            "events": len(frame),
        }

        for column in core_columns:
            if column in frame.columns:
                statistics = numeric_summary(frame[column])
                for name, value in statistics.items():
                    row[f"{column}_{name}"] = value

        log_path = root / f"run_{run_id}.log"

        if log_path.exists():
            text = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            network_matches = list(
                network_pattern.finditer(text)
            )
            twin_matches = list(
                twin_pattern.finditer(text)
            )

            if network_matches:
                values = list(
                    map(int, network_matches[-1].groups())
                )
                (
                    row["received"],
                    row["delivered"],
                    row["dropped_random"],
                    row["dropped_starved"],
                    row["dropped_queue"],
                ) = values

            if twin_matches:
                values = list(
                    map(int, twin_matches[-1].groups())
                )
                (
                    row["steps"],
                    row["logged_events"],
                    row["accepted"],
                    row["stale"],
                    row["malformed"],
                ) = values

        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    output / "cell_manipulation_summary.csv",
    index=False,
)

comparison_rows = []

reference_level = "bw01_10kbps"

for seed in seeds:
    reference = frames[
        (seed, reference_level)
    ].sort_values("event_id")

    for target_level in [
        "bw02_100kbps",
        "bw03_1mbps",
        "bw04_oracle",
    ]:
        target = frames[
            (seed, target_level)
        ].sort_values("event_id")

        merged = reference.merge(
            target,
            on="event_id",
            how="inner",
            suffixes=("_reference", "_target"),
            validate="one_to_one",
        )

        if len(merged) != 1100:
            raise ValueError(
                f"Seed {seed}, {target_level}: "
                f"event alignment produced {len(merged)} rows"
            )

        result = {
            "seed": seed,
            "reference": reference_level,
            "target": target_level,
            "events": len(merged),
        }

        changed_masks = []

        for column in core_columns:
            reference_column = f"{column}_reference"
            target_column = f"{column}_target"

            if (
                reference_column not in merged.columns
                or target_column not in merged.columns
            ):
                continue

            left = pd.to_numeric(
                merged[reference_column],
                errors="coerce",
            ).to_numpy(float)

            right = pd.to_numeric(
                merged[target_column],
                errors="coerce",
            ).to_numpy(float)

            equal = np.isclose(
                left,
                right,
                rtol=1e-12,
                atol=1e-15,
                equal_nan=True,
            )

            changed = ~equal
            changed_masks.append(changed)

            finite = np.isfinite(left) & np.isfinite(right)

            if finite.any():
                absolute_difference = np.abs(
                    left[finite] - right[finite]
                )
                maximum_difference = float(
                    np.max(absolute_difference)
                )
                mean_difference = float(
                    np.mean(absolute_difference)
                )
            else:
                maximum_difference = np.nan
                mean_difference = np.nan

            result[
                f"{column}_changed_fraction"
            ] = float(np.mean(changed))

            result[
                f"{column}_maximum_absolute_difference"
            ] = maximum_difference

            result[
                f"{column}_mean_absolute_difference"
            ] = mean_difference

        if changed_masks:
            any_change = np.logical_or.reduce(changed_masks)
            result[
                "any_diagnostic_changed_fraction"
            ] = float(np.mean(any_change))
        else:
            result[
                "any_diagnostic_changed_fraction"
            ] = np.nan

        comparison_rows.append(result)

comparisons = pd.DataFrame(comparison_rows)
comparisons.to_csv(
    output / "nonfloor_pairwise_comparisons.csv",
    index=False,
)

summary_records = json.loads(
    summary.to_json(orient="records")
)
comparison_records = json.loads(
    comparisons.to_json(orient="records")
)

report = {
    "schema": "paper1.bandwidth.manipulation.audit.v1",
    "seeds": list(seeds),
    "levels": levels,
    "performance_outcomes_inspected": False,
    "cell_summary": summary_records,
    "nonfloor_pairwise_comparisons": comparison_records,
}

report_path = output / "manipulation_audit.json"
report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

maximum_changed_fraction = float(
    comparisons[
        "any_diagnostic_changed_fraction"
    ].max()
)

print("PAPER1_MANIPULATION_AUDIT_OK")
print("CELLS_AUDITED=", len(summary))
print("EXPECTED_CELLS=", len(list(seeds)) * len(levels))
print("PAIRWISE_COMPARISONS=", len(comparisons))
print(
    "MAXIMUM_NONFLOOR_DIAGNOSTIC_CHANGED_FRACTION=",
    maximum_changed_fraction,
)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print()
print("NONFLOOR_PAIRWISE_RESULTS")
print(comparisons.to_string(index=False))
print()
print("OUTPUT_DIRECTORY=", output)
