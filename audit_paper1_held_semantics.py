from pathlib import Path

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

print("HELD_SEMANTICS_DATA_AUDIT")
print(
    "seed level step_rows event_rows "
    "step_held_mean event_held_mean "
    "event_held_zero_fraction event_held_one_fraction "
    "aggregation_error observable_step_mean"
)

for seed in seeds:
    for level in levels:
        run_id = f"paper1_s{seed:03d}_{level}"

        step_path = (
            root / "runs" / run_id /
            "twin" / "scores.parquet"
        )

        event_path = (
            root / "runs" / run_id /
            "twin" / "scores_events.parquet"
        )

        steps = pd.read_parquet(step_path)
        events = pd.read_parquet(event_path)

        if "held" not in steps.columns:
            raise ValueError(
                f"{run_id}: held is absent from step results"
            )

        if "held" not in events.columns:
            raise ValueError(
                f"{run_id}: held is absent from event results"
            )

        if "event_id" not in steps.columns:
            raise ValueError(
                f"{run_id}: event_id absent from step results"
            )

        if "event_id" not in events.columns:
            raise ValueError(
                f"{run_id}: event_id absent from event results"
            )

        step_held = pd.to_numeric(
            steps["held"],
            errors="coerce",
        ).astype(float)

        event_held = pd.to_numeric(
            events["held"],
            errors="coerce",
        ).astype(float)

        aggregated = (
            steps.assign(
                held_numeric=step_held
            )
            .groupby("event_id")["held_numeric"]
            .mean()
        )

        aligned = (
            events[["event_id"]]
            .assign(event_held=event_held)
            .merge(
                aggregated.rename("recomputed_held"),
                left_on="event_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
        )

        aggregation_error = float(
            np.nanmax(
                np.abs(
                    aligned["event_held"].to_numpy(float)
                    - aligned["recomputed_held"].to_numpy(float)
                )
            )
        )

        zero_fraction = float(
            np.mean(np.isclose(event_held, 0.0))
        )

        one_fraction = float(
            np.mean(np.isclose(event_held, 1.0))
        )

        if "observable" in steps.columns:
            observable = pd.to_numeric(
                steps["observable"],
                errors="coerce",
            ).astype(float)

            observable_mean = float(
                observable.mean()
            )
        else:
            observable_mean = np.nan

        print(
            seed,
            level,
            len(steps),
            len(events),
            f"{step_held.mean():.9f}",
            f"{event_held.mean():.9f}",
            f"{zero_fraction:.9f}",
            f"{one_fraction:.9f}",
            f"{aggregation_error:.3e}",
            f"{observable_mean:.9f}",
        )

print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("HELD_SEMANTICS_DATA_AUDIT_OK")