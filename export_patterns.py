#!/usr/bin/env python3
"""Export patterns.npz to the exact long-form CSV consumed by net_fed.cc."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export impairment NPZ to CSV")
    parser.add_argument("--input", default="patterns.npz")
    parser.add_argument("--output", default="patterns.csv")
    args = parser.parse_args(argv)

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source = np.load(input_path, allow_pickle=False)
    P = np.asarray(source["P"], dtype=float)
    D = np.asarray(source["D"], dtype=float)
    B = np.asarray(source["B"], dtype=float)
    if P.ndim != 2 or D.shape != P.shape or B.shape != P.shape:
        raise ValueError("P, D, and B must be same-shape 2-D arrays")
    if (
        not np.all(np.isfinite(P))
        or not np.all(np.isfinite(D))
        or not np.all(np.isfinite(B))
        or np.any((P < 0.0) | (P > 1.0))
        or np.any(D < 0.0)
        or np.any(B < 0.0)
    ):
        raise ValueError("pattern arrays contain nonfinite or out-of-range values")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event_id", "channel_id", "pi", "delta_s", "bandwidth_bps"])
        for event_id in range(P.shape[0]):
            writer.writerows(
                (
                    event_id,
                    channel_id,
                    format(P[event_id, channel_id], ".17g"),
                    format(D[event_id, channel_id], ".17g"),
                    format(B[event_id, channel_id], ".17g"),
                )
                for channel_id in range(P.shape[1])
            )
    temporary.replace(output_path)
    print(
        f"PATTERNS_EXPORT_OK rows={P.shape[0] * P.shape[1]} "
        f"events={P.shape[0]} channels={P.shape[1]} path={output_path.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
