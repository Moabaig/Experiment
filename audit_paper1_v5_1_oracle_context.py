from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/workspace")
RUN_ID = "paper1_v5_1mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID

EXPECTED_TRUST = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)

EXPECTED_TWIN = (
    "39e6729af233032ab9c58851c9682252"
    "f02d36eed739eb2ec769e165659da34c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def flatten_scalars(value, prefix=""):
    output = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_scalars(child, child_prefix))

    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value

    elif isinstance(value, list):
        if (
            len(value) <= 10
            and all(
                isinstance(item, (str, int, float, bool))
                or item is None
                for item in value
            )
        ):
            output[prefix] = value
        else:
            output[f"{prefix}.__length__"] = len(value)

    return output


def selected_context(value):
    flat = flatten_scalars(value)

    pattern = re.compile(
        r"(?i)("
        r"schema|run_id|seed|step|event|status|complete|"
        r"bandwidth|sha256|version|solver|normal_equation|"
        r"guard|telemetry_source|factor_design|authorization|"
        r"performance_outcomes"
        r")"
    )

    return {
        key: flat[key]
        for key in sorted(flat)
        if pattern.search(key)
    }


required_files = [
    ROOT / "trust_metric.py",
    ROOT / "twin_fed.py",
    RUN_ROOT / "power" / "meta.json",
    RUN_ROOT / "net" / "meta.json",
    RUN_ROOT / "twin" / "meta.json",
    RUN_ROOT / "oracle" / "meta.json",
    RUN_ROOT / "twin" / "scores.parquet",
    RUN_ROOT / "twin" / "scores_events.parquet",
    RUN_ROOT / "cell_record.paper1.v5_1.mechanical.json",
    RUN_ROOT / "CELL_OUTPUT_SHA256SUMS.csv",
]

missing = [str(path) for path in required_files if not path.is_file()]

if missing:
    raise RuntimeError(f"required files are missing: {missing}")

if sha256(ROOT / "trust_metric.py") != EXPECTED_TRUST:
    raise RuntimeError("live trust_metric.py hash changed")

if sha256(ROOT / "twin_fed.py") != EXPECTED_TWIN:
    raise RuntimeError("live twin_fed.py hash changed")

print("PAPER1_V5_1_ORACLE_AUDIT_CONTEXT")
print("RUN_ID=", RUN_ID)
print("IMPLEMENTATION_MODIFIED=False")
print("SIMULATION_STARTED=False")
print("PERFORMANCE_OUTCOME_VALUES_INSPECTED=False")
print()

record_path = (
    RUN_ROOT /
    "cell_record.paper1.v5_1.mechanical.json"
)

record = read_json(record_path)

print("=== CELL RECORD ===")
print("PATH=", record_path)
print("SHA256=", sha256(record_path))
print(
    "CONTENT=",
    json.dumps(record, sort_keys=True, separators=(",", ":")),
)
print()

for component in ("power", "net", "twin", "oracle"):
    meta_path = RUN_ROOT / component / "meta.json"
    meta = read_json(meta_path)

    print(f"=== {component.upper()} META ===")
    print("PATH=", meta_path)
    print("SHA256=", sha256(meta_path))
    print("TOP_LEVEL_KEYS=", sorted(meta.keys()))
    print(
        "SELECTED_CONTEXT=",
        json.dumps(
            selected_context(meta),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    print()

step_path = RUN_ROOT / "twin" / "scores.parquet"
event_path = RUN_ROOT / "twin" / "scores_events.parquet"

for label, path in (
    ("STEP", step_path),
    ("EVENT", event_path),
):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow

    print(f"=== {label} PARQUET SCHEMA ===")
    print("PATH=", path)
    print("SHA256=", sha256(path))
    print("ROWS=", parquet.metadata.num_rows)
    print("ROW_GROUPS=", parquet.metadata.num_row_groups)
    print("COLUMN_COUNT=", len(schema.names))
    print("COLUMNS=", json.dumps(schema.names))
    print(
        "TYPES=",
        json.dumps(
            {
                field.name: str(field.type)
                for field in schema
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    print()

manifest_path = RUN_ROOT / "CELL_OUTPUT_SHA256SUMS.csv"

with manifest_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    reader = csv.DictReader(handle)
    manifest_rows = list(reader)
    manifest_fields = reader.fieldnames

print("=== CELL MANIFEST STRUCTURE ===")
print("PATH=", manifest_path)
print("SHA256=", sha256(manifest_path))
print("FIELDS=", manifest_fields)
print("ROWS=", len(manifest_rows))
print()

truth_name = record.get("truth_file")

if not isinstance(truth_name, str) or not truth_name:
    raise RuntimeError("cell record does not identify truth_file")

truth_path = ROOT / truth_name

if not truth_path.is_file():
    raise RuntimeError(f"truth file is missing: {truth_path}")

with np.load(truth_path, allow_pickle=False) as truth:
    truth_structure = {
        key: {
            "shape": list(np.asarray(truth[key]).shape),
            "dtype": str(np.asarray(truth[key]).dtype),
        }
        for key in truth.files
    }

print("=== TRUTH STRUCTURE ===")
print("PATH=", truth_path)
print("SHA256=", sha256(truth_path))
print(
    "ARRAYS=",
    json.dumps(
        truth_structure,
        sort_keys=True,
        separators=(",", ":"),
    ),
)
print()

print("SCHEMA_VALUES_INSPECTED=False")
print("PERFORMANCE_OUTCOME_VALUES_INSPECTED=False")
print("IMPLEMENTATION_MODIFIED=False")
print("SIMULATION_STARTED=False")
print("PAPER1_V5_1_ORACLE_AUDIT_CONTEXT_OK")