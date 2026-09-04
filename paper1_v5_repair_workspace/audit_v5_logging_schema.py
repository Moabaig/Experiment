from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
report_path = workspace / "paper1_v5_logging_schema_audit.json"
context_path = workspace / "paper1_v5_logging_schema_context.txt"

text = source_path.read_text(encoding="utf-8-sig")
lines = text.splitlines()
tree = ast.parse(text, filename=str(source_path))

source_sha256 = hashlib.sha256(
    source_path.read_bytes()
).hexdigest()

parent = {}

for node in ast.walk(tree):
    for child in ast.iter_child_nodes(node):
        parent[child] = node


def enclosing_function(node):
    current = node

    while current in parent:
        current = parent[current]

        if isinstance(
            current,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return current.name

    return None


def string_dict_keys(node):
    keys = []

    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)

    return keys


diagnostic_dicts = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
        continue

    keys = string_dict_keys(node)

    if "held" in keys or "solve_exact" in keys:
        diagnostic_dicts.append(
            {
                "line_start": node.lineno,
                "line_end": getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                ),
                "function": enclosing_function(node),
                "keys": keys,
            }
        )

term_patterns = {
    "held_key": re.compile(r"""["']held["']\s*:"""),
    "solve_exact_key": re.compile(
        r"""["']solve_exact["']\s*:"""
    ),
    "rows_append": re.compile(r"\brows?\.append\s*\("),
    "dataframe": re.compile(r"\bDataFrame\s*\("),
    "to_parquet": re.compile(r"\.to_parquet\s*\("),
    "groupby": re.compile(r"\.groupby\s*\("),
    "aggregation": re.compile(r"\.agg(?:regate)?\s*\("),
    "event_output": re.compile(
        r"scores_events|event_rows|events_frame",
        re.IGNORECASE,
    ),
}

occurrences = {}

for name, pattern in term_patterns.items():
    occurrences[name] = [
        number
        for number, line in enumerate(lines, start=1)
        if pattern.search(line)
    ]

if not diagnostic_dicts:
    raise RuntimeError(
        "No output dictionary containing held or solve_exact was found."
    )

if not occurrences["to_parquet"]:
    raise RuntimeError("No Parquet output call was found.")

selected_lines = set()

for item in diagnostic_dicts:
    selected_lines.add(item["line_start"])
    selected_lines.add(item["line_end"])

for key in (
    "held_key",
    "solve_exact_key",
    "rows_append",
    "to_parquet",
    "groupby",
    "aggregation",
    "event_output",
):
    selected_lines.update(occurrences[key])

windows = []

for centre in sorted(selected_lines):
    start = max(1, centre - 15)
    end = min(len(lines), centre + 25)

    if windows and start <= windows[-1][1] + 1:
        windows[-1] = (
            windows[-1][0],
            max(windows[-1][1], end),
        )
    else:
        windows.append((start, end))

context_sections = []

for start, end in windows:
    context_sections.append(
        f"\n===== SOURCE LINES {start}-{end} =====\n"
    )

    for line_number in range(start, end + 1):
        context_sections.append(
            f"{line_number:05d}: "
            f"{lines[line_number - 1]}\n"
        )

context_path.write_text(
    "".join(context_sections),
    encoding="utf-8",
)

step_path = Path(
    "/workspace/runs/"
    "paper1_s002_bw01_10kbps/"
    "twin/scores.parquet"
)

event_path = Path(
    "/workspace/runs/"
    "paper1_s002_bw01_10kbps/"
    "twin/scores_events.parquet"
)

if not step_path.exists():
    raise FileNotFoundError(
        f"Schema-only reference file is missing: {step_path}"
    )

if not event_path.exists():
    raise FileNotFoundError(
        f"Schema-only reference file is missing: {event_path}"
    )

step_schema = pq.ParquetFile(step_path).schema_arrow
event_schema = pq.ParquetFile(event_path).schema_arrow

step_columns = [
    {
        "name": field.name,
        "type": str(field.type),
        "nullable": field.nullable,
    }
    for field in step_schema
]

event_columns = [
    {
        "name": field.name,
        "type": str(field.type),
        "nullable": field.nullable,
    }
    for field in event_schema
]

step_names = [item["name"] for item in step_columns]
event_names = [item["name"] for item in event_columns]

for required in ("held", "solve_exact"):
    if required not in step_names:
        raise RuntimeError(
            f"Existing step schema is missing {required}."
        )

report = {
    "schema": "paper1.v5.logging.schema.audit.v1",
    "source": str(source_path),
    "source_sha256": source_sha256,
    "diagnostic_dicts": diagnostic_dicts,
    "source_occurrences": occurrences,
    "step_schema_reference": str(step_path),
    "event_schema_reference": str(event_path),
    "step_columns": step_columns,
    "event_columns": event_columns,
    "step_column_count": len(step_columns),
    "event_column_count": len(event_columns),
    "performance_outcomes_inspected": False,
    "schema_only_inspection": True,
    "context_file": str(context_path),
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_SCHEMA_AUDIT_OK")
print("SOURCE_SHA256=", source_sha256)
print("DIAGNOSTIC_DICT_COUNT=", len(diagnostic_dicts))
print("STEP_COLUMN_COUNT=", len(step_columns))
print("EVENT_COLUMN_COUNT=", len(event_columns))
print("STEP_HAS_HELD=", "held" in step_names)
print("STEP_HAS_SOLVE_EXACT=", "solve_exact" in step_names)
print("EVENT_HAS_HELD=", "held" in event_names)
print(
    "EVENT_HAS_SOLVE_EXACT=",
    "solve_exact" in event_names,
)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("SCHEMA_ONLY_INSPECTION=True")
print()
print("DIAGNOSTIC_DICTIONARIES")

for item in diagnostic_dicts:
    print(
        "function=",
        item["function"],
        "lines=",
        f"{item['line_start']}-{item['line_end']}",
        "keys=",
        ",".join(item["keys"]),
    )

print()
print(context_path.read_text(encoding="utf-8"))