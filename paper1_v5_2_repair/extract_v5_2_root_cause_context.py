from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/workspace")
WORKSPACE = ROOT / "paper1_v5_2_repair"
RUN_ID = "paper1_v5_1mv_s002_bw04_oracle"
RUN_ROOT = ROOT / "runs" / RUN_ID

TRUST_PATH = ROOT / "trust_metric.py"
TWIN_PATH = ROOT / "twin_fed.py"
FEEDER_PATH = ROOT / "feeder.npz"
PATTERNS_PATH = ROOT / "patterns.npz"
CONTEXT_PATH = WORKSPACE / "root_cause_context.txt"

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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_segment(lines, node):
    start = int(node.lineno)
    end = int(node.end_lineno)
    output = [
        f"{number:05d}: {lines[number - 1]}"
        for number in range(start, end + 1)
    ]
    return start, end, output


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def add_section(output, title, content):
    output.append("")
    output.append(f"===== {title} =====")
    output.extend(content)


def relevant_method(source):
    terms = (
        "estimate(",
        "lstsq",
        "candidate_norm",
        "previous_norm",
        "jump_norm",
        "jump_limit",
        "model_increment_scale",
        "solved_exactly",
        "estimator_reliable",
        "self.samples",
        "gamma",
        "pseudo",
        "x_hat",
        "last_x",
        "hold_reason",
    )
    return any(term in source for term in terms)


require(TRUST_PATH.is_file(), f"missing {TRUST_PATH}")
require(TWIN_PATH.is_file(), f"missing {TWIN_PATH}")
require(RUN_ROOT.is_dir(), f"missing {RUN_ROOT}")

trust_hash = sha256(TRUST_PATH)
twin_hash = sha256(TWIN_PATH)

require(trust_hash == EXPECTED_TRUST, "trust source hash mismatch")
require(twin_hash == EXPECTED_TWIN, "twin source hash mismatch")

trust_text = TRUST_PATH.read_text(encoding="utf-8-sig")
twin_text = TWIN_PATH.read_text(encoding="utf-8-sig")

trust_lines = trust_text.splitlines()
twin_lines = twin_text.splitlines()

trust_tree = ast.parse(trust_text)
twin_tree = ast.parse(twin_text)

output = [
    "PAPER1_V5_2_ROOT_CAUSE_CONTEXT",
    f"RUN_ID={RUN_ID}",
    f"TRUST_SHA256={trust_hash}",
    f"TWIN_SHA256={twin_hash}",
    "SOURCE_AND_SCHEMA_INSPECTION_ONLY=True",
    "PERFORMANCE_OUTCOME_COLUMNS_READ=False",
    "PERFORMANCE_OUTCOMES_INSPECTED=False",
]

metric_config = find_class(trust_tree, "MetricConfig")
require(metric_config is not None, "MetricConfig class not found")

_, _, content = source_segment(trust_lines, metric_config)
add_section(output, "METRICCONFIG", content)

trust_class = find_class(trust_tree, "TrustMetric")
require(trust_class is not None, "TrustMetric class not found")

for node in trust_class.body:
    if not isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    ):
        continue

    start, end, content = source_segment(trust_lines, node)
    method_source = "\n".join(
        trust_lines[start - 1:end]
    )

    if node.name == "estimate" or relevant_method(method_source):
        add_section(
            output,
            f"TRUSTMETRIC.{node.name} LINES {start}-{end}",
            content,
        )

production_twin = find_class(twin_tree, "ProductionTwin")
require(production_twin is not None, "ProductionTwin class not found")

for node in production_twin.body:
    if not isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    ):
        continue

    start, end, content = source_segment(twin_lines, node)
    method_source = "\n".join(
        twin_lines[start - 1:end]
    )

    if relevant_method(method_source):
        add_section(
            output,
            f"PRODUCTIONTWIN.{node.name} LINES {start}-{end}",
            content,
        )

for node in twin_tree.body:
    if not isinstance(node, ast.FunctionDef):
        continue

    start, end, content = source_segment(twin_lines, node)
    function_source = "\n".join(
        twin_lines[start - 1:end]
    )

    if (
        node.name == "parse_telemetry_payload"
        or (
            any(
                term in function_source
                for term in (
                    "hold_factor",
                    "model_increment_scale",
                    "jump_limit",
                    "estimator_rcond",
                    "pseudo",
                    "gamma",
                )
            )
            and len(content) <= 250
        )
    ):
        add_section(
            output,
            f"TWIN_FUNCTION.{node.name} LINES {start}-{end}",
            content,
        )

occurrence_terms = (
    "ridge",
    "rcond",
    "lstsq",
    "sqrt_w",
    "model_increment_scale",
    "jump_limit",
    "hold_factor",
    "candidate_norm",
    "previous_norm",
    "estimator_reliable",
    "pseudo",
    "self.samples",
)

for label, lines in (
    ("TRUST", trust_lines),
    ("TWIN", twin_lines),
):
    selected = set()

    for index, line in enumerate(lines):
        if any(term in line for term in occurrence_terms):
            for number in range(
                max(0, index - 2),
                min(len(lines), index + 3),
            ):
                selected.add(number)

    occurrence_content = []
    previous = -2

    for number in sorted(selected):
        if number > previous + 1:
            occurrence_content.append("-----")
        occurrence_content.append(
            f"{number + 1:05d}: {lines[number]}"
        )
        previous = number

    add_section(
        output,
        f"{label}_RELEVANT_OCCURRENCES",
        occurrence_content,
    )

run_files = []

for path in sorted(RUN_ROOT.rglob("*")):
    if not path.is_file():
        continue

    relative = path.relative_to(RUN_ROOT).as_posix()

    run_files.append({
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    })

add_section(
    output,
    "RUN_FILE_INVENTORY",
    [
        json.dumps(item, sort_keys=True)
        for item in run_files
    ],
)

for path in sorted(RUN_ROOT.rglob("*.parquet")):
    parquet = pq.ParquetFile(path)
    schema = {
        "path": path.relative_to(RUN_ROOT).as_posix(),
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "columns": [
            {
                "name": field.name,
                "type": str(field.type),
            }
            for field in parquet.schema_arrow
        ],
        "values_read": False,
    }

    add_section(
        output,
        "PARQUET_SCHEMA",
        [json.dumps(schema, sort_keys=True)],
    )

for meta_path in sorted(RUN_ROOT.rglob("meta.json")):
    content = json.loads(
        meta_path.read_text(encoding="utf-8-sig")
    )

    add_section(
        output,
        "META " + meta_path.relative_to(RUN_ROOT).as_posix(),
        [
            json.dumps(
                content,
                sort_keys=True,
                separators=(",", ":"),
            )
        ],
    )

for label, path in (
    ("FEEDER", FEEDER_PATH),
    ("PATTERNS", PATTERNS_PATH),
):
    require(path.is_file(), f"missing {path}")

    arrays = []

    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = archive[key]
            arrays.append({
                "key": key,
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "values_read": False,
            })

    add_section(
        output,
        f"{label}_ARRAY_SCHEMA",
        [
            f"SHA256={sha256(path)}",
            *[
                json.dumps(item, sort_keys=True)
                for item in arrays
            ],
        ],
    )

output.extend([
    "",
    "LIVE_FILES_MODIFIED=False",
    "SIMULATION_STARTED=False",
    "PERFORMANCE_OUTCOME_COLUMNS_READ=False",
    "PERFORMANCE_OUTCOMES_INSPECTED=False",
    "FULL_CAMPAIGN_AUTHORIZED=False",
    "CALIBRATION_AUTHORIZED=False",
    "PAPER1_V5_2_ROOT_CAUSE_CONTEXT_READY",
])

CONTEXT_PATH.write_text(
    "\n".join(output) + "\n",
    encoding="utf-8",
)

print("\n".join(output))
print(f"CONTEXT_SHA256={sha256(CONTEXT_PATH)}")
