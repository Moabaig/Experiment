from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
report_path = workspace / "paper1_v5_logging_patch_v2_report.json"

original = source_path.read_bytes()
before_hash = hashlib.sha256(original).hexdigest()
text = original.decode("utf-8-sig")
lines = text.splitlines()
tree = ast.parse(text, filename=str(source_path))

parents = {}

for node in ast.walk(tree):
    for child in ast.iter_child_nodes(node):
        parents[child] = node


def enclosing(node, node_type):
    current = node

    while current in parents:
        current = parents[current]

        if isinstance(current, node_type):
            return current

    return None


def mapping(node):
    output = {}

    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if key.value in output:
                raise RuntimeError(f"Duplicate key: {key.value}")

            output[key.value] = value

    return output


step_candidates = []
publish_candidates = []
aggregation_candidates = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
        continue

    fields = mapping(node)

    if "held" not in fields or "solve_exact" not in fields:
        continue

    function = enclosing(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    )

    class_node = enclosing(node, ast.ClassDef)

    function_name = function.name if function else None
    class_name = class_node.name if class_node else None

    held_value = fields["held"]

    if (
        isinstance(held_value, ast.Constant)
        and isinstance(held_value.value, str)
    ):
        aggregation_candidates.append(node)
    elif class_name == "ProductionTwin" and function_name == "update":
        step_candidates.append(node)
    elif class_name is None and function_name == "publish_score":
        publish_candidates.append(node)

if len(step_candidates) != 1:
    raise RuntimeError(
        f"Expected one ProductionTwin.update row; "
        f"found {len(step_candidates)}."
    )

if len(publish_candidates) != 1:
    raise RuntimeError(
        f"Expected one publish_score payload; "
        f"found {len(publish_candidates)}."
    )

if len(aggregation_candidates) != 1:
    raise RuntimeError(
        f"Expected one event aggregation dictionary; "
        f"found {len(aggregation_candidates)}."
    )

step_dict = step_candidates[0]
publish_dict = publish_candidates[0]
aggregation_dict = aggregation_candidates[0]

step_fields = [
    ("hold_reason", "self.last_hold_reason"),
    ("state_update_accepted_step", "not bool(held)"),
    (
        "bootstrap_accept_step",
        'self.last_hold_reason == "bootstrap_accept"',
    ),
    (
        "solve_inexact_hold_step",
        'self.last_hold_reason == "solve_inexact"',
    ),
    (
        "nonfinite_candidate_hold_step",
        'self.last_hold_reason == "nonfinite_candidate"',
    ),
    (
        "jump_guard_hold_step",
        'self.last_hold_reason == "jump_guard"',
    ),
    ("candidate_norm", "self.last_candidate_norm"),
    ("previous_norm", "self.last_previous_norm"),
    ("jump_norm", "self.last_jump_norm"),
    ("jump_limit", "self.last_jump_limit"),
    (
        "external_received_count",
        "self.last_external_received_count",
    ),
    ("external_total", "self.last_external_total"),
    (
        "external_support_fraction",
        "self.last_external_support_fraction",
    ),
    (
        "pseudo_received_count",
        "self.last_pseudo_received_count",
    ),
    ("pseudo_only_step", "self.last_pseudo_only"),
    (
        "external_support_present_step",
        "self.last_external_received_count > 0",
    ),
    (
        "no_received_measurements_step",
        (
            "self.last_external_received_count == 0 "
            "and self.last_pseudo_received_count == 0"
        ),
    ),
]

publish_fields = [
    (name, f'row[{name!r}]')
    for name, _ in step_fields
]

event_aggregations = [
    ("state_update_accepted_step", "'mean'"),
    ("bootstrap_accept_step", "'mean'"),
    ("solve_inexact_hold_step", "'mean'"),
    ("nonfinite_candidate_hold_step", "'mean'"),
    ("jump_guard_hold_step", "'mean'"),
    ("pseudo_only_step", "'mean'"),
    ("external_support_present_step", "'mean'"),
    ("no_received_measurements_step", "'mean'"),
    ("candidate_norm", "'max'"),
    ("previous_norm", "'max'"),
    ("jump_norm", "'max'"),
    ("jump_limit", "'min'"),
    ("external_received_count", "'min'"),
    ("external_total", "'first'"),
    ("external_support_fraction", "'min'"),
    ("pseudo_received_count", "'max'"),
]

event_renames = {
    "state_update_accepted_step":
        "state_update_accepted_fraction",
    "bootstrap_accept_step":
        "bootstrap_accept_fraction",
    "solve_inexact_hold_step":
        "solve_inexact_hold_fraction",
    "nonfinite_candidate_hold_step":
        "nonfinite_candidate_hold_fraction",
    "jump_guard_hold_step":
        "jump_guard_hold_fraction",
    "pseudo_only_step":
        "pseudo_only_fraction",
    "external_support_present_step":
        "external_support_present_fraction",
    "no_received_measurements_step":
        "no_received_measurements_fraction",
    "candidate_norm":
        "candidate_norm_max",
    "previous_norm":
        "previous_norm_max",
    "jump_norm":
        "jump_norm_max",
    "jump_limit":
        "jump_limit_min",
    "external_received_count":
        "external_received_count_min",
    "external_total":
        "external_total",
    "external_support_fraction":
        "external_support_fraction_min",
    "pseudo_received_count":
        "pseudo_received_count_max",
}

for label, node, additions in (
    ("step", step_dict, step_fields),
    ("publish", publish_dict, publish_fields),
    ("aggregation", aggregation_dict, event_aggregations),
):
    existing = set(mapping(node))
    duplicates = existing.intersection(name for name, _ in additions)

    if duplicates:
        raise RuntimeError(
            f"{label} fields already exist: {sorted(duplicates)}"
        )

line_offsets = []
position = 0

for line in text.splitlines(keepends=True):
    line_offsets.append(position)
    position += len(line)


def modifications_for_dict(node, additions):
    end_line_index = node.end_lineno - 1
    end_line = lines[end_line_index]
    closing_column = node.end_col_offset - 1

    if end_line[closing_column] != "}":
        raise RuntimeError("AST dictionary closing brace was not found.")

    closing_absolute = (
        line_offsets[end_line_index] + closing_column
    )

    prefix_on_closing_line = end_line[:closing_column]

    rendered = [
        repr(name) + ": " + expression
        for name, expression in additions
    ]

    if prefix_on_closing_line.strip():
        previous = text[:closing_absolute].rstrip()
        separator = " " if previous.endswith(",") else ", "

        return [
            (
                closing_absolute,
                separator + ", ".join(rendered),
            )
        ]

    line_start = line_offsets[end_line_index]
    last_nonspace = line_start - 1

    while last_nonspace >= 0 and text[last_nonspace].isspace():
        last_nonspace -= 1

    changes = []

    if last_nonspace < 0:
        raise RuntimeError("Unable to inspect dictionary punctuation.")

    if text[last_nonspace] != ",":
        changes.append((last_nonspace + 1, ","))

    key_indent = node.keys[-1].col_offset

    insertion = "".join(
        " " * key_indent + item + ",\n"
        for item in rendered
    )

    changes.append((line_start, insertion))
    return changes


changes = []

changes.extend(modifications_for_dict(step_dict, step_fields))
changes.extend(modifications_for_dict(publish_dict, publish_fields))
changes.extend(
    modifications_for_dict(
        aggregation_dict,
        event_aggregations,
    )
)

group_assignments = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue

    segment = ast.get_source_segment(text, node) or ""

    if ".groupby(" in segment and ".agg(" in segment:
        group_assignments.append(node)

if len(group_assignments) != 1:
    raise RuntimeError(
        f"Expected one event groupby assignment; "
        f"found {len(group_assignments)}."
    )

group_assignment = group_assignments[0]

if len(group_assignment.targets) != 1:
    raise RuntimeError("Ambiguous event dataframe assignment.")

event_frame = ast.unparse(group_assignment.targets[0])
group_end_offset = sum(
    len(line)
    for line in text.splitlines(keepends=True)[
        :group_assignment.end_lineno
    ]
)

rename_text = (
    " " * group_assignment.col_offset
    + f"{event_frame}.rename(\n"
    + " " * (group_assignment.col_offset + 4)
    + "columns="
    + repr(event_renames)
    + ",\n"
    + " " * (group_assignment.col_offset + 4)
    + "inplace=True,\n"
    + " " * group_assignment.col_offset
    + ")\n"
)

changes.append((group_end_offset, rename_text))

patched = text

for offset, insertion in sorted(
    changes,
    key=lambda item: item[0],
    reverse=True,
):
    patched = patched[:offset] + insertion + patched[offset:]

ast.parse(patched, filename=str(source_path))
compile(patched, str(source_path), "exec")

source_path.write_text(patched, encoding="utf-8")

after_hash = hashlib.sha256(
    source_path.read_bytes()
).hexdigest()

if after_hash == before_hash:
    raise RuntimeError("The v2 logging patch changed nothing.")

report = {
    "schema": "paper1.v5.logging.patch.v2",
    "source_sha256_before": before_hash,
    "source_sha256_after": after_hash,
    "authoritative_step_target": "ProductionTwin.update",
    "publication_target": "publish_score",
    "event_dataframe": event_frame,
    "step_fields_added": [name for name, _ in step_fields],
    "publication_fields_added": [
        name for name, _ in publish_fields
    ],
    "event_aggregations_added": {
        name: expression.strip("'")
        for name, expression in event_aggregations
    },
    "event_columns_renamed": event_renames,
    "performance_outcomes_inspected": False,
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_PATCH_V2_OK")
print("STEP_TARGET=ProductionTwin.update")
print("PUBLISH_TARGET=publish_score")
print("EVENT_DATAFRAME=", event_frame)
print("STEP_FIELDS_ADDED=", len(step_fields))
print("PUBLICATION_FIELDS_ADDED=", len(publish_fields))
print("EVENT_AGGREGATIONS_ADDED=", len(event_aggregations))
print("SOURCE_SHA256_BEFORE=", before_hash)
print("SOURCE_SHA256_AFTER=", after_hash)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")