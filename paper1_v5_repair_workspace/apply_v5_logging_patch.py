from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
audit_path = workspace / "paper1_v5_logging_schema_audit.json"
report_path = workspace / "paper1_v5_logging_patch_report.json"

original_bytes = source_path.read_bytes()
before_hash = hashlib.sha256(original_bytes).hexdigest()
text = original_bytes.decode("utf-8-sig")
lines = text.splitlines()
tree = ast.parse(text, filename=str(source_path))

audit = json.loads(audit_path.read_text(encoding="utf-8"))

if audit["performance_outcomes_inspected"] is not False:
    raise RuntimeError("Invalid audit outcome-inspection status.")

if audit["schema_only_inspection"] is not True:
    raise RuntimeError("The prerequisite audit was not schema-only.")


def dict_mapping(node):
    result = {}

    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value

    return result


row_candidates = []
aggregation_candidates = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
        continue

    mapping = dict_mapping(node)

    if "held" not in mapping or "solve_exact" not in mapping:
        continue

    held_value = mapping["held"]

    if (
        isinstance(held_value, ast.Constant)
        and isinstance(held_value.value, str)
    ):
        aggregation_candidates.append(node)
    else:
        row_candidates.append(node)

if len(row_candidates) != 1:
    raise RuntimeError(
        "Expected exactly one step-output dictionary; "
        f"found {len(row_candidates)}."
    )

if len(aggregation_candidates) != 1:
    raise RuntimeError(
        "Expected exactly one event-aggregation dictionary; "
        f"found {len(aggregation_candidates)}."
    )

row_dict = row_candidates[0]
aggregation_dict = aggregation_candidates[0]

step_fields = [
    ("hold_reason", "self.last_hold_reason"),
    (
        "state_update_accepted_step",
        "not bool(held)",
    ),
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

event_aggregations = [
    ("state_update_accepted_step", "mean"),
    ("bootstrap_accept_step", "mean"),
    ("solve_inexact_hold_step", "mean"),
    ("nonfinite_candidate_hold_step", "mean"),
    ("jump_guard_hold_step", "mean"),
    ("pseudo_only_step", "mean"),
    ("external_support_present_step", "mean"),
    ("no_received_measurements_step", "mean"),
    ("candidate_norm", "max"),
    ("previous_norm", "max"),
    ("jump_norm", "max"),
    ("jump_limit", "min"),
    ("external_received_count", "min"),
    ("external_total", "first"),
    ("external_support_fraction", "min"),
    ("pseudo_received_count", "max"),
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

existing_row_keys = set(dict_mapping(row_dict))
existing_aggregation_keys = set(dict_mapping(aggregation_dict))

duplicate_step = existing_row_keys.intersection(
    name for name, _ in step_fields
)

duplicate_event = existing_aggregation_keys.intersection(
    name for name, _ in event_aggregations
)

if duplicate_step:
    raise RuntimeError(
        f"Step diagnostics already exist: {sorted(duplicate_step)}"
    )

if duplicate_event:
    raise RuntimeError(
        f"Event diagnostics already exist: {sorted(duplicate_event)}"
    )


def line_start_offset(source_lines, line_number):
    return sum(
        len(line) + 1
        for line in source_lines[:line_number - 1]
    )


def dictionary_insertion(node, fields):
    insertion_offset = line_start_offset(
        lines,
        node.end_lineno,
    )

    preceding = text[:insertion_offset].rstrip()

    if not preceding.endswith(","):
        raise RuntimeError(
            "The target dictionary does not have a trailing comma; "
            "automatic insertion was refused."
        )

    key_indent = node.keys[-1].col_offset

    insertion_text = "".join(
        " " * key_indent
        + repr(name)
        + ": "
        + expression
        + ",\n"
        for name, expression in fields
    )

    return insertion_offset, insertion_text


row_offset, row_insertion = dictionary_insertion(
    row_dict,
    step_fields,
)

aggregation_fields = [
    (name, repr(method))
    for name, method in event_aggregations
]

agg_offset, agg_insertion = dictionary_insertion(
    aggregation_dict,
    aggregation_fields,
)

group_assignments = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue

    segment = ast.get_source_segment(text, node) or ""

    if (
        ".groupby(" in segment
        and ".agg(" in segment
        and ".reset_index(" in segment
    ):
        group_assignments.append(node)

if len(group_assignments) != 1:
    raise RuntimeError(
        "Expected exactly one event groupby assignment; "
        f"found {len(group_assignments)}."
    )

group_assignment = group_assignments[0]

if len(group_assignment.targets) != 1:
    raise RuntimeError("The event dataframe target is ambiguous.")

event_frame_name = ast.unparse(group_assignment.targets[0])
statement_indent = group_assignment.col_offset

after_group_offset = sum(
    len(line) + 1
    for line in lines[:group_assignment.end_lineno]
)

rename_literal = json.dumps(event_renames, sort_keys=True)

rename_insertion = (
    " " * statement_indent
    + f"{event_frame_name}.rename(\n"
    + " " * (statement_indent + 4)
    + f"columns={rename_literal},\n"
    + " " * (statement_indent + 4)
    + "inplace=True,\n"
    + " " * statement_indent
    + ")\n"
)

insertions = [
    (row_offset, row_insertion),
    (agg_offset, agg_insertion),
    (after_group_offset, rename_insertion),
]

patched_text = text

for offset, insertion in sorted(
    insertions,
    key=lambda item: item[0],
    reverse=True,
):
    patched_text = (
        patched_text[:offset]
        + insertion
        + patched_text[offset:]
    )

patched_tree = ast.parse(
    patched_text,
    filename=str(source_path),
)

compile(patched_text, str(source_path), "exec")

source_path.write_text(patched_text, encoding="utf-8")

after_bytes = source_path.read_bytes()
after_hash = hashlib.sha256(after_bytes).hexdigest()

if after_hash == before_hash:
    raise RuntimeError("The logging patch did not change the candidate.")

report = {
    "schema": "paper1.v5.logging.patch.v1",
    "source_sha256_before": before_hash,
    "source_sha256_after": after_hash,
    "step_fields_added": [name for name, _ in step_fields],
    "event_aggregations_added": dict(event_aggregations),
    "event_columns_renamed": event_renames,
    "event_dataframe": event_frame_name,
    "performance_outcomes_inspected": False,
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_PATCH_OK")
print("SOURCE_SHA256_BEFORE=", before_hash)
print("SOURCE_SHA256_AFTER=", after_hash)
print("STEP_FIELDS_ADDED=", len(step_fields))
print("EVENT_AGGREGATIONS_ADDED=", len(event_aggregations))
print("EVENT_DATAFRAME=", event_frame_name)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")