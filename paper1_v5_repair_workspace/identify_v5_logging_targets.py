from __future__ import annotations

import ast
import json
from pathlib import Path

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
report_path = workspace / "paper1_v5_logging_target_candidates.json"

text = source_path.read_text(encoding="utf-8-sig")
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


def dict_mapping(node):
    result = {}

    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value

    return result


candidates = []

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
        continue

    function = enclosing(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    )

    class_node = enclosing(node, ast.ClassDef)
    parent_node = parents.get(node)

    start = max(1, node.lineno - 10)
    end = min(
        len(lines),
        getattr(node, "end_lineno", node.lineno) + 10,
    )

    context = "\n".join(
        f"{number:05d}: {lines[number - 1]}"
        for number in range(start, end + 1)
    )

    candidates.append(
        {
            "candidate_number": len(candidates) + 1,
            "class": class_node.name if class_node else None,
            "function": function.name if function else None,
            "parent_type": (
                type(parent_node).__name__
                if parent_node is not None
                else None
            ),
            "line_start": node.lineno,
            "line_end": getattr(
                node,
                "end_lineno",
                node.lineno,
            ),
            "keys": list(mapping),
            "held_expression": ast.unparse(mapping["held"]),
            "solve_exact_expression": ast.unparse(
                mapping["solve_exact"]
            ),
            "context": context,
        }
    )

if len(candidates) != 2:
    raise RuntimeError(
        f"Expected two candidates after the failed patch; "
        f"found {len(candidates)}."
    )

report = {
    "schema": "paper1.v5.logging.targets.audit.v1",
    "candidate_count": len(candidates),
    "candidates": candidates,
    "performance_outcomes_inspected": False,
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_TARGET_DISAMBIGUATION")
print("CANDIDATE_COUNT=", len(candidates))
print("PERFORMANCE_OUTCOMES_INSPECTED=False")

for candidate in candidates:
    print()
    print(
        "===== CANDIDATE",
        candidate["candidate_number"],
        "====="
    )
    print("CLASS=", candidate["class"])
    print("FUNCTION=", candidate["function"])
    print("PARENT_TYPE=", candidate["parent_type"])
    print(
        "LINES=",
        f"{candidate['line_start']}-{candidate['line_end']}",
    )
    print("KEYS=", ",".join(candidate["keys"]))
    print(
        "HELD_EXPRESSION=",
        candidate["held_expression"],
    )
    print(
        "SOLVE_EXACT_EXPRESSION=",
        candidate["solve_exact_expression"],
    )
    print(candidate["context"])

print()
print("PAPER1_V5_LOGGING_TARGETS_IDENTIFIED")