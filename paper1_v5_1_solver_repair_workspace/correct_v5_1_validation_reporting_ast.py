from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SOURCE = Path(
    "/workspace/paper1_v5_1_solver_repair_workspace/"
    "validate_installed_v5_1_pair.py"
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

text = SOURCE.read_text(encoding="utf-8-sig")
tree = ast.parse(text)

if any(
    isinstance(node, ast.Name)
    and node.id == "pseudo_rank"
    for node in ast.walk(tree)
):
    raise RuntimeError(
        "pseudo_rank already exists; correction was not reapplied."
    )

pseudo_assignments = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue

    assigned_names = set()

    for target in node.targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                assigned_names.add(child.id)

    if "pseudo_reliable" in assigned_names:
        pseudo_assignments.append(node)

if len(pseudo_assignments) != 1:
    raise RuntimeError(
        "Expected exactly one pseudo_reliable assignment; "
        f"found {len(pseudo_assignments)}."
    )

rank_prints = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Expr):
        continue

    call = node.value

    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "print"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "PSEUDO_ONLY_RANK="
    ):
        rank_prints.append(node)

if len(rank_prints) != 1:
    raise RuntimeError(
        "Expected exactly one PSEUDO_ONLY_RANK print; "
        f"found {len(rank_prints)}."
    )

assignment = pseudo_assignments[0]
rank_print = rank_prints[0]

lines = text.splitlines()

print_indent = lines[rank_print.lineno - 1][
    :len(lines[rank_print.lineno - 1])
    - len(lines[rank_print.lineno - 1].lstrip())
]

lines[
    rank_print.lineno - 1 : rank_print.end_lineno
] = [
    f'{print_indent}print("PSEUDO_ONLY_RANK=", pseudo_rank)'
]

assignment_indent = lines[assignment.lineno - 1][
    :len(lines[assignment.lineno - 1])
    - len(lines[assignment.lineno - 1].lstrip())
]

lines.insert(
    assignment.end_lineno,
    f"{assignment_indent}"
    "pseudo_rank = int(metric.last_estimator_rank)",
)

corrected = "\n".join(lines) + "\n"
ast.parse(corrected)

SOURCE.write_text(corrected, encoding="utf-8")

print("V5_1_VALIDATION_REPORTING_AST_CORRECTION_OK")
print("PSEUDO_RANK_CAPTURE_INSTALLED=True")
print("CORRECTED_SCRIPT_SHA256=", sha256(SOURCE))
print("IMPLEMENTATION_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")