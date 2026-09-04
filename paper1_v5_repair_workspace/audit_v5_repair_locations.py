from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

root = Path("/workspace/paper1_v5_repair_workspace")
source_path = root / "twin_fed.paper1.v5.candidate.py"
report_path = root / "paper1_v5_source_audit.json"
context_path = root / "paper1_v5_source_context.txt"

text = source_path.read_text(encoding="utf-8-sig")
lines = text.splitlines()
tree = ast.parse(text, filename=str(source_path))

sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

terms = {
    "absolute_candidate_guard": "np.linalg.norm(candidate)",
    "held_assignment": "held = (",
    "previous_state": "self.x_previous",
    "hold_factor": "hold_factor",
    "solved_exactly": "solved_exactly",
    "estimator_call": ".estimate(z, rx, gamma)",
    "telemetry_boundary": "n_telemetry",
    "pseudo_comment": "pseudo",
    "received_slice": "rx[self.n_telemetry",
    "age_slice": "age[self.n_telemetry",
    "bandwidth_slice": "bandwidth_effective[self.n_telemetry",
}

occurrences = {}

for name, term in terms.items():
    occurrences[name] = [
        index
        for index, line in enumerate(lines, start=1)
        if term.lower() in line.lower()
    ]

functions = []

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        segment = ast.get_source_segment(text, node) or ""

        relevant_terms = [
            term
            for term in (
                "hold_factor",
                "self.x_previous",
                "solved_exactly",
                "n_telemetry",
                "pseudo",
            )
            if term.lower() in segment.lower()
        ]

        if relevant_terms:
            functions.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "relevant_terms": relevant_terms,
                }
            )

guard_lines = occurrences["held_assignment"]

if len(guard_lines) != 1:
    raise RuntimeError(
        "Expected exactly one multi-line held assignment; "
        f"found {guard_lines}"
    )

if len(occurrences["estimator_call"]) != 1:
    raise RuntimeError(
        "Expected exactly one estimator call; found "
        f"{occurrences['estimator_call']}"
    )

guard_line = guard_lines[0]

selected_centres = set()

for key in (
    "held_assignment",
    "previous_state",
    "estimator_call",
    "received_slice",
    "age_slice",
    "bandwidth_slice",
):
    selected_centres.update(occurrences[key])

windows = []

for centre in sorted(selected_centres):
    start = max(1, centre - 12)
    end = min(len(lines), centre + 18)

    if windows and start <= windows[-1][1] + 1:
        windows[-1] = (windows[-1][0], max(windows[-1][1], end))
    else:
        windows.append((start, end))

context_sections = []

for start, end in windows:
    context_sections.append(
        f"\n===== SOURCE LINES {start}-{end} =====\n"
    )

    for line_number in range(start, end + 1):
        context_sections.append(
            f"{line_number:05d}: {lines[line_number - 1]}\n"
        )

context_path.write_text(
    "".join(context_sections),
    encoding="utf-8",
)

report = {
    "schema": "paper1.v5.source.audit.v1",
    "source": str(source_path),
    "source_sha256": sha256,
    "source_lines": len(lines),
    "syntax_valid": True,
    "performance_outcomes_inspected": False,
    "guard_line": guard_line,
    "occurrences": occurrences,
    "relevant_functions": functions,
    "context_file": str(context_path),
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

compile(
    text,
    str(source_path),
    "exec",
)

print("PAPER1_V5_SOURCE_AUDIT_OK")
print("SOURCE_SHA256=", sha256)
print("SOURCE_LINES=", len(lines))
print("HELD_ASSIGNMENT_LINE=", guard_line)
print("ESTIMATOR_CALL_LINE=", occurrences["estimator_call"][0])
print("PREVIOUS_STATE_OCCURRENCES=", occurrences["previous_state"])
print("RECEIVED_SLICE_OCCURRENCES=", occurrences["received_slice"])
print("AGE_SLICE_OCCURRENCES=", occurrences["age_slice"])
print(
    "BANDWIDTH_SLICE_OCCURRENCES=",
    occurrences["bandwidth_slice"],
)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("CONTEXT_FILE=", context_path)
print()
print(context_path.read_text(encoding="utf-8"))