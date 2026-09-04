from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

root = Path("/workspace/paper1_v5_repair_workspace")
source_path = root / "twin_fed.paper1.v5.candidate.py"
report_path = root / "paper1_v5_guard_patch_report.json"

original_bytes = source_path.read_bytes()
original_sha256 = hashlib.sha256(original_bytes).hexdigest()
text = original_bytes.decode("utf-8-sig")
lines = text.splitlines()

tree = ast.parse(text, filename=str(source_path))

target_class = None
target_function = None
start_statement = None
end_statement = None

for class_node in [
    node for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef)
]:
    for function_node in [
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        estimator_statements = []

        for statement in function_node.body:
            segment = ast.get_source_segment(text, statement) or ""

            if ".estimate(z, rx, gamma)" in segment:
                estimator_statements.append(statement)

        if estimator_statements:
            if target_function is not None:
                raise RuntimeError(
                    "More than one production function contains the "
                    "target estimator call."
                )

            if len(estimator_statements) != 1:
                raise RuntimeError(
                    "The target function contains an ambiguous number "
                    "of estimator statements."
                )

            target_class = class_node
            target_function = function_node
            start_statement = estimator_statements[0]

if target_function is None or start_statement is None:
    raise RuntimeError("The production estimator-hold function was not found.")

start_index = target_function.body.index(start_statement)

for statement in target_function.body[start_index:]:
    segment = ast.get_source_segment(text, statement) or ""

    if (
        isinstance(statement, ast.Return)
        and "self.x_previous.copy()" in segment
        and "False" in segment
        and "solved_exactly" in segment
    ):
        end_statement = statement
        break

if end_statement is None:
    raise RuntimeError("The end of the existing hold block was not found.")

start_line = start_statement.lineno
end_line = getattr(end_statement, "end_lineno", end_statement.lineno)

old_region = "\n".join(lines[start_line - 1:end_line])

required_fragments = [
    "self.metric.estimate(z, rx, gamma)",
    "np.linalg.norm(candidate)",
    "self.hold_factor * scale",
    "return self.x_previous.copy(), True, solved_exactly",
    "return self.x_previous.copy(), False, solved_exactly",
]

missing_fragments = [
    fragment
    for fragment in required_fragments
    if fragment not in old_region
]

if missing_fragments:
    raise RuntimeError(
        "The existing guard block does not match the audited v4 "
        f"semantics. Missing fragments: {missing_fragments}"
    )

if end_line - start_line > 45:
    raise RuntimeError(
        "The proposed replacement region is unexpectedly large."
    )

first_line = lines[start_line - 1]
indent = first_line[:len(first_line) - len(first_line.lstrip())]

replacement_lines = [
    "candidate, solved_exactly = self.metric.estimate(z, rx, gamma)",
    "candidate_array = np.asarray(candidate, dtype=float)",
    "candidate_finite = bool(np.all(np.isfinite(candidate_array)))",
    "",
    "had_valid_estimate = bool(",
    '    getattr(self, "has_valid_estimate", False)',
    ")",
    "",
    "previous_norm = float(np.linalg.norm(self.x_previous))",
    "state_scale = max(",
    "    previous_norm,",
    "    math.sqrt(self.n) * 0.01,",
    "    1e-12,",
    ")",
    "jump_limit = float(self.hold_factor * state_scale)",
    "",
    "if candidate_finite:",
    "    candidate_norm = float(np.linalg.norm(candidate_array))",
    "else:",
    '    candidate_norm = float("nan")',
    "",
    "if had_valid_estimate and candidate_finite:",
    "    jump_norm = float(",
    "        np.linalg.norm(candidate_array - self.x_previous)",
    "    )",
    "else:",
    '    jump_norm = float("nan")',
    "",
    "rx_array = np.asarray(rx, dtype=bool).reshape(-1)",
    "configured_telemetry = int(",
    '    getattr(self, "n_telemetry", 0)',
    ")",
    "external_total = min(configured_telemetry, len(rx_array))",
    "external_received = int(",
    "    np.count_nonzero(rx_array[:external_total])",
    ")",
    "pseudo_received = int(",
    "    np.count_nonzero(rx_array[external_total:])",
    ")",
    "",
    "if external_received > 0:",
    '    external_support_state = "external_present"',
    "elif pseudo_received > 0:",
    '    external_support_state = "pseudo_only"',
    "else:",
    '    external_support_state = "no_received_measurements"',
    "",
    "if not solved_exactly:",
    '    decision_reason = "solve_inexact"',
    "    held = True",
    "elif not candidate_finite:",
    '    decision_reason = "nonfinite_candidate"',
    "    held = True",
    "elif not had_valid_estimate:",
    '    decision_reason = "bootstrap_accept"',
    "    held = False",
    "elif jump_norm > jump_limit:",
    '    decision_reason = "jump_guard"',
    "    held = True",
    "else:",
    '    decision_reason = "accepted"',
    "    held = False",
    "",
    "self.last_hold_reason = decision_reason",
    "self.last_candidate_norm = candidate_norm",
    "self.last_previous_norm = previous_norm",
    "self.last_jump_norm = jump_norm",
    "self.last_jump_limit = jump_limit",
    "self.last_candidate_finite = candidate_finite",
    "self.last_solved_exactly = bool(solved_exactly)",
    "self.last_external_received_count = external_received",
    "self.last_external_total = external_total",
    "self.last_pseudo_received_count = pseudo_received",
    "self.last_external_support_state = external_support_state",
    "self.last_pseudo_only = external_support_state == \"pseudo_only\"",
    "self.last_external_support_fraction = (",
    "    float(external_received / external_total)",
    "    if external_total > 0",
    "    else 0.0",
    ")",
    "",
    "if held:",
    "    return self.x_previous.copy(), True, solved_exactly",
    "",
    "self.x_previous = candidate_array.copy()",
    "self.has_valid_estimate = True",
    "return self.x_previous.copy(), False, solved_exactly",
]

replacement = "\n".join(
    indent + line if line else ""
    for line in replacement_lines
)

new_lines = (
    lines[:start_line - 1]
    + replacement.splitlines()
    + lines[end_line:]
)

new_text = "\n".join(new_lines) + "\n"

ast.parse(new_text, filename=str(source_path))
compile(new_text, str(source_path), "exec")

source_path.write_text(new_text, encoding="utf-8")

patched_bytes = source_path.read_bytes()
patched_sha256 = hashlib.sha256(patched_bytes).hexdigest()

if patched_sha256 == original_sha256:
    raise RuntimeError("The v5 candidate hash did not change.")

report = {
    "schema": "paper1.v5.guard.patch.v1",
    "source": str(source_path),
    "source_sha256_before": original_sha256,
    "source_sha256_after": patched_sha256,
    "patched_class": target_class.name,
    "patched_method": target_function.name,
    "old_line_start": start_line,
    "old_line_end": end_line,
    "repair_semantics": [
        "finite_exact_first_candidate_bootstrap",
        "post_bootstrap_increment_guard",
        "explicit_hold_reason",
        "external_and_pseudo_support_diagnostics",
        "unchanged_three_value_return_interface",
    ],
    "performance_outcomes_inspected": False,
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_GUARD_PATCH_OK")
print("PATCHED_CLASS=", target_class.name)
print("PATCHED_METHOD=", target_function.name)
print("OLD_LINE_RANGE=", f"{start_line}-{end_line}")
print("SOURCE_SHA256_BEFORE=", original_sha256)
print("SOURCE_SHA256_AFTER=", patched_sha256)
print("PERFORMANCE_OUTCOMES_INSPECTED=False")