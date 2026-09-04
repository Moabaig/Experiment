from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

workspace = Path("/workspace/paper1_v5_repair_workspace")
source_path = workspace / "twin_fed.paper1.v5.candidate.py"
report_path = workspace / "paper1_v5_logging_patch_v3_report.json"

before_bytes = source_path.read_bytes()
before_hash = hashlib.sha256(before_bytes).hexdigest()
text = before_bytes.decode("utf-8-sig")

step_anchor = '            "solve_exact": bool(solved_exactly),\n'
publish_anchor = '        "solve_exact": row["solve_exact"],\n'
event_anchor = '            "held_any": bool(block["held"].any()),\n'

for name, anchor in (
    ("step", step_anchor),
    ("publication", publish_anchor),
    ("event", event_anchor),
):
    count = text.count(anchor)

    if count != 1:
        raise RuntimeError(
            f"Expected one {name} anchor; found {count}."
        )

step_addition = '''            "hold_reason": self.last_hold_reason,
            "state_update_accepted_step": not bool(held),
            "bootstrap_accept_step": self.last_hold_reason == "bootstrap_accept",
            "solve_inexact_hold_step": self.last_hold_reason == "solve_inexact",
            "nonfinite_candidate_hold_step": self.last_hold_reason == "nonfinite_candidate",
            "jump_guard_hold_step": self.last_hold_reason == "jump_guard",
            "candidate_norm": self.last_candidate_norm,
            "previous_norm": self.last_previous_norm,
            "jump_norm": self.last_jump_norm,
            "jump_limit": self.last_jump_limit,
            "external_received_count": self.last_external_received_count,
            "external_total": self.last_external_total,
            "external_support_fraction": self.last_external_support_fraction,
            "pseudo_received_count": self.last_pseudo_received_count,
            "pseudo_only_step": self.last_pseudo_only,
            "external_support_present_step": self.last_external_received_count > 0,
            "no_received_measurements_step": (
                self.last_external_received_count == 0
                and self.last_pseudo_received_count == 0
            ),
'''

publish_names = [
    "hold_reason",
    "state_update_accepted_step",
    "bootstrap_accept_step",
    "solve_inexact_hold_step",
    "nonfinite_candidate_hold_step",
    "jump_guard_hold_step",
    "candidate_norm",
    "previous_norm",
    "jump_norm",
    "jump_limit",
    "external_received_count",
    "external_total",
    "external_support_fraction",
    "pseudo_received_count",
    "pseudo_only_step",
    "external_support_present_step",
    "no_received_measurements_step",
]

publish_addition = "".join(
    f'        "{name}": row["{name}"],\n'
    for name in publish_names
)

event_addition = '''            "solve_exact_fraction": float(block["solve_exact"].mean()),
            "solve_exact_all": bool(block["solve_exact"].all()),
            "state_update_accepted_fraction": float(
                block["state_update_accepted_step"].mean()
            ),
            "bootstrap_accept_fraction": float(
                block["bootstrap_accept_step"].mean()
            ),
            "solve_inexact_hold_fraction": float(
                block["solve_inexact_hold_step"].mean()
            ),
            "nonfinite_candidate_hold_fraction": float(
                block["nonfinite_candidate_hold_step"].mean()
            ),
            "jump_guard_hold_fraction": float(
                block["jump_guard_hold_step"].mean()
            ),
            "pseudo_only_fraction": float(
                block["pseudo_only_step"].mean()
            ),
            "external_support_present_fraction": float(
                block["external_support_present_step"].mean()
            ),
            "no_received_measurements_fraction": float(
                block["no_received_measurements_step"].mean()
            ),
            "candidate_norm_max": float(
                pd.to_numeric(
                    block["candidate_norm"], errors="coerce"
                ).max()
            ),
            "previous_norm_max": float(
                pd.to_numeric(
                    block["previous_norm"], errors="coerce"
                ).max()
            ),
            "jump_norm_max": float(
                pd.to_numeric(
                    block["jump_norm"], errors="coerce"
                ).max()
            ),
            "jump_limit_min": float(
                pd.to_numeric(
                    block["jump_limit"], errors="coerce"
                ).min()
            ),
            "external_received_count_min": int(
                block["external_received_count"].min()
            ),
            "external_total": int(block["external_total"].iloc[0]),
            "external_support_fraction_min": float(
                block["external_support_fraction"].min()
            ),
            "pseudo_received_count_max": int(
                block["pseudo_received_count"].max()
            ),
'''

patched = text.replace(
    step_anchor,
    step_anchor + step_addition,
)

patched = patched.replace(
    publish_anchor,
    publish_anchor + publish_addition,
)

patched = patched.replace(
    event_anchor,
    event_anchor + event_addition,
)

ast.parse(patched, filename=str(source_path))
compile(patched, str(source_path), "exec")

source_path.write_text(patched, encoding="utf-8")

after_hash = hashlib.sha256(
    source_path.read_bytes()
).hexdigest()

if after_hash == before_hash:
    raise RuntimeError("The v3 logging patch changed nothing.")

report = {
    "schema": "paper1.v5.logging.patch.v3",
    "source_sha256_before": before_hash,
    "source_sha256_after": after_hash,
    "step_target": "ProductionTwin.update",
    "publication_target": "publish_score",
    "event_target": "aggregate_events",
    "step_fields_added": publish_names,
    "event_fields_added": [
        "solve_exact_fraction",
        "solve_exact_all",
        "state_update_accepted_fraction",
        "bootstrap_accept_fraction",
        "solve_inexact_hold_fraction",
        "nonfinite_candidate_hold_fraction",
        "jump_guard_hold_fraction",
        "pseudo_only_fraction",
        "external_support_present_fraction",
        "no_received_measurements_fraction",
        "candidate_norm_max",
        "previous_norm_max",
        "jump_norm_max",
        "jump_limit_min",
        "external_received_count_min",
        "external_total",
        "external_support_fraction_min",
        "pseudo_received_count_max",
    ],
    "performance_outcomes_inspected": False,
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAPER1_V5_LOGGING_PATCH_V3_OK")
print("SOURCE_SHA256_BEFORE=", before_hash)
print("SOURCE_SHA256_AFTER=", after_hash)
print("STEP_FIELDS_ADDED=", len(publish_names))
print("PUBLICATION_FIELDS_ADDED=", len(publish_names))
print("EVENT_FIELDS_ADDED=", len(report["event_fields_added"]))
print("PERFORMANCE_OUTCOMES_INSPECTED=False")