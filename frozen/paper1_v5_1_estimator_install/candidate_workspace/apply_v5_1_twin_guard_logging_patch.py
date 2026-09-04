from __future__ import annotations

import ast
import hashlib
from pathlib import Path

WORKSPACE = Path("/workspace/paper1_v5_1_solver_repair_workspace")
SOURCE = WORKSPACE / "twin_fed.v5_1.candidate.py"

EXPECTED_BEFORE = (
    "009bdcc85fec147d0885bec97f0c762de"
    "924626a5f3b8148516ccb56a8d822cf"
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source block; found {count}"
        )
    return text.replace(old, new, 1)

if sha256(SOURCE) != EXPECTED_BEFORE:
    raise RuntimeError(
        "Initial V5.1 twin candidate hash does not match the approved V5 source."
    )

text = SOURCE.read_text(encoding="utf-8-sig")

old_guard = '''        previous_norm = float(np.linalg.norm(self.x_previous))
        state_scale = max(
            previous_norm,
            math.sqrt(self.n) * 0.01,
            1e-12,
        )
        jump_limit = float(self.hold_factor * state_scale)
'''

new_guard = '''        previous_norm = float(np.linalg.norm(self.x_previous))

        omega = float(self.metric.cfg.omega)
        if not math.isfinite(omega) or omega <= 0.0:
            raise RuntimeError(
                "metric omega must be finite and positive"
            )

        model_increment_scale = max(
            math.sqrt(self.n) * omega,
            1e-12,
        )
        jump_limit = float(
            self.hold_factor * model_increment_scale
        )
'''

text = replace_once(
    text,
    old_guard,
    new_guard,
    "fixed model-increment guard",
)

old_attributes = '''        self.last_jump_limit = jump_limit
        self.last_candidate_finite = candidate_finite
        self.last_solved_exactly = bool(solved_exactly)
'''

new_attributes = '''        self.last_jump_limit = jump_limit
        self.last_jump_guard_policy = "fixed_model_increment"
        self.last_model_increment_scale = model_increment_scale
        self.last_candidate_finite = candidate_finite
        self.last_solved_exactly = bool(solved_exactly)
        self.last_estimator_reliable = bool(solved_exactly)
        self.last_estimator_solver = str(
            getattr(
                self.metric,
                "last_estimator_solver",
                "unavailable",
            )
        )
        self.last_estimator_rcond = float(
            getattr(
                self.metric,
                "last_estimator_rcond",
                float("nan"),
            )
        )
        self.last_estimator_effective_rows = int(
            getattr(
                self.metric,
                "last_estimator_effective_rows",
                0,
            )
        )
        self.last_estimator_rank = int(
            getattr(
                self.metric,
                "last_estimator_rank",
                0,
            )
        )
        self.last_estimator_condition = float(
            getattr(
                self.metric,
                "last_estimator_condition",
                float("inf"),
            )
        )
        self.last_estimator_singular_max = float(
            getattr(
                self.metric,
                "last_estimator_singular_max",
                float("nan"),
            )
        )
        self.last_estimator_singular_min = float(
            getattr(
                self.metric,
                "last_estimator_singular_min",
                float("nan"),
            )
        )
        self.last_estimator_residual_norm = float(
            getattr(
                self.metric,
                "last_estimator_residual_norm",
                float("nan"),
            )
        )
'''

text = replace_once(
    text,
    old_attributes,
    new_attributes,
    "guard and estimator diagnostic attributes",
)

old_step = '''            "held": bool(held),
            "solve_exact": bool(solved_exactly),
            "hold_reason": self.last_hold_reason,
'''

new_step = '''            "held": bool(held),
            "solve_exact": bool(solved_exactly),
            "estimator_reliable": self.last_estimator_reliable,
            "estimator_solver": self.last_estimator_solver,
            "estimator_rcond": self.last_estimator_rcond,
            "estimator_effective_rows": self.last_estimator_effective_rows,
            "estimator_rank": self.last_estimator_rank,
            "estimator_condition": self.last_estimator_condition,
            "estimator_singular_max": self.last_estimator_singular_max,
            "estimator_singular_min": self.last_estimator_singular_min,
            "estimator_residual_norm": self.last_estimator_residual_norm,
            "hold_reason": self.last_hold_reason,
'''

text = replace_once(
    text,
    old_step,
    new_step,
    "step estimator diagnostics",
)

old_step_guard = '''            "jump_norm": self.last_jump_norm,
            "jump_limit": self.last_jump_limit,
            "external_received_count": self.last_external_received_count,
'''

new_step_guard = '''            "jump_norm": self.last_jump_norm,
            "jump_limit": self.last_jump_limit,
            "jump_guard_policy": self.last_jump_guard_policy,
            "model_increment_scale": self.last_model_increment_scale,
            "external_received_count": self.last_external_received_count,
'''

text = replace_once(
    text,
    old_step_guard,
    new_step_guard,
    "step fixed-guard diagnostics",
)

old_publication = '''        "held": row["held"],
        "solve_exact": row["solve_exact"],
        "hold_reason": row["hold_reason"],
'''

new_publication = '''        "held": row["held"],
        "solve_exact": row["solve_exact"],
        "estimator_reliable": row["estimator_reliable"],
        "estimator_solver": row["estimator_solver"],
        "estimator_rcond": row["estimator_rcond"],
        "estimator_effective_rows": row["estimator_effective_rows"],
        "estimator_rank": row["estimator_rank"],
        "estimator_condition": row["estimator_condition"],
        "estimator_singular_max": row["estimator_singular_max"],
        "estimator_singular_min": row["estimator_singular_min"],
        "estimator_residual_norm": row["estimator_residual_norm"],
        "hold_reason": row["hold_reason"],
'''

text = replace_once(
    text,
    old_publication,
    new_publication,
    "publication estimator diagnostics",
)

old_publication_guard = '''        "jump_norm": row["jump_norm"],
        "jump_limit": row["jump_limit"],
        "external_received_count": row["external_received_count"],
'''

new_publication_guard = '''        "jump_norm": row["jump_norm"],
        "jump_limit": row["jump_limit"],
        "jump_guard_policy": row["jump_guard_policy"],
        "model_increment_scale": row["model_increment_scale"],
        "external_received_count": row["external_received_count"],
'''

text = replace_once(
    text,
    old_publication_guard,
    new_publication_guard,
    "publication fixed-guard diagnostics",
)

old_event_solver = '''            "solve_exact_all": bool(block["solve_exact"].all()),
            "state_update_accepted_fraction": float(
'''

new_event_solver = '''            "solve_exact_all": bool(block["solve_exact"].all()),
            "estimator_reliable_fraction": float(
                block["estimator_reliable"].mean()
            ),
            "estimator_reliable_all": bool(
                block["estimator_reliable"].all()
            ),
            "estimator_solver": str(first["estimator_solver"]),
            "estimator_rcond": float(first["estimator_rcond"]),
            "estimator_effective_rows_min": int(
                block["estimator_effective_rows"].min()
            ),
            "estimator_effective_rows_max": int(
                block["estimator_effective_rows"].max()
            ),
            "estimator_rank_min": int(
                block["estimator_rank"].min()
            ),
            "estimator_condition_max": float(
                pd.to_numeric(
                    block["estimator_condition"],
                    errors="coerce",
                ).max()
            ),
            "estimator_singular_max_max": float(
                pd.to_numeric(
                    block["estimator_singular_max"],
                    errors="coerce",
                ).max()
            ),
            "estimator_singular_min_min": float(
                pd.to_numeric(
                    block["estimator_singular_min"],
                    errors="coerce",
                ).min()
            ),
            "estimator_residual_norm_max": float(
                pd.to_numeric(
                    block["estimator_residual_norm"],
                    errors="coerce",
                ).max()
            ),
            "state_update_accepted_fraction": float(
'''

text = replace_once(
    text,
    old_event_solver,
    new_event_solver,
    "event estimator diagnostics",
)

old_event_guard = '''            "jump_limit_min": float(
                pd.to_numeric(
                    block["jump_limit"], errors="coerce"
                ).min()
            ),
            "external_received_count_min": int(
'''

new_event_guard = '''            "jump_limit_min": float(
                pd.to_numeric(
                    block["jump_limit"], errors="coerce"
                ).min()
            ),
            "jump_guard_policy": str(first["jump_guard_policy"]),
            "model_increment_scale": float(
                first["model_increment_scale"]
            ),
            "external_received_count_min": int(
'''

text = replace_once(
    text,
    old_event_guard,
    new_event_guard,
    "event fixed-guard diagnostics",
)

ast.parse(text)

if "self.hold_factor * state_scale" in text:
    raise RuntimeError("Self-scaling jump guard remains in candidate source.")

if '"fixed_model_increment"' not in text:
    raise RuntimeError("Fixed guard policy marker is absent.")

SOURCE.write_text(text, encoding="utf-8")

print("PAPER1_V5_1_TWIN_GUARD_LOGGING_PATCH_OK")
print("SOURCE_SHA256_BEFORE=", EXPECTED_BEFORE)
print("SOURCE_SHA256_AFTER=", sha256(SOURCE))
print("JUMP_GUARD_POLICY=fixed_model_increment")
print("JUMP_LIMIT_FORMULA=hold_factor*sqrt(n)*omega")
print("ESTIMATOR_DIAGNOSTIC_STEP_FIELDS=9")
print("GUARD_DIAGNOSTIC_STEP_FIELDS=2")
print("LIVE_FILES_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")