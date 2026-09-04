from pathlib import Path
import ast
import hashlib

path = Path(
    "/workspace/paper1_v5_1_solver_repair_workspace/"
    "trust_metric.v5_1.candidate.py"
)

expected_hash = (
    "be0d1dc0c5f8924a7794b9923d9ee3bb"
    "373da08266c65ff16a968cbe2c3e1ab4"
)

source_bytes = path.read_bytes()
actual_hash = hashlib.sha256(source_bytes).hexdigest()

if actual_hash != expected_hash:
    raise RuntimeError(
        f"candidate source hash mismatch: {actual_hash}"
    )

source = source_bytes.decode("utf-8-sig")
tree = ast.parse(source)

target = None

for node in tree.body:
    if (
        isinstance(node, ast.ClassDef)
        and node.name == "TrustMetric"
    ):
        for child in node.body:
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "estimate"
            ):
                target = child
                break

if target is None:
    raise RuntimeError(
        "TrustMetric.estimate was not found"
    )

old_method = "\n".join(
    source.splitlines()[
        target.lineno - 1:target.end_lineno
    ]
)

required_old_fragments = [
    "A = Hr.T @ (Hr * w[:, None])",
    "np.linalg.solve(A,",
    "np.linalg.lstsq(A,",
]

for fragment in required_old_fragments:
    if fragment not in old_method:
        raise RuntimeError(
            f"expected old solver fragment missing: {fragment}"
        )

new_method = '''    def estimate(self, z, rx, gamma):
        """Solve the age-aware WLS problem without forming normal equations.

        Rows are scaled by sqrt(gamma / sigma2), and the resulting
        least-squares system is solved directly by rank-revealing SVD.
        The Boolean return value means full numerical column rank and
        a finite solution; a LAPACK call merely returning is not enough.
        """
        z_array = np.asarray(z, dtype=float).reshape(-1)
        rx_array = np.asarray(rx, dtype=bool).reshape(-1)
        gamma_array = np.asarray(gamma, dtype=float).reshape(-1)

        if (
            len(z_array) != self.m
            or len(rx_array) != self.m
            or len(gamma_array) != self.m
        ):
            raise ValueError(
                "z, rx, and gamma must match the measurement dimension"
            )

        idx = np.flatnonzero(rx_array)
        raw_weights = gamma_array[idx] / self.s2[idx]

        valid = (
            np.isfinite(raw_weights)
            & (raw_weights > 0.0)
            & np.isfinite(z_array[idx])
        )

        idx = idx[valid]
        weights = raw_weights[valid]

        rcond = float(np.sqrt(np.finfo(float).eps))
        ridge = float(self.cfg.ridge)

        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError(
                "estimator ridge must be finite and nonnegative"
            )

        self.last_estimator_solver = "weighted_lstsq_svd"
        self.last_estimator_rcond = rcond
        self.last_estimator_effective_rows = int(len(idx))
        self.last_estimator_rank = 0
        self.last_estimator_condition = float("inf")
        self.last_estimator_singular_max = float("nan")
        self.last_estimator_singular_min = float("nan")
        self.last_estimator_residual_norm = float("nan")

        if len(idx) == 0:
            return np.full(self.n, np.nan), False

        square_root_weights = np.sqrt(weights)
        weighted_h = (
            self.H[idx]
            * square_root_weights[:, None]
        )
        weighted_z = (
            z_array[idx]
            * square_root_weights
        )

        if (
            not np.all(np.isfinite(weighted_h))
            or not np.all(np.isfinite(weighted_z))
        ):
            return np.full(self.n, np.nan), False

        if ridge > 0.0:
            weighted_h = np.vstack(
                [
                    weighted_h,
                    np.eye(self.n) * np.sqrt(ridge),
                ]
            )
            weighted_z = np.concatenate(
                [
                    weighted_z,
                    np.zeros(self.n),
                ]
            )

        try:
            (
                candidate,
                _,
                rank,
                singular_values,
            ) = np.linalg.lstsq(
                weighted_h,
                weighted_z,
                rcond=rcond,
            )
        except np.linalg.LinAlgError:
            return np.full(self.n, np.nan), False

        candidate = np.asarray(
            candidate,
            dtype=float,
        ).reshape(-1)

        rank = int(rank)
        singular_values = np.asarray(
            singular_values,
            dtype=float,
        )

        if len(singular_values) > 0:
            singular_max = float(
                singular_values[0]
            )
            singular_min = float(
                singular_values[-1]
            )

            condition = (
                float(singular_max / singular_min)
                if singular_min > 0.0
                else float("inf")
            )
        else:
            singular_max = float("nan")
            singular_min = float("nan")
            condition = float("inf")

        candidate_finite = bool(
            candidate.shape == (self.n,)
            and np.all(np.isfinite(candidate))
        )

        residual_norm = (
            float(
                np.linalg.norm(
                    weighted_h @ candidate
                    - weighted_z
                )
            )
            if candidate_finite
            else float("nan")
        )

        full_numerical_rank = bool(
            rank == self.n
            and np.isfinite(condition)
            and condition <= 1.0 / rcond
        )

        self.last_estimator_rank = rank
        self.last_estimator_condition = condition
        self.last_estimator_singular_max = singular_max
        self.last_estimator_singular_min = singular_min
        self.last_estimator_residual_norm = residual_norm

        solved_reliably = bool(
            candidate_finite
            and full_numerical_rank
        )

        return candidate, solved_reliably'''

lines = source.splitlines()

updated_lines = (
    lines[:target.lineno - 1]
    + new_method.splitlines()
    + lines[target.end_lineno:]
)

updated = "\n".join(updated_lines) + "\n"

compile(
    updated,
    str(path),
    "exec",
)

path.write_text(
    updated,
    encoding="utf-8",
)

new_hash = hashlib.sha256(
    path.read_bytes()
).hexdigest()

print("PAPER1_V5_1_STABLE_SOLVER_PATCH_OK")
print("OLD_SHA256=", actual_hash)
print("NEW_SHA256=", new_hash)
print("OLD_METHOD_LINES=", target.lineno, target.end_lineno)
print("SOLVER=weighted_lstsq_svd")
print("RCOND=sqrt(machine_epsilon)")
print("NORMAL_EQUATIONS_REMOVED=True")
print("LIVE_FILES_MODIFIED=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")