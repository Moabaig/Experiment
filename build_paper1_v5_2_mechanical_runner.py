from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(os.environ.get("PAPER1_V5_2_ROOT", "/workspace"))
WORKSPACE = ROOT / "paper1_v5_2_repair"
SOURCE_RUNNER = ROOT / "run_paper1_factor_campaign_v5_1_mechanical.ps1"
RUNNER_CANDIDATE = ROOT / "run_paper1_factor_campaign_v5_2_mechanical.ps1"
CONTRACT = WORKSPACE / "paper1_v5_2_mechanical_validation_contract.json"
REPORT = WORKSPACE / "v5_2_mechanical_runner_build.json"

LIVE_TRUST = ROOT / "trust_metric.py"
LIVE_TWIN = ROOT / "twin_fed.py"
INSTALLED_VALIDATION = (
    WORKSPACE / "v5_2_installed_pair_validation.json"
)
ROOT_CAUSE_REPORT = (
    WORKSPACE / "v5_2_observability_ratcheting_diagnostic.json"
)
ESTIMATOR_REPORT = (
    WORKSPACE / "v5_2_estimator_candidate_tests.json"
)
INTEGRATION_REPORT = (
    WORKSPACE / "v5_2_twin_integration_tests.json"
)

EXPECTED_SOURCE_RUNNER_SHA256 = (
    "31cc460690323100fdcc10df7162db0a"
    "92d615035dbc6e9319c09d1610b1dae9"
)
EXPECTED_PREVIOUS_TRUST_SHA256 = (
    "0a2627bdaacad03e582bb039eeb2fb3ac"
    "73d33d20b77e96881ebceec64aae437"
)
EXPECTED_PREVIOUS_TWIN_SHA256 = (
    "39e6729af233032ab9c58851c968225"
    "2f02d36eed739eb2ec769e165659da34c"
)
EXPECTED_INSTALLED_TRUST_SHA256 = (
    "936dd373a2d8a2f0b905604ca4c3de61"
    "ec2cc889ba233aa150a24f44f2926fe6"
)
EXPECTED_INSTALLED_TWIN_SHA256 = (
    "9cd9ffaa32dcfe2f12ed161a8d62d2d9"
    "7b2ab0b4d462fda0e97e7f46572043a4"
)
EXPECTED_INSTALLED_VALIDATION_SHA256 = (
    "8ebedb33ea9ed9d2c8488b4d6d6245a"
    "9f36632776b569a5c683c81619817ac72"
)
EXPECTED_ROOT_CAUSE_REPORT_SHA256 = (
    "deb3a8b15d8f6bf2cf942e06d45404b6"
    "c5459db783c293d375ec0421a7c21ab2"
)
EXPECTED_ESTIMATOR_REPORT_SHA256 = (
    "8a132c97ea5b41ce69127c05c377536d"
    "4441721583eb049737a129897390f9ae"
)
EXPECTED_INTEGRATION_REPORT_SHA256 = (
    "00e56b05edd24581ebef8daecaff9eed"
    "0acbbeae41029f43d97921823b4903a8"
)

AUTHORIZED_RUN_ID = "paper1_v5_2mv_s002_bw04_oracle"
AUTHORIZED_SEED_FROM = 2
AUTHORIZED_SEED_TO = 2
AUTHORIZED_BANDWIDTH_FROM = 4
AUTHORIZED_BANDWIDTH_TO = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_required(
    source: str,
    old: str,
    new: str,
    label: str,
) -> tuple[str, int]:
    count = source.count(old)
    require(count > 0, f"required runner token is absent: {label}")
    return source.replace(old, new), count


required_files = {
    SOURCE_RUNNER: EXPECTED_SOURCE_RUNNER_SHA256,
    LIVE_TRUST: EXPECTED_INSTALLED_TRUST_SHA256,
    LIVE_TWIN: EXPECTED_INSTALLED_TWIN_SHA256,
    INSTALLED_VALIDATION: EXPECTED_INSTALLED_VALIDATION_SHA256,
    ROOT_CAUSE_REPORT: EXPECTED_ROOT_CAUSE_REPORT_SHA256,
    ESTIMATOR_REPORT: EXPECTED_ESTIMATOR_REPORT_SHA256,
    INTEGRATION_REPORT: EXPECTED_INTEGRATION_REPORT_SHA256,
}
for path, expected_hash in required_files.items():
    require(path.is_file(), f"required input is missing: {path}")
    observed_hash = sha256(path)
    require(
        observed_hash == expected_hash,
        f"input hash mismatch for {path}: {observed_hash}",
    )

source_bytes = SOURCE_RUNNER.read_bytes()
source = source_bytes.decode("utf-8-sig")
require("\x00" not in source, "runner source contains NUL bytes")

replacement_counts: dict[str, int] = {}

# Paths are replaced before the general version tokens so a V5.1 workspace
# name cannot be converted into a nonexistent V5.2 solver-repair directory.
optional_path_replacements = [
    (
        "paper1_v5_1_solver_repair_workspace",
        "paper1_v5_2_repair",
        "repair_workspace",
    ),
]
for old, new, label in optional_path_replacements:
    count = source.count(old)
    if count:
        source = source.replace(old, new)
    replacement_counts[label] = count

required_text_replacements = [
    (
        "paper1_v5_1",
        "paper1_v5_2",
        "lowercase_version_namespace",
    ),
    (
        "PAPER1_V5_1",
        "PAPER1_V5_2",
        "uppercase_version_namespace",
    ),
    (
        "v5_1",
        "v5_2",
        "lowercase_standalone_version_token",
    ),
    (
        "weighted_lstsq_svd",
        "q_prior_innovation_after_measurement_bootstrap",
        "estimator_descriptor",
    ),
    (
        "fixed_model_increment",
        "q_process_mahalanobis",
        "guard_policy",
    ),
]

for old, new, label in [
    (
        EXPECTED_PREVIOUS_TRUST_SHA256,
        EXPECTED_INSTALLED_TRUST_SHA256.upper(),
        "trust_hash_any_case",
    ),
    (
        EXPECTED_PREVIOUS_TWIN_SHA256,
        EXPECTED_INSTALLED_TWIN_SHA256.upper(),
        "twin_hash_any_case",
    ),
]:
    source, count = re.subn(
        re.escape(old),
        new,
        source,
        flags=re.IGNORECASE,
    )
    require(count > 0, f"required runner token is absent: {label}")
    replacement_counts[label] = count

for old, new, label in required_text_replacements:
    source, count = replace_required(source, old, new, label)
    replacement_counts[label] = count

# Human-readable spellings may or may not occur. Replace every occurrence
# without weakening the required machine-token checks above.
for old, new, label in [
    ("V5.1", "V5.2", "display_version_upper"),
    ("v5.1", "v5.2", "display_version_lower"),
    ("V5_1", "V5_2", "uppercase_standalone_version_token"),
]:
    count = source.count(old)
    if count:
        source = source.replace(old, new)
    replacement_counts[label] = count

for forbidden in (
    "paper1_v5_1",
    "PAPER1_V5_1",
    "v5_1",
    "weighted_lstsq_svd",
    "fixed_model_increment",
    EXPECTED_PREVIOUS_TRUST_SHA256,
    EXPECTED_PREVIOUS_TWIN_SHA256,
):
    require(
        forbidden.lower() not in source.lower(),
        f"legacy token remains in V5.2 runner: {forbidden}",
    )

required_candidate_tokens = (
    "paper1_factor_campaign_v5_2_mechanical",
    "paper1_v5_2mv_",
    "paper1_v5_2",
    "PAPER1_V5_2",
    EXPECTED_INSTALLED_TRUST_SHA256,
    EXPECTED_INSTALLED_TWIN_SHA256,
    "q_prior_innovation_after_measurement_bootstrap",
    "q_process_mahalanobis",
)
for token in required_candidate_tokens:
    require(
        token.lower() in source.lower(),
        f"required V5.2 runner token is missing: {token}",
    )

require(
    "paper1_factor_campaign_v5_1_mechanical" not in source.lower(),
    "V5.2 runner still targets the frozen V5.1 campaign directory",
)

# Reject accidental broad authorization embedded into the runner source.
dangerous_assignments = [
    r"FULL_CAMPAIGN_AUTHORIZED\s*=\s*True",
    r"CALIBRATION_AUTHORIZED\s*=\s*True",
    r"PERFORMANCE_OUTCOMES_INSPECTED\s*=\s*True",
]
for pattern in dangerous_assignments:
    require(
        re.search(pattern, source, flags=re.IGNORECASE) is None,
        f"forbidden authorization appears in runner: {pattern}",
    )

WORKSPACE.mkdir(parents=True, exist_ok=True)
RUNNER_CANDIDATE.write_text(
    source,
    encoding="utf-8",
    newline="\n",
)
runner_hash = sha256(RUNNER_CANDIDATE)

contract = {
    "schema": "paper1.v5_2.mechanical_validation_contract.v1",
    "purpose": "single_cell_estimator_and_state_scale_mechanical_validation",
    "candidate_version": "paper1_v5_2",
    "run_prefix": "paper1_v5_2mv_",
    "runner": {
        "path": str(RUNNER_CANDIDATE),
        "sha256": runner_hash,
        "source_v5_1_sha256": EXPECTED_SOURCE_RUNNER_SHA256,
    },
    "installed_pair": {
        "trust_metric_sha256": EXPECTED_INSTALLED_TRUST_SHA256,
        "twin_fed_sha256": EXPECTED_INSTALLED_TWIN_SHA256,
        "validation_report_sha256": (
            EXPECTED_INSTALLED_VALIDATION_SHA256
        ),
    },
    "estimator": {
        "mode": "q_prior_innovation_after_measurement_bootstrap",
        "prior_center": "previous_state",
        "prior_covariance": "Q_dt_1_second",
        "normal_equations_used": False,
    },
    "guard": {
        "policy": "q_process_mahalanobis",
        "limit_source": "existing_hold_factor_dimensionless",
        "legacy_euclidean_jump_guard_active": False,
    },
    "authorized_cells": [
        {
            "run_id": AUTHORIZED_RUN_ID,
            "seed_from": AUTHORIZED_SEED_FROM,
            "seed_to": AUTHORIZED_SEED_TO,
            "bandwidth_from": AUTHORIZED_BANDWIDTH_FROM,
            "bandwidth_to": AUTHORIZED_BANDWIDTH_TO,
            "bandwidth_level": "bw04_oracle",
            "bandwidth_cap_bps": 1_000_000_000_000.0,
        }
    ],
    "authorization": {
        "mechanical_validation_only": True,
        "authorized_cell_count": 1,
        "calibration_authorized": False,
        "full_campaign_authorized": False,
        "performance_outcome_columns_may_be_read": False,
        "performance_outcomes_may_be_inspected": False,
    },
    "source_evidence": {
        "root_cause_report_sha256": EXPECTED_ROOT_CAUSE_REPORT_SHA256,
        "estimator_candidate_report_sha256": (
            EXPECTED_ESTIMATOR_REPORT_SHA256
        ),
        "twin_integration_report_sha256": (
            EXPECTED_INTEGRATION_REPORT_SHA256
        ),
    },
}
CONTRACT.write_text(
    json.dumps(contract, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
contract_hash = sha256(CONTRACT)

report = {
    "schema": "paper1.v5_2.mechanical_runner_build.v1",
    "source_runner_sha256": EXPECTED_SOURCE_RUNNER_SHA256,
    "candidate_runner_sha256": runner_hash,
    "contract_sha256": contract_hash,
    "replacement_counts": replacement_counts,
    "authorized_run_id": AUTHORIZED_RUN_ID,
    "authorized_cell_count": 1,
    "live_files_modified": False,
    "simulation_started": False,
    "calibration_authorized": False,
    "full_campaign_authorized": False,
    "performance_outcome_columns_read": False,
    "performance_outcomes_inspected": False,
}
REPORT.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

require(
    sha256(LIVE_TRUST) == EXPECTED_INSTALLED_TRUST_SHA256,
    "live trust changed during runner build",
)
require(
    sha256(LIVE_TWIN) == EXPECTED_INSTALLED_TWIN_SHA256,
    "live twin changed during runner build",
)

print("PAPER1_V5_2_MECHANICAL_RUNNER_BUILD_OK")
print("SOURCE_RUNNER_SHA256=", EXPECTED_SOURCE_RUNNER_SHA256)
print("RUNNER_CANDIDATE_SHA256=", runner_hash)
print("CONTRACT_SHA256=", contract_hash)
print("BUILD_REPORT_SHA256=", sha256(REPORT))
print("AUTHORIZED_RUN_ID=", AUTHORIZED_RUN_ID)
print("AUTHORIZED_CELL_COUNT=1")
print("ESTIMATOR_MODE=q_prior_innovation_after_measurement_bootstrap")
print("PROCESS_GUARD_POLICY=q_process_mahalanobis")
print("LIVE_FILES_MODIFIED=False")
print("SIMULATION_STARTED=False")
print("FULL_CAMPAIGN_AUTHORIZED=False")
print("CALIBRATION_AUTHORIZED=False")
print("PERFORMANCE_OUTCOME_COLUMNS_READ=False")
print("PERFORMANCE_OUTCOMES_INSPECTED=False")
print("PAPER1_V5_2_RUNNER_CANDIDATE_READY_FOR_HOST_PARSE_AND_PREFLIGHT")
