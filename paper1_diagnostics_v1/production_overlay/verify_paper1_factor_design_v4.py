#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("/workspace"))
args = parser.parse_args()
root = args.root

v3_path = root / "factor_design.production.v3.json"
v4_path = root / "factor_design.paper1.v4.json"
v3 = json.loads(v3_path.read_text(encoding="utf-8"))
v4 = json.loads(v4_path.read_text(encoding="utf-8"))

assert sha256(v3_path) == v4["amendment"]["original_design_sha256"]
assert v3["schema"] == "twin.factor.design.v3"
assert v4["schema"] == "twin.factor.design.paper1.v4"
assert v4["amendment"]["original_design_modified"] is False
assert v4["amendment"]["method_or_threshold_changed"] is False

runner_path = root / v4["production_overlay"]["runner_file"]
assert sha256(runner_path) == v4["production_overlay"]["runner_sha256"]
runner_text = runner_path.read_text(encoding="utf-8-sig")
for marker in (
    "[ValidateRange(2, 31)]",
    'factor_design.paper1.v4.json',
    'paper1_matched_far_thresholds.v1.json',
    'MATCHED_FAR_THRESHOLD_SHA256.txt',
    'test_regressions.py',
    'test_factor_extension.py',
    'truth.eval.paper1.v4.seed$seedText.npz',
    'paper1_s$seedText',
    'cell_record.paper1.v4.json',
):
    assert marker in runner_text, marker
assert '-p "test_*.py"' not in runner_text
assert 'factor_v3_s$seedText' not in runner_text

seeds = v4["seed_policy"]["confirmatory_seed_indices"]
assert seeds == list(range(2, 32))
assert len(seeds) == 30
assert 1 not in seeds
assert v4["qualification"]["seed_index"] == 1
assert v4["qualification"]["physical_seed"] == 81001
assert v4["campaign_cells"] == 150

assert v4["bandwidth_levels"] == v3["bandwidth_levels"]
assert v4["within_run_factors"] == v3["within_run_factors"]
assert v4["exposure_outputs"] == v3["exposure_outputs"]
for key in (
    "calibration_file",
    "calibration_sha256",
    "gamma_file",
    "gamma_sha256",
    "weight_file",
    "weight_sha256",
    "physical_design_file",
    "physical_design_sha256",
):
    assert v4["frozen_inputs"][key] == v3["frozen_inputs"][key], key

contract = v4["matched_far_contract"]
assert contract["source_split"] == "calibration_only"
assert contract["target_far"] == 0.01
assert contract["quantile_method"] == "higher"
assert set(contract["required_metrics"]) == {"s", "chi2"}

threshold_path = root / v4["frozen_inputs"]["matched_far_threshold_file"]
if threshold_path.exists():
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    assert thresholds["schema"] == "paper1.matched_far.thresholds.v1"
    assert thresholds["source_split"] == "calibration_only"
    assert thresholds["target_far"] == 0.01
    assert thresholds["quantile_method"] == "higher"
    assert set(contract["required_metrics"]).issubset(thresholds["thresholds"])
    threshold_status = "FROZEN_AND_VALID"
    threshold_hash = sha256(threshold_path)
else:
    threshold_status = "NOT_YET_FROZEN"
    threshold_hash = "NOT_AVAILABLE"

print("PAPER1_FACTOR_DESIGN_V4_VERIFY_OK")
print("V3_SHA256=", sha256(v3_path))
print("V4_SHA256=", sha256(v4_path))
print("RUNNER_SHA256=", sha256(runner_path))
print("CONFIRMATORY_SEEDS=2-31")
print("CAMPAIGN_CELLS=150")
print("THRESHOLD_STATUS=", threshold_status)
print("THRESHOLD_SHA256=", threshold_hash)
