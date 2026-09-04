# Paper 1 v4 production overlay

Copy the following files into the root of the real experiment bundle without
replacing `factor_design.production.v3.json`:

- `factor_design.paper1.v4.json`
- `verify_paper1_factor_design_v4.py`
- `run_paper1_factor_campaign_v4.ps1`

Also copy the complete `paper1_diagnostics_v1` directory so the calibration
threshold freezer and campaign analyzer are available inside `/workspace`.

The v4 amendment makes one design change only: it excludes inspected physical
seed 81001 and adds unseen physical seed 81031. All frozen calibration,
physical-design, gamma, weight, estimator, network, bandwidth, event, and
analysis definitions remain unchanged.

Execution order:

1. Freeze `paper1_matched_far_thresholds.v1.json` from calibration-only data.
2. Run `verify_paper1_factor_design_v4.py`; it should report
   `THRESHOLD_STATUS=FROZEN_AND_VALID`.
3. Run the v4 campaign in small restartable seed batches.
4. Run `analyze_factor_campaign.py` only after all 150 cells are complete.

