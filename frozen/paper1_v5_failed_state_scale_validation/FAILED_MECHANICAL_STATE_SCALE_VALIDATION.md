# Paper 1 V5 Estimator Disposition

Status: FAILED_MECHANICAL_STATE_SCALE_VALIDATION

The V5 estimator must not be used for calibration, qualification, or confirmatory simulation.

Mechanical evidence:

- Oracle steps: 13,200
- Numerically exact solves: 98.2955%
- Accepted state updates: 94.7348%
- Solve-inexact holds: 225
- Jump-guard holds: 470
- Bootstrap candidate norm: 1206.2694
- Physical truth-state norm: approximately 30.8
- Accepted candidate median norm: approximately 1.8485e7
- Accepted candidate maximum norm: approximately 1.2601e39
- Longest consecutive jump-guard sequence: 60 steps

Root-cause direction:

The estimator forms weighted normal equations and treats a non-throwing numerical solve as exact. The bootstrap accepts any finite solution. Subsequent jump limits scale from the previous accepted norm, permitting runaway expansion.

Required successor:

A V5.1 candidate using a numerically stable weighted least-squares implementation, explicit numerical-rank and conditioning diagnostics, and a bootstrap/state-validity rule anchored independently of the previous estimate.

Performance outcomes were not inspected.

The existing V5 oracle run remains preserved as failed mechanical-validation evidence.