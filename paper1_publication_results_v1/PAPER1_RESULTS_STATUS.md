# Paper 1 results status and citation boundary

## Manuscript and analysis readiness

The campaign-ready manuscript is
`Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.tex`. Its
mathematical formulation, actual OpenDSS--HELICS--ns-3 implementation,
diagnostic results, limitations, and reproducibility protocol are complete.
The v10 manuscript is retained for provenance only and must not be submitted.

The final analyzer is also complete and tested. When, and only when, all 150
predeclared cells pass the evidence gate, it generates the confirmatory
results section plus numerical abstract, discussion, and conclusion fragments
from the same seed-cluster summaries. This prevents values in the manuscript
from drifting away from the archived analysis tables.

## What can enter the paper now

The following are completed mechanism-characterization results:

| Result | Value | Permissible claim |
|---|---:|---|
| Perfect-information `lambda_min(G_inf)` | 5.6876895 | Numerical reduction-limit characterization |
| Perfect-information condition number | 4.4021 x 10^9 | The feeder has a highly anisotropic information geometry |
| Perfect-information exposures | `u_lmax = u_trace = 0` | Numerical implementation satisfies the perfect-telemetry reduction check |
| Spearman availability versus exposure | -0.7414 | Strong collinearity exists and must be checked before interpreting F3 |
| Arm-C events | 300 | Mechanism identity is tested under matched communication strata |
| Near-unobservable Arm-C fraction | 10%, 30%, 53% across strata | Measurement identity changes observability within fixed count/age strata |
| Top single-channel weak-direction loss | about 21.6% | Individual measurement identity can materially affect the weakest direction |
| Retained information after top-eight removal | about 4.86 x 10^-13 | A clear observability cliff exists for coordinated weak-axis loss |

These values do not establish detector superiority, recall improvement,
latency improvement, or false-alarm stability.

## What remains unavailable

The following require the complete seed-81002--81031 campaign:

- F1 residual-silence rate with a physical-seed cluster interval;
- recall at exactly the frozen, matched calibration FAR;
- F2 seed-paired AUC differences by bandwidth;
- detection-by-deadline on residual-silent positives;
- FAR transport as realized coverage and age worsen;
- F3 Arm-G full-versus-hybrid geometry premium;
- robustness and between-seed heterogeneity;
- final F1/F2/F3 pass/fail interpretation.

Seed 81001 remains a qualification result only. Its unfavorable outcome must
not be pooled with, substituted for, or used to tune the confirmatory campaign.

Accordingly, Paper 1 is **manuscript-complete and campaign-ready**, but it is
not yet **empirically final**. No claim of detector superiority should enter a
submission until the confirmatory manifest reports `confirmatory_complete`,
`cells_found = 150`, and `qualification_seed_excluded = true`.

## Interpretation rule after the campaign

1. Report F1, F2, and F3 independently.
2. Call the geometry term incrementally useful only if the Arm-G comparison
   against residual-plus-B1/B2 is positive under paired seed inference.
3. Report the B1--`u` Spearman relationship and `q` before interpreting a null
   F3 result.
4. Treat Arm-C success as a mechanism demonstration, not an operational F3
   pass.
5. Retain unfavorable or null results; do not recalibrate on campaign output.
