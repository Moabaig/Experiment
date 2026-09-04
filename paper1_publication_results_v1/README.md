# Paper 1 publication-results package

This package converts the frozen Paper-1 experiment into reproducible,
LaTeX-ready results. It does **not** alter the v4 design, detector, calibration,
gamma, event window, or calibration-only thresholds.

## Evidence status at package creation

- Complete and usable now: static feeder/geometry diagnostics, perfect-
  telemetry reduction check, Arm-C matched-count/age mechanism check, uniform
  age/loss response, and channel-identity sensitivity.
- Complete but excluded from confirmatory inference: physical seed 81001
  qualification result. It remains a sealed, unfavorable qualification result.
- Not available in this workspace: the 150 confirmatory cells for seed indices
  2--31 at five bandwidth levels. No final Paper-1 detector-performance value
  has been invented or inferred from the qualification seed.

The generator writes vector PDF and 300-dpi PNG versions of each figure. The
PDF files are the preferred LaTeX inputs.

## Generate the completed diagnostic figures

From the experiment-bundle root, after copying this directory to
`paper1_publication_results_v1`:

```powershell
docker compose --profile cosim run --rm --no-deps dev `
    python /workspace/paper1_publication_results_v1/paper1_final_results.py `
    diagnostic `
    --static-results /workspace/paper1_publication_results_v1/diagnostic_inputs `
    --output /workspace/paper1_publication_results_v1/paper1_generated
```

The current package already contains those generated diagnostic figures and
the fragment:

```text
paper1_generated/latex/paper1_diagnostic_results.tex
```

The fragment inserts the two core mechanism figures and the Arm-C table. Two
additional publication-ready files are supplied for optional use in the main
paper or supplement:

- `fig_diagnostic_channel_influence.pdf` (single and cumulative weak-axis
  sensitivity); and
- `fig_diagnostic_collinearity.pdf` (availability--exposure relationship by
  arm).

## Generate the final 30-seed results

Run this only after all `paper1_s002_*` through `paper1_s031_*` directories
exist and the matched-FAR threshold file has been frozen from calibration-only
data:

```powershell
docker compose --profile cosim run --rm --no-deps dev `
    python /workspace/paper1_publication_results_v1/paper1_final_results.py `
    confirmatory `
    --runs-root /workspace/runs `
    --design /workspace/factor_design.paper1.v4.json `
    --thresholds /workspace/paper1_matched_far_thresholds.v1.json `
    --static-results /workspace/paper1_publication_results_v1/diagnostic_inputs `
    --output /workspace/paper1_publication_results_v1/paper1_generated `
    --confirmatory-seeds 2-31 `
    --bootstrap-draws 2000
```

The command is fail closed. Without `--allow-partial`, it refuses final
rendering unless:

1. all 150 seed/bandwidth cells exist;
2. every cell has complete power, network, twin, and oracle metadata;
3. every cell's v4 provenance record matches the frozen design and threshold
   SHA-256 hashes;
4. event and step cardinalities are 1,100 and 13,200, respectively;
5. seed 81001 is absent from confirmatory inference; and
6. the thresholds are the frozen calibration-only 1% FAR thresholds.

`--allow-partial` is suitable only for campaign progress checks. Such output is
marked `partial_nonconfirmatory`, and the script refuses to create the final
LaTeX performance fragment.

## Final outputs

When the 150-cell gate passes, the generator creates:

- `paper1_seed_bandwidth_metrics.csv`: seed-level AUC, recall, FAR, precision,
  residual silence, and abstention;
- `paper1_condition_metrics.csv`: marginal and joint results by arm, regime,
  and drift family;
- `paper1_paired_metric_contrasts.csv`: paired seed contrasts with 95% cluster-
  bootstrap intervals, Wilcoxon tests, and Holm correction;
- `paper1_arm_paired_contrasts.csv`: Arm-G F3 contrasts against B1/B2 hybrids;
- `paper1_latency_events.csv`: latency with non-detections retained and a flag
  for residual-silent positives;
- `paper1_collinearity_and_q.csv`: the pre-interpretation B1--geometry
  collinearity check, reproducible tied-pair fraction `q`, and the associated
  B1 AUC ceiling;
- `paper1_network_conditions.csv`: realized coverage, age, drops, and holds;
- descriptive ROC points plus seed-clustered summary tables;
- publication PDF/PNG figures;
- `paper1_generated/latex/paper1_confirmatory_results.tex`;
- numerical abstract, discussion, and conclusion fragments generated from
  the same seed-cluster summaries; and
- `paper1_confirmatory_headline.json`, a machine-readable record of every
  value inserted into the narrative.

Pooled ROC curves are descriptive. The primary inference is the paired
physical-seed AUC or matched-FAR difference; the event is never replaced by the
timestep as the analysis unit. DeLong can be reported as a secondary pooled-
event sensitivity check, not as the primary test.

## Manuscript integration

`Communication-Aware_Trust_Metric_PAPER_v11_campaign_ready.tex` uses
`\IfFileExists` gates:

- the completed diagnostic fragment is inserted now;
- the confirmatory fragment is inserted only after final analysis succeeds;
- confirmatory abstract, discussion, and conclusion sentences are inserted
  from generated evidence rather than hand-entered values;
- otherwise the paper states explicitly that confirmatory results are pending.

The v11 methods section documents the executed OpenDSS truth export and
four-federate HELICS/ns-3 stack. The v10 file is retained only as provenance
and must not be submitted.

Keep the generated directory adjacent to the `.tex` file. Compile with an IEEE
LaTeX installation that provides `IEEEtran.cls`.

## Verification

```powershell
docker compose --profile cosim run --rm --no-deps dev `
    python -m unittest discover `
    -s /workspace/paper1_publication_results_v1 `
    -p "test_paper1_final_results.py" `
    -v
```

The tests cover tied-score AUC, ROC monotonicity, the reproducible `q`
definition, Holm correction, the raw partial-campaign path, refusal to render
partial results as final, and the complete publication renderer.
