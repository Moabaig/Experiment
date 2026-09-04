# Paper 1 campaign-ready validation report

Validation date: 2026-09-01 (UTC)

## Outcome

The v11 Paper-1 manuscript and publication-results pipeline are ready for the
predeclared 150-cell campaign. The checks below validate code behavior,
diagnostic rendering, manuscript consistency, and LaTeX syntax. They do not
replace the missing confirmatory experiment and do not authorize performance
claims.

## Executed checks

| Check | Result | Evidence |
|---|---:|---|
| Python syntax compilation | Pass | Analyzer, unit tests, threshold freezer, and v4 design verifier compiled without error |
| Analyzer unit tests | 7/7 pass | Tied-score AUC, monotone ROC, tied-pair `q`, Holm correction, partial-campaign analysis, fail-closed rendering, and complete renderer |
| Diagnostic publication generation | Pass | Terminal marker `PAPER1_DIAGNOSTIC_PUBLICATION_OUTPUT_OK` |
| Static LaTeX audit | Pass | 78 labels, 252 references, 57 bibliography entries, and 117 citation uses; no duplicate labels, unresolved static references/citations, unused bibliography entries, placeholder macros, or obsolete Simscape/pandapower stack terms |
| Two-pass LaTeX syntax smoke test | Pass | 29-page PDF produced without LaTeX errors, undefined references, or multiply-defined labels |
| Confirmatory evidence gate | Pending by design | Seed indices 2--31 crossed with five bandwidth levels have not been supplied in this workspace |

## LaTeX qualification

The available runtime did not provide `IEEEtran.cls`. For syntax and reference
validation only, the exact v11 manuscript was compiled twice with a temporary
article-compatible class shim. The shim was removed after the test. Therefore,
the check establishes source-level compilability but not final IEEE pagination
or typesetting. A final two-pass compile with the real IEEEtran distribution is
required on the experiment host or submission system.

The article-shim PDF and logs are retained under `validation_artifacts/` for
audit only. They are deliberately excluded from the submission archive and
must not be treated as the final paper PDF.

## Evidence boundary

The existing diagnostic outputs support mechanism claims about anisotropic
information geometry, perfect-telemetry reduction, measurement-identity
sensitivity, the observability cliff, and availability--exposure collinearity.
They do not establish recall improvement, matched-FAR superiority, latency
improvement, false-alarm transport, or robustness across physical seeds.

Final detector-performance prose and figures may be used only after the
analyzer writes a manifest satisfying all of the following:

```text
status = confirmatory_complete
cells_found = 150
qualification_seed_excluded = true
```

The exact execution and interpretation sequence is recorded in
`PAPER1_CAMPAIGN_HANDOFF.md`.
