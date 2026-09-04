# Verification record

Date: 2026-08-29 UTC

- Python syntax compilation: passed for all four repaired Python modules and
  the regression suite.
- Regression suite: 7/7 passed.
- Compose YAML parsing: passed; six services resolved.
- Regression coverage:
  - event trust uses minimum `T` and maximum `s`;
  - coverage uses minima and solver/held fields use explicit semantics;
  - the combined event score is the maximum of the stepwise combined score;
  - event-level calibration input is rejected;
  - calibration v1 is rejected and v2 is accepted;
  - missing physical `z_true` fails closed in production;
  - physical `z_true` is accepted and the smoke fallback requires opt-in.

Docker/HELICS execution was not rerun in the review environment because the
Docker CLI and the user's runtime image are not present there. Run the bundled
tests and a new smoke test on the experiment host after installing the patch.
