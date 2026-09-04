# V5.1 Installed-Pair Validation Reporting Correction

The validation logic already confirmed:
- pseudo-only estimator reliability was false;
- pseudo-only rank was below the 491-state dimension.

The original displayed rank was overwritten by a later full-design solve.
The corrected validator now captures the pseudo-only rank immediately.

Corrected pseudo-only rank: 489

Implementation modified by this correction: False
Performance outcomes inspected: False