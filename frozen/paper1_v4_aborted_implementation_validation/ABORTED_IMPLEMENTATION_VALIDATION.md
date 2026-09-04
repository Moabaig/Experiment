# Paper 1 v4 Campaign Disposition

Status: ABORTED_IMPLEMENTATION_VALIDATION

The v4 campaign must not be used as the final confirmatory Paper-1 campaign.

Reason:
The estimator's absolute candidate-norm guard rejected physically valid state estimates during bootstrap. The initial acceptance limit was approximately 11.0793, whereas the physical state norm was approximately 30.8. Oracle cells therefore retained the previous state at every step despite approximately 98.3% exact solver success.

Additional concern:
Local pseudo-measurements were recorded as fresh and received, requiring explicit separation between numerical solvability and external measurement support.

Performance outcomes were not inspected before identifying this implementation defect.

Required successor:
A versioned v5 implementation, new mechanical validation, new calibration, new threshold freeze, and a new confirmatory campaign.
