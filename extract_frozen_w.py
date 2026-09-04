from pathlib import Path
import numpy as np

source = Path("/workspace/truth.calibration.npz")
output = Path("/workspace/W.frozen.v2.npy")

with np.load(source, allow_pickle=False) as data:
    assert "W" in data.files, "Source truth has no W array"
    W = np.asarray(data["W"], dtype=np.float64)

assert W.shape == (491, 491), W.shape
assert np.isfinite(W).all()

symmetry_error = float(np.max(np.abs(W - W.T)))
assert symmetry_error <= 1e-8, symmetry_error

minimum_eigenvalue = float(np.linalg.eigvalsh(W).min())
assert minimum_eigenvalue >= -1e-8, minimum_eigenvalue

np.save(output, W, allow_pickle=False)

print(
    "FROZEN_W_OK",
    "shape=", W.shape,
    "symmetry_error=", symmetry_error,
    "min_eigenvalue=", minimum_eigenvalue,
    "trace=", float(np.trace(W)),
)
