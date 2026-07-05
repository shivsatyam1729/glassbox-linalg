# glassbox-linalg

Transparent, from-scratch implementations of classical linear algebra algorithms — see what's actually inside QR, SVD, and least squares instead of treating LAPACK as a black box.

This isn't meant to replace `numpy.linalg` or `scipy.linalg` — those are faster, more robust, and battle-tested. **This is a reference and benchmarking tool**: if you're implementing your own QR decomposition, eigenvalue solver, or SVD, you can validate correctness against these implementations and read through code that doesn't hide behind a single `lapack_*` call.

## Why this exists

Most linear algebra libraries call into LAPACK, which is fast but opaque — you can't easily see *why* an eigenvalue solver converges slowly, or what a Householder reflector is actually doing to a matrix. Every algorithm here is implemented directly in NumPy, with comments explaining the non-obvious parts (numerical stability tricks, convergence acceleration, etc.), so you can step through the actual math instead of a compiled binary.

## What's implemented

- **Householder QR decomposition** — `solvers/power_qr_svd.py`
- **Symmetric QR algorithm with Wilkinson shift + deflation** for eigenvalue decomposition — converges in ~10s of iterations instead of hundreds, even for tightly clustered eigenvalue spectra
- **SVD** via eigendecomposition of `AᵀA` (no bidiagonalization — intentionally simple, not the fastest approach)
- **Least squares** solved via the SVD above (`lstsq`)

More algorithms may be added over time (see [Roadmap](#roadmap)).

## Installation

```bash
git clone https://github.com/<your-username>/glassbox-linalg.git
cd glassbox-linalg
pip install -r requirements.txt
```

## Usage

```python
from solvers.power_qr_svd import svd, lstsq
import numpy as np

X = np.random.randn(100, 10)
U, s, Vt = svd(X)

y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
beta = lstsq(X, y)
```

## Benchmarking your own implementation

`benchmark.py` runs a given solver against NumPy's equivalent and reports error metrics (reconstruction error for SVD, prediction/beta error for least squares, etc.). It's invoked as:

```bash
python benchmark.py [solver_name] [benchmark_label]
```

- `solver_name` — the module in `solvers/` to test (e.g. `power_qr_svd`)
- `benchmark_label` — which benchmark to run against that solver (e.g. `svd`, `lstsq`)

To validate your own implementation:

1. Drop your solver into `solvers/` with the same function interface (e.g. a `svd(A) -> (U, s, Vt)` signature).
2. Run `python benchmark.py your_solver_name svd` (or whichever benchmark label applies).
3. Compare error metrics and iteration counts against the reference implementation.


If your implementation disagrees with the reference by more than floating-point noise, that's usually a sign of either a correctness bug or an under-converged iterative step (see the Wilkinson shift notes in `power_qr_svd.py` for an example of the latter).

## Project structure

```
glassbox-linalg/
├── solvers/
│   └── power_qr_svd.py   # Householder QR, symmetric QR eigensolver, SVD, least squares
├── linalg/
│   └── norm.py           # shared numerical utilities
├── benchmark.py           # runs implementations against NumPy references
└── README.md
```

## Roadmap

- [ ] Bidiagonalization-based SVD (faster than the current `AᵀA`-based approach)
- [ ] LU decomposition with partial pivoting
- [ ] Cholesky decomposition
- [ ] Conjugate gradient solver

## Contributing

PRs welcome — especially additional reference implementations, edge-case tests, or clearer explanatory comments. If you add an algorithm, please also add a corresponding entry in `benchmark.py` comparing against NumPy/SciPy.

## License

MIT
