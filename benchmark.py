import time
import argparse
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

"""
How far can textbook solvers take you without LAPACK, compared to
np.linalg.lstsq (SVD-based)?

Three stages, each building on the winner of the last:
  1. Scale n_samples up (fixed n_features, noise) until error blows up
     or a trial gets too slow to be worth continuing.
  2. Freeze n_samples at the best value found, scale n_features up.
  3. Freeze n_samples & n_features at their best values, scale noise up.

Noise is added to y, not X, so it shouldn't move the numerical-error
needle at all -- stage 3 is really a sanity check on that claim.
"""

parser = argparse.ArgumentParser()
parser.add_argument("solver_name", help="e.g. gsbs, gsgj, hhbs")
parser.add_argument("label", help="e.g. MGS-QR")
args = parser.parse_args()

try:
    module = importlib.import_module(f"solvers.{args.solver_name}")
except ModuleNotFoundError:
    print(f"Use gsbs or gsgj... Can't find {args.solver_name}!")
    raise SystemExit(1)   

ERROR_THRESHOLD = 1e-6   # beyond this, we call custom solver "problematic"
TIME_BUDGET     = 5.0    # seconds; stop scaling a stage once a trial is this slow

def run_trial(n_samples, n_features, noise, random_state=42):
    X, y, coef = make_regression(
        n_samples=n_samples, n_features=n_features,
        noise=noise, coef=True, random_state=random_state,
    )

    t0 = time.perf_counter()
    beta_qr = module.lstsq(X, y)
    t_qr = time.perf_counter() - t0

    t0 = time.perf_counter()
    beta_np = np.linalg.lstsq(X, y, rcond=None)[0]
    t_np = time.perf_counter() - t0

    err = np.max(np.abs(beta_qr - beta_np))
    return err, t_qr, t_np


results = []


def log(stage, n_samples, n_features, noise, err, t_qr, t_np):
    results.append(dict(
        stage=stage, n_samples=n_samples, n_features=n_features, noise=noise,
        max_abs_error=err, time_qr=t_qr, time_np=t_np,
    ))
    print(f"[{stage:>10}] n={n_samples:>7} p={n_features:>5} noise={noise:>6} "
          f"| err={err:.3e}  t_qr={t_qr:6.3f}s  t_np={t_np:6.3f}s")


# --- Stage 1: Scale n_features ---

n_features_fixed = 10
noise_fixed = 10.0
n_samples_candidates = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
best_n_samples = n_samples_candidates[0]
for n in n_samples_candidates:
    err, t_qr, t_np = run_trial(n, n_features_fixed, noise_fixed)
    log("n_samples", n, n_features_fixed, noise_fixed, err, t_qr, t_np)
    if err >= ERROR_THRESHOLD:
        print(f"  -> error threshold exceeded at n_samples={n}, stopping stage 1")
        break
    best_n_samples = n
    if t_qr >= TIME_BUDGET:
        print(f"  -> time budget exceeded at n_samples={n}, stopping stage 1")
        break

print(f"\nStage 1 result: best_n_samples (pure scaling test) = {best_n_samples}\n")

# ---------------------------------------------------------------------
# Stage 2: scale n_features.
#
# Note: we do NOT reuse the full best_n_samples from stage 1 here. Each
# n_features trial costs O(n_samples * n_features^2), and at n_samples in
# the hundreds-of-thousands range even p=25 already took ~8s -- we'd
# never get to explore large p within a reasonable runtime. What actually
# stresses Modified Gram-Schmidt's orthogonality loss is p approaching n
# (i.e. an ill-conditioned/near-square design matrix), not n by itself.
# So we cap n_samples here to something that lets us push p close to n.
# ---------------------------------------------------------------------
n_samples_stage23 = min(best_n_samples, 5000)
noise_fixed = 10.0
n_features_candidates = [10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 4500, 4800, 4950, 4990]
n_features_candidates = [p for p in n_features_candidates if p < n_samples_stage23]
best_n_features = n_features_fixed
for p in n_features_candidates:
    err, t_qr, t_np = run_trial(n_samples_stage23, p, noise_fixed)
    log("n_features", n_samples_stage23, p, noise_fixed, err, t_qr, t_np)
    if err >= ERROR_THRESHOLD:
        print(f"  -> error threshold exceeded at n_features={p}, stopping stage 2")
        break
    best_n_features = p
    if t_qr >= TIME_BUDGET:
        print(f"  -> time budget exceeded at n_features={p}, stopping stage 2")
        break

print(f"\nStage 2 result: best_n_features = {best_n_features} "
      f"(n_samples held at {n_samples_stage23})\n")

# --- Stage 3: scale noise, n_samples & n_features frozen at stage 2's values ---
noise_candidates = [0, 1, 10, 50, 100, 500, 1000, 5000]

for nz in noise_candidates:
    err, t_qr, t_np = run_trial(n_samples_stage23, best_n_features, nz)
    log("noise", n_samples_stage23, best_n_features, nz, err, t_qr, t_np)

print("\nStage 3 done (noise shouldn't move the error at all, since it only "
      "perturbs y, not the conditioning of X).\n")

# --- Save results ---
df = pd.DataFrame(results)
df.to_csv(f"{args.label}_benchmark_results.csv", index=False)
print(f"Saved results to {args.label}_benchmark_results.csv")

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

stage_titles = {
    "n_samples": "Stage 1: scaling n_samples",
    "n_features": "Stage 2: scaling n_features",
    "noise": "Stage 3: scaling noise",
}
stage_xcols = {"n_samples": "n_samples", "n_features": "n_features", "noise": "noise"}

for col_idx, stage in enumerate(["n_samples", "n_features", "noise"]):
    sub = df[df.stage == stage]
    x = sub[stage_xcols[stage]]

    ax_err = axes[0, col_idx]
    ax_err.plot(x, sub.max_abs_error, marker="o", color="crimson")
    ax_err.axhline(ERROR_THRESHOLD, color="gray", linestyle="--", linewidth=1,
                    label=f"threshold ({ERROR_THRESHOLD:g})")
    ax_err.set_yscale("log")
    if stage != "noise":
        ax_err.set_xscale("log")
    ax_err.set_title(stage_titles[stage])
    ax_err.set_xlabel(stage_xcols[stage])
    ax_err.set_ylabel("max |beta_qr - beta_np|")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)

    ax_t = axes[1, col_idx]
    ax_t.plot(x, sub.time_qr, marker="o", label=f"{args.label}", color="steelblue")
    ax_t.plot(x, sub.time_np, marker="s", label="np.linalg.lstsq (SVD)", color="darkorange")
    ax_t.set_yscale("log")
    if stage != "noise":
        ax_t.set_xscale("log")
    ax_t.set_xlabel(stage_xcols[stage])
    ax_t.set_ylabel("time (s)")
    ax_t.legend(fontsize=8)
    ax_t.grid(alpha=0.3)

fig.suptitle(f"How far did {args.label} take you?", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{args.label}_benchmark_plot.png", dpi=150)
print(f"Saved plot to {args.label}_benchmark_plot.png")