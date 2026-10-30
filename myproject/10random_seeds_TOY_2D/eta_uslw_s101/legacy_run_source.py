"""Faithful reproduction of the mil NN_US_LW active-learning run inside POIS_2D.

This is the lambda=0 (pure MSE) degenerate case of our eta-AL pipeline,
reproducing mil_2d1d_tReTaoMean/results/comparison_20261023_094736_12401678/
NN_US_LW exactly:

  - initial design : 3-point LHS, np.random.RandomState(101), unit =
      column_stack([(rng.permutation(n) + rng.rand(n)) / n for each dim])
  - ensemble       : mlp_repro.MLPEnsemble (local line-for-line copy of the
      mil EnsembleMLP), layers [2, 200x5, 1] ReLU, seeded member init
      (RandomState(seed), glorot std, clip +-2std, zero bias), round-k member
      seeds [2k-1, 2k] (config member_seed_schedule = "sequential_pairs")
  - training       : full-batch Adam(lr 1e-3, eps 1e-8), 1000 steps, final
      model (no early/best selection), inputs scaled to [-1,1], target
      z = y/300, torch cpu 1 thread, deterministic algorithms
  - acquisition    : US_LW (variance * p_X / KDE(ensemble mean), verbatim
      copy), candidates = RandomState(len(y)) 1000 points, normalized-distance
      dedup > 1e-8
  - error          : e_log10_pdf of the ensemble-mean KDE on the SAME fixed
      100,000 MC set against the SAME truth KDE as mil (both from the local
      copy data/mc_truth_milref.npz)

Truth values come from POIS_2D problem.py (identical code and byte-identical
CSV to mil, sha256 fd9b75...). The mil project itself is never modified.

Usage: python run_pois_mil_repro.py [--iters 100] [--smoke]   (resumable)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde

from mlp_repro import MLPEnsemble
from problem import evaluate, normalize_y, denormalize_y, DOMAIN, Y_MAX

PROJ = Path(__file__).resolve().parent
MIL_TRUTH = PROJ / "data" / "mc_truth_milref.npz"   # local copy of mil's truth artifact

GRID = np.linspace(0.0, 1.0, 1000)
FLOOR = 1e-16
RIDGE = (0.45, 0.85)          # mil's ridge band in theta (90% of its samples)


def scale_x(x):
    return 2 * (np.asarray(x) - DOMAIN[:, 0]) / np.diff(DOMAIN, axis=1).ravel() - 1


def kde(values, grid):
    """Exact SciPy Gaussian KDE, Scott bandwidth; original density floor."""
    values = np.asarray(values, dtype=float).ravel()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("KDE requires at least two finite values")
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, np.abs(values).max()):
        h = 1e-6 * max(1.0, abs(values[0]))
        result = np.exp(-0.5 * ((np.asarray(grid) - values[0]) / h) ** 2) / (np.sqrt(2 * np.pi) * h)
    else:
        result = gaussian_kde(values)(grid)
    return np.maximum(result, FLOOR)


def pdf_error(predicted, truth):
    """Base-10 log-density error, left rectangles (mil pdf_workflow verbatim)."""
    delta = np.abs(np.log10(predicted) - np.log10(truth))[:-1]
    return float(delta.sum() * (GRID[1] - GRID[0]))


def acquisition_uslw(mean, variance):
    """US_LW copied verbatim from mil pdf_workflow.py / run_pois_al.py."""
    mu = mean.ravel()
    padding = 1e-6 * np.ptp(mu) + 1e-12
    grid = np.linspace(mu.min() - padding, mu.max() + padding, 10000)
    density = kde(mu, grid)
    p_x = 1 / np.prod(np.diff(DOMAIN, axis=1))
    return variance.ravel() * p_x / np.maximum(np.interp(mu, grid, density), FLOOR)


def predict(model, x):
    """Physical seconds mean/var in bounded batches, as in mil predict()."""
    means, variances = [], []
    for start in range(0, len(x), 4096):
        mean, var = model.predict_noiseless(scale_x(x[start:start + 4096]))
        means.append(mean)
        variances.append(var)
    mean, var = np.vstack(means), np.vstack(variances)
    if not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise FloatingPointError("Nonfinite surrogate prediction")
    return mean, var


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="2 iterations, wiring check")
    args = parser.parse_args()
    iters = 2 if args.smoke else args.iters

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    out = args.out or (PROJ / "results" / ("milrepro_smoke" if args.smoke else "milrepro_uslw"))
    out.mkdir(parents=True, exist_ok=True)

    # ---- fixed MC set + truth KDE, shared with mil (READ-ONLY) ----
    with np.load(MIL_TRUTH, allow_pickle=False) as data:
        mc, truth_pdf = data["X"].copy(), data["pdf_z"].copy()
        assert np.array_equal(data["z_grid"], GRID), "truth grid mismatch"
    meta = {"mil_truth": str(MIL_TRUTH), "lhs_seed": 101, "n_init": 3,
            "layers": [2, 200, 200, 200, 200, 200, 1], "lr": 1e-3, "steps": 1000,
            "members": 2, "seed_schedule": "sequential_pairs [2k-1,2k]",
            "candidate_count": 1000, "acquisition": "US_LW", "lambda": 0.0,
            "torch": torch.__version__, "numpy": np.__version__}
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- initial design: 3-point LHS, seed 101 (mil pdf_workflow verbatim) ----
    rng = np.random.RandomState(101)
    n = 3
    unit = np.column_stack([(rng.permutation(n) + rng.rand(n)) / n for _ in range(2)])
    x = DOMAIN[:, 0] + unit * np.diff(DOMAIN, axis=1).ravel()
    y = evaluate(x)
    print(f"LHS init (seed 101): {np.round(x, 5).tolist()} -> t = {y.ravel().round(1).tolist()}",
          flush=True)

    # ---- resume support ----
    done = sorted(int(p.stem[4:]) for p in out.glob("iter*.npz") if p.stem[4:].isdigit())
    if done:
        last = np.load(out / f"iter{done[-1]:04d}.npz")
        x, y = last["X"].copy(), last["Y"].copy()
        print(f"resume from iteration {done[-1]} (n={len(x)})", flush=True)
    start = (done[-1] + 1) if done else 1

    hist_path = out / "history.csv"
    if not hist_path.exists() or not done:
        with hist_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["iteration", "n_train", "e_log10_pdf",
                                    "q_A", "q_theta", "q_t", "in_ridge"])

    for k in range(start, iters + 1):
        t0 = time.time()
        seeds = [2 * k - 1, 2 * k]                       # sequential_pairs
        model = MLPEnsemble(
            scale_x(x), y, [2, 200, 200, 200, 200, 200, 1],
            1e-3, 1000, 2, normalize_y, denormalize_y,
            member_seeds=seeds, device="cpu")

        mc_mean, _ = predict(model, mc)
        density = kde(normalize_y(mc_mean), GRID)
        e = pdf_error(density, truth_pdf)

        candidates = DOMAIN[:, 0] + np.random.RandomState(len(y)).rand(1000, 2) * \
            np.diff(DOMAIN, axis=1).ravel()
        distance = np.linalg.norm(
            (candidates[:, None] - x[None, :]) / np.diff(DOMAIN, axis=1).ravel(), axis=2)
        candidates = candidates[distance.min(axis=1) > 1e-8]
        mean, variance = predict(model, candidates)
        scores = acquisition_uslw(mean, variance)
        index = int(np.argmax(scores))
        query = candidates[index:index + 1]
        observed = evaluate(query)

        np.savez_compressed(out / f"iter{k:04d}.npz",
                            X=x, Y=y, query=query, observed=observed, e=e,
                            member_seeds=np.asarray(seeds))
        with hist_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([k, len(x), f"{e:.9g}",
                                    f"{query[0, 0]:.6g}", f"{query[0, 1]:.6g}",
                                    f"{observed[0, 0]:.6g}",
                                    int(RIDGE[0] <= query[0, 1] <= RIDGE[1])])
        x, y = np.vstack((x, query)), np.vstack((y, observed))
        print(f"iter {k:3d}/{iters} n={len(x)} e={e:.5g} "
              f"query=({query[0,0]:.4f},{query[0,1]:.3f}) t={observed[0,0]:.1f} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    np.savez_compressed(out / "all_observations.npz", X=x, Y=y)
    print(f"done: {len(x)} total samples, artifacts in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
