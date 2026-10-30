"""MMT_2D system definition — Majda-McLaughlin-Tabak rank-1 wave model, 2D -> 1D.

Input x = (x1, x2): real/imaginary parts of the rank-1 KLE amplitude xi =
x1 + i*x2.  Declared input law: independent standard normal N(0, I), density
UNtruncated (dnosearch convention).  Design/candidate/LHS domain: [-6, 6]^2;
to_unit/from_unit are LINEAR over that box, so LHS and candidates are uniform
on the domain exactly as in dnosearch examples/nls_MLP (and the MLP consumes
the RAW x, not unit coordinates — see run.py).

Truth: y0 = sqrt(lam_1) * xi * phi_1 (512-point complex field, KLE basis in
data/savedata_phi_lam.mat) integrated by the UNCHANGED original MATLAB ETDRK4
solver (solver/, copied from project mmt_mlp; 512 points, dt=0.001, t=10 then
50x200 steps to t=20, nonlinearity lam=-0.5).  Output y = max_x |Re u(x, t=20)|,
always >= 0, observed range [0.0002, 0.1163].  oracle(x) runs one MATLAB batch
call per invocation (parfor inside when n >= 8).

Data (no interpolation anywhere): the 100x100 uniform grid over [-6,6]^2 was
already truth-solved on the HPC with MATLAB 2020b (~/scratch/mmt2d, sharded,
IC generation bitwise-checked against this machine).  prepare() only assembles
data/evaluation.npz (x_mc/x_grid = the grid, w = N(0,I) input-law weights) and
data/reference_law.npz (weighted quantiles of the grid truth).  Because the
evaluation set is a uniform grid rather than input-law samples, EVERY density
or distribution metric downstream must use the weights w (gaussian_kde with
weights=..., weighted moments) — see run.py / acquisition.py.

Generate the data files once with:

    python problem.py          # writes data/evaluation.npz + data/reference_law.npz

MATLAB executable: env MMATLAB_EXE overrides; default is the local R2026a
install when present, else 'matlab' from PATH (HPC jobs: module load MATLAB).
After that follow README_TODO.md (smoke tests, then PLAYBOOK section 4).
"""

import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

DOMAIN = np.array([[-6.0, 6.0], [-6.0, 6.0]])

# y = max_x |Re u| at t=20 stays in [0, 0.117] on the whole domain; 1.0 is the
# trivially safe upper bound AND keeps the PDF-error window at [0, 1] exactly
# like the original dnosearch nls plotting convention.
SCALE_FIXED = 1.0

MMT_LAM = -0.5  # nonlinearity coefficient of the MMT equation

MATLAB_WIN_DEFAULT = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"


# ----------------------------------------------------------------------------
# Input distribution: declared N(0, I) product density (untruncated); LHS /
# candidates are uniform on DOMAIN via the LINEAR copula below, matching
# dnosearch examples/nls_MLP (candidates uniform in the box, p_X used as the
# US_LW weight and the KDE weights).
# ----------------------------------------------------------------------------
def to_unit(x):
    """Physical x -> box coordinates u in (0,1)^2 (LHS/candidates/dedup only;
    the MLP itself consumes the RAW x)."""
    x = np.asarray(x, dtype=float)
    return (x - DOMAIN[:, 0]) / (DOMAIN[:, 1] - DOMAIN[:, 0])


def from_unit(u):
    u = np.asarray(u, dtype=float)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("unit coordinates must be open (0,1)")
    return DOMAIN[:, 0] + u * (DOMAIN[:, 1] - DOMAIN[:, 0])


def input_pdf(x):
    """p_X in PHYSICAL coordinates: N(0, I) product density, untruncated."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * np.sum(x * x, axis=1)) / (2.0 * np.pi)


def candidate_pdf(x):
    """Density of the candidate/LHS proposal (candidates = from_unit(U(0,1)^2)).
    The LINEAR box map makes the proposal uniform on DOMAIN (1/area).  A
    PIT-copula from_unit would make this identical to input_pdf, in which case
    the USLW importance weights input_pdf/candidate_pdf collapse to 1 (plain
    KDE, the TOY/Poiseuille behaviour).  acquisition.py uses the ratio and
    never needs to know which convention a system chose."""
    x = np.asarray(x, dtype=float)
    return np.full(len(x), 1.0 / np.prod(DOMAIN[:, 1] - DOMAIN[:, 0]))


def net_features(x):
    """Model-input coordinates for the soft (eta) nets: IDENTITY here, i.e.
    the RAW physical x (dnosearch nls_MLP convention).  A copula system would
    return 2*to_unit(x)-1 instead.  Everything model-facing in run.py routes
    through this transform, so a system swap touches problem.py only."""
    return np.asarray(x, dtype=float)


def law_copula(x):
    """PIT copula of the box-truncated input law: x -> v in [0,1]^2 with
    v ~ U(0,1)^2 exactly when x follows the input law.  Used ONLY by the
    hard-PDF template: its probability mesh must MEASURE THE INPUT LAW for
    the exact constraint to target the weighted reference law (TOY's to_unit
    was ndtr for the same reason; Poiseuille's uniform law made the linear
    map identical).  NOT used for LHS/candidates (those stay box-uniform).
    """
    from scipy.special import ndtr
    lo, hi = ndtr(DOMAIN[0, 0]), ndtr(DOMAIN[0, 1])
    x = np.asarray(x, dtype=float)
    return (np.clip(ndtr(x), lo, hi) - lo) / (hi - lo)


def lhs(n=3, seed=101):
    """Reference initial design (run.py re-implements this with --lhs-seed)."""
    rng = np.random.RandomState(seed)
    u = np.column_stack([(rng.permutation(n) + rng.rand(n)) / n for _ in range(2)])
    return from_unit(u)


# ----------------------------------------------------------------------------
# Ground truth: rank-1 KLE initial field + original MATLAB ETDRK4 solver.
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _kle_basis():
    d = loadmat(DATA / "savedata_phi_lam.mat")
    return d["phi"][:, :1], d["lam"][:1]  # rank 1: first KLE mode only


def initial_field(X):
    """(n, 2) -> complex y0 (512, n): y0 = sqrt(lam_1) * (x1 + i x2) * phi_1."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite (n, 2) array of (x1, x2)")
    phi, lam = _kle_basis()
    xi = X[:, 0] + 1j * X[:, 1]
    return np.outer((phi[:, 0] * np.sqrt(lam[0, 0])).ravel(), xi)


def _matlab_exe():
    exe = os.environ.get("MMATLAB_EXE")
    if exe:
        return exe
    if os.path.isfile(MATLAB_WIN_DEFAULT):
        return MATLAB_WIN_DEFAULT
    return "matlab"


def _matlab_quote(path):
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _solve_batch(y0, lam):
    """One MATLAB invocation: y0 (512, n) -> y (n,). Bridge adapted from
    mmt_mlp/problem.evaluate; solver files are unchanged. A valid output.mat
    is the completion signal (MATLAB can hang during shutdown; never wait on
    the process, kill it after the result is verified)."""
    exe = _matlab_exe()
    tmp_root = ROOT / "oracle_tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="call_", dir=str(tmp_root)) as tmp:
        inp, out = Path(tmp) / "input.mat", Path(tmp) / "output.mat"
        savemat(inp, {"y0": y0, "lam": float(lam)})
        script = (f"addpath('{_matlab_quote(ROOT / 'solver')}'); "
                  f"evaluate_mmt('{_matlab_quote(inp)}','{_matlab_quote(out)}'); exit(0);")
        log_path = tmp_root / "matlab.log"
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen([exe, "-batch", script], stdout=log, stderr=log)
            deadline = time.monotonic() + 900
            while True:
                if out.exists():
                    try:
                        cand = loadmat(out)["Y"]
                        if cand.shape == (y0.shape[1], 1) and np.isfinite(cand).all():
                            y = cand.ravel().astype(float)
                            break
                    except (OSError, ValueError, KeyError):
                        pass  # the writer may still be finishing the MAT file
                if proc.poll() is not None:
                    raise RuntimeError(f"MATLAB exited ({proc.returncode}) without "
                                       f"valid output; see {log_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"MATLAB produced no valid output within "
                                       f"900 s; see {log_path}")
                time.sleep(0.5)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("MATLAB output verified; stopping stalled shutdown.", flush=True)
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=30)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return y


def oracle(x):
    """2D -> 1D ground truth: (x1, x2) -> max_x |Re u(x, t=20)| via the
    original MATLAB ETDRK4 solver (one batch invocation per call)."""
    x = np.asarray(x, dtype=float)
    single = x.shape == (2,)
    x = np.atleast_2d(x)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n, 2) array of (x1, x2)")
    tol = 1e-9 * float(np.abs(DOMAIN).max())
    if np.any(x < DOMAIN[:, 0] - tol) or np.any(x > DOMAIN[:, 1] + tol):
        raise ValueError(f"x outside DOMAIN {DOMAIN.tolist()}")
    x = np.clip(x, DOMAIN[:, 0], DOMAIN[:, 1])
    y = _solve_batch(initial_field(x), MMT_LAM)
    return float(y[0]) if single else y


# ----------------------------------------------------------------------------
# Data assembly: evaluation set = remote-solved 100x100 uniform grid with
# input-law weights; reference law = weighted quantiles of the same grid.
# ----------------------------------------------------------------------------
def weighted_quantiles(values, weights, p):
    """Inverse weighted ECDF (midpoint convention), linear in between."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(values, kind="mergesort")
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= cw[-1]
    return np.interp(p, cw, v)


def prepare():
    """Generate data/evaluation.npz and data/reference_law.npz (one-time).

    evaluation.npz  x_mc(10000,2) y_mc — the 100x100 uniform truth grid;
                    x_grid/y_grid — the same set (kept for the unweighted
                    'grid_*' domain metrics); axis — the 100 grid coordinates;
                    w — NORMALIZED N(0,I) input-law weights of the grid nodes.
                    The set is a uniform grid, NOT input-law samples: every
                    PDF/distribution use must pass weights=w.
    reference_law.npz  p (quantile nodes) / q (weighted quantiles of y_mc) /
                    scale — the W1 reference law and z-normalization.
    """
    DATA.mkdir(exist_ok=True)
    theta = np.load(DATA / "mmt_grid_theta.npy")            # (10000, 2)
    raw = np.fromfile(DATA / "mmt_grid_Y.bin", dtype="<f8")  # complex128, re/im interleaved
    y = raw[0::2].copy()
    if len(theta) != len(y) or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("mmt_grid_Y.bin / mmt_grid_theta.npy inconsistent or non-finite")
    w = input_pdf(theta)
    w = w / w.sum()
    axis = np.unique(theta[:, 0])
    np.savez_compressed(DATA / "evaluation.npz",
                        x_mc=theta, y_mc=y, x_grid=theta.copy(), y_grid=y.copy(),
                        axis=axis, w=w)
    p = np.unique(np.r_[np.linspace(0.0, 1.0, 8193),
                        np.geomspace(1e-7, 0.01, 128),
                        1 - np.geomspace(1e-7, 0.01, 128)])
    q = weighted_quantiles(y, w, p)
    keep = np.r_[True, np.diff(q) > 0]
    p, q = p[keep], q[keep]
    scale = float(SCALE_FIXED)
    np.savez_compressed(DATA / "reference_law.npz", p=p, q=q, scale=scale)
    qshow = weighted_quantiles(y, w, [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    print(f"evaluation: 10000 grid points, y in [{y.min():.6g}, {y.max():.6g}]")
    print(f"weighted quantiles: {np.round(qshow, 6).tolist()}")
    print(f"weights: min {w.min():.3e} max {w.max():.3e} (ratio {w.max()/w.min():.3e})")
    print(f"reference law: {len(p)} knots, scale={scale:.6g}")
    print("Wrote data/evaluation.npz and data/reference_law.npz")


if __name__ == "__main__":
    prepare()
