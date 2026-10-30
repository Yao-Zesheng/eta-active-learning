"""pois_transion_2D system definition — Poiseuille transition time, 2D -> 1D.

Input x = (A_3d, theta_rel_rad) on [[0.001, 0.06], [0, pi]] with theta
pi-periodic; output y = t_transition / 300 (t clipped to [0, 300] s) obtained
by bicubic spline interpolation of the t_RetaoMean_576.csv grid. Generate the
data files once with:

    python problem.py          # writes data/evaluation.npz + data/reference_law.npz

After that follow README_TODO.md (smoke tests, then PLAYBOOK_10random_seeds.md
section 4 for the full ten-seed run).
"""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CSV_FILE = DATA / "t_RetaoMean_576.csv"
T_MIN, T_MAX = 0.0, 300.0  # physical transition-time bounds [s]

# Input x = (A_3d, theta_rel_rad); theta = pi is physically identical to
# theta = 0, so the domain includes the periodic endpoint.
DOMAIN = np.array([[0.001, 0.06], [0.0, np.pi]])

# y = t_transition / T_MAX with t in [0, 300] s  ->  y in [0, 1]; 1.0 is the
# known safe upper bound (z = y / scale).
SCALE_FIXED = 1.0


@lru_cache(maxsize=1)
def _load_spline():
    """Build the bicubic spline of t_RetaoMean over the (A, theta) CSV grid."""
    with CSV_FILE.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        cols = ("A_3d", "theta_rel_rad", "t_RetaoMean")
        if not set(cols).issubset(reader.fieldnames or []):
            raise ValueError(f"{CSV_FILE.name} must contain columns {cols}")
        data = np.asarray([[float(row[c]) for c in cols] for row in reader])
    if not np.isfinite(data).all() or data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("CSV data must be nonempty and finite")
    amplitude, angle = np.unique(data[:, 0]), np.unique(data[:, 1])
    if len(data) != len(amplitude) * len(angle):
        raise ValueError("CSV must form a complete rectangular (A, theta) grid")
    values = np.empty((len(amplitude), len(angle)))
    values[np.searchsorted(amplitude, data[:, 0]),
           np.searchsorted(angle, data[:, 1])] = data[:, 2]
    if not np.allclose(amplitude[[0, -1]], DOMAIN[0], rtol=0, atol=1e-9):
        raise ValueError("Amplitude grid must span the [0.001, 0.06] domain")
    amplitude[0], amplitude[-1] = DOMAIN[0]  # endpoints off by <1e-9 in source
    if angle[0] != 0.0 or angle[-1] >= np.pi:
        raise ValueError("Angle grid must start at 0 and stop below pi")
    # Periodic closure: the CSV has no theta = pi rows because t(A, pi) equals
    # t(A, 0) exactly — append the pi column as a copy of the theta = 0 column.
    angle = np.append(angle, np.pi)
    values = np.concatenate((values, values[:, :1]), axis=1)
    return RectBivariateSpline(amplitude, angle, values, kx=3, ky=3)


def oracle(x):
    """2D -> 1D ground truth: (A_3d, theta_rel_rad) -> transition time / 300.

    Bicubic spline of the t_RetaoMean CSV grid; theta is pi-periodic (queries
    wrap to [0, pi), where the grid is closed with t(A, pi) = t(A, 0)); the
    interpolated time is clipped to [0, 300] s before normalization.
    """
    x = np.asarray(x, dtype=float)
    single = x.shape == (2,)
    x = np.atleast_2d(x)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n, 2) array of (A, theta)")
    tol = 1e-9 * np.maximum(1.0, np.abs(DOMAIN).max(axis=1))
    if np.any(x < DOMAIN[:, 0] - tol) or np.any(x > DOMAIN[:, 1] + tol):
        raise ValueError(f"x outside DOMAIN {DOMAIN.tolist()}")
    amplitude = np.clip(x[:, 0], *DOMAIN[0])
    theta = np.remainder(x[:, 1], np.pi)
    t = _load_spline().ev(amplitude, theta)
    y = np.clip(t, T_MIN, T_MAX) / T_MAX
    return float(y[0]) if single else y


# ----------------------------------------------------------------------------
# Input distribution. Default: INDEPENDENT UNIFORM on DOMAIN. For any other
# law (e.g. Gaussian like the toy), replace these three functions — see
# TODO(5). They must be mutually consistent: from_unit(to_unit(x)) == x.
# ----------------------------------------------------------------------------
def to_unit(x):
    """Physical x -> copula coordinates u in (0,1)^2 (used by nets/LHS/candidates)."""
    x = np.asarray(x, dtype=float)
    return (x - DOMAIN[:, 0]) / (DOMAIN[:, 1] - DOMAIN[:, 0])


def from_unit(u):
    u = np.asarray(u, dtype=float)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("unit coordinates must be open (0,1)")
    return DOMAIN[:, 0] + u * (DOMAIN[:, 1] - DOMAIN[:, 0])


def input_pdf(x):
    """p_X in PHYSICAL coordinates (numerator of the USLW weight)."""
    x = np.asarray(x, dtype=float)
    return np.full(len(x), 1.0 / np.prod(DOMAIN[:, 1] - DOMAIN[:, 0]))


# TODO(5): non-uniform input example (Gaussian N(0, 10 I), as in the toy):
#   STD = np.sqrt(10.)
#   def to_unit(x):  return scipy.special.ndtr(np.asarray(x)/STD)
#   def from_unit(u): return STD*scipy.special.ndtri(u)
#   def input_pdf(x): return np.exp(-np.sum(x**2, axis=1)/20)/(20*np.pi)


def lhs(n=3, seed=101):
    """Reference initial design (run.py re-implements this with --lhs-seed)."""
    rng = np.random.RandomState(seed)
    u = np.column_stack([(rng.permutation(n) + rng.rand(n)) / n for _ in range(2)])
    return from_unit(u)


def prepare(mc=100_000, ref_mc=1_000_000, seed=90210, grid_n=201):
    """Generate data/evaluation.npz and data/reference_law.npz (one-time).

    evaluation.npz  x_mc(mc,2) y_mc(mc) x_grid(grid_n^2,2) y_grid  — fixed,
                    independent test set (never shown to the learner).
    reference_law.npz  p (quantile nodes) / q (quantiles of ref_mc truth
                    samples) / scale — the W1 reference law and z-normalization.
    """
    DATA.mkdir(exist_ok=True)
    # independent MC test set
    x_mc = np.random.RandomState(seed).uniform(DOMAIN[:, 0], DOMAIN[:, 1],
                                               size=(mc, 2))
    y_mc = oracle(x_mc)
    axis = np.linspace(DOMAIN[:, 0], DOMAIN[:, 1], grid_n)
    xx, yy = np.meshgrid(axis[:, 0], axis[:, 1], indexing="ij")
    x_grid = np.column_stack([xx.ravel(), yy.ravel()])
    y_grid = oracle(x_grid)
    assert np.isfinite(y_mc).all() and np.isfinite(y_grid).all(), \
        "oracle returned non-finite values — check TODO(3)"
    np.savez_compressed(DATA / "evaluation.npz", x_mc=x_mc, y_mc=y_mc,
                        x_grid=x_grid, y_grid=y_grid, axis=axis[:, 0])
    # reference law from a large independent sample
    xr = np.random.RandomState(seed + 1).uniform(DOMAIN[:, 0], DOMAIN[:, 1],
                                                 size=(ref_mc, 2))
    values = np.sort(oracle(xr))
    p = np.unique(np.r_[np.linspace(0.0, 1.0, 8193),
                        np.geomspace(1e-7, 0.01, 128),
                        1 - np.geomspace(1e-7, 0.01, 128)])
    q = np.quantile(values, p)
    keep = np.r_[True, np.diff(q) > 0]
    p, q = p[keep], q[keep]
    scale = float(q[-1]) if SCALE_FIXED is None else float(SCALE_FIXED)
    np.savez_compressed(DATA / "reference_law.npz", p=p, q=q, scale=scale)
    print(f"evaluation: mc={mc} grid={grid_n}^2  | y in "
          f"[{y_mc.min():.6g}, {y_mc.max():.6g}]")
    print(f"reference law: {len(p)} knots, scale={scale:.6g}")
    print("Wrote data/evaluation.npz and data/reference_law.npz")


if __name__ == "__main__":
    prepare()
