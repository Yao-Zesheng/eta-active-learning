"""Seed-averaged summary: two panels (PDF error / extreme-region MSE).

Each of the 7 methods (5 lambdas + hard-PDF member1/member2) is shown as a
solid line = mean over the 10 seed groups, with a same-color shaded band =
min..max over the 10 seed groups.

Data sources identical to plot_per_seed.py (s101 local originals,
s102-110 from summary_seeds.npz).

Usage: python -B plot_seed_summary.py   -> figures/seed_summary_mean_range.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
S101 = ROOT.parent / "TOY_2D_ETA_LAMBDA0_USLW" / "results"
OUT = ROOT / "figures" / "seed_summary_mean_range_en.png"

LAMS = ["0", "1e-9", "1e-6", "0.001", "1"]
LAM_LABELS = ["λ=0 (pure MSE)", "λ=1e-9", "λ=1e-6", "λ=1e-3", "λ=1 (full-strength η)"]
COLORS = ["#00441b", "#29a35a", "#b8d950", "#f4a23a", "#c4261e"]
HARD_COLORS = ["#6a3d9a", "#b08fd6"]
ROUNDS = np.arange(101)
SEEDS = list(range(101, 111))

summary = np.load(ROOT / "summary_seeds.npz")
ext101 = np.load(S101 / "extreme_region_mse.npz")
hard101 = np.load(S101 / "hard_pdf_curves.npz")
S101_LAM = {"0": "lam0", "1e-9": "lam1e-09", "1e-6": "lam1e-06",
            "0.001": "lam0.001", "1": "lam1"}
S101_NAMES = {"0": "uslw_seed101", "1e-9": "uslw_lam1e-9_seed101",
              "1e-6": "uslw_lam1e-6_seed101", "0.001": "uslw_lam0.001_seed101",
              "1": "uslw_lam1_seed101"}


def curves_for(S: int):
    out = {}
    if S == 101:
        for lam in LAMS:
            rows = json.loads((S101 / S101_NAMES[lam] / "history.json").read_text())
            out[lam] = (np.array([r["pdf_e_mean"] for r in rows]), ext101[S101_LAM[lam]])
        hard = {k: hard101[k] for k in ("pdf_e_m1", "pdf_e_m2", "ext_mse_m1", "ext_mse_m2")}
    else:
        for lam in LAMS:
            out[lam] = (summary[f"s{S}_eta_{lam}_pdf_e"], summary[f"s{S}_eta_{lam}_ext_mse"])
        hard = {k: summary[f"s{S}_hard_{k}"] for k in
                ("pdf_e_m1", "pdf_e_m2", "ext_mse_m1", "ext_mse_m2")}
    return out, hard


# method -> (10,101) arrays for both metrics
pdf = {}
ext = {}
for lam in LAMS:
    pdf[lam] = np.array([curves_for(S)[0][lam][0] for S in SEEDS])
    ext[lam] = np.array([curves_for(S)[0][lam][1] for S in SEEDS])
for m, key in ((1, "pdf_e_m1"), (2, "pdf_e_m2")):
    pdf[f"hard{m}"] = np.array([curves_for(S)[1][key] for S in SEEDS])
    ext[f"hard{m}"] = np.array([curves_for(S)[1]["ext_mse_m%d" % m] for S in SEEDS])

print("=== 10-seed statistics (final value, round 100) ===")
print(f"{'method':>14} {'mean':>10} {'min':>10} {'max':>10} {'mean of per-seed medians':>16}")
for name, d in [("λ=0", pdf["0"]), ("λ=1e-9", pdf["1e-9"]), ("λ=1e-6", pdf["1e-6"]),
                ("λ=1e-3", pdf["0.001"]), ("λ=1", pdf["1"]),
                ("hard-m1", pdf["hard1"]), ("hard-m2", pdf["hard2"])]:
    med = np.median(d[:, 50:], axis=1).mean()
    print(f"{name:>14} {d[:,100].mean():>10.4f} {d[:,100].min():>10.4f} "
          f"{d[:,100].max():>10.4f} {med:>16.4f}")
print("(extreme-region MSE table omitted; see figure)")

fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.6))

panels = [
    (axes[0], pdf, r"$e_{\log_{10}p}$ (PDF error, lower is better)", "PDF error"),
    (axes[1], ext, "Extreme-region MSE (top/bottom 0.1% of truth)", "Extreme-region MSE"),
]
series_all = {id(ax): [] for ax, _, _, _ in panels}
for ax, data, ylabel, title in panels:
    for lam, color, label in zip(LAMS, COLORS, LAM_LABELS):
        d = data[lam]
        ax.plot(ROUNDS, d.mean(0), color=color, lw=2.0, label=label)
        ax.fill_between(ROUNDS, d.min(0), d.max(0), color=color, alpha=0.18, lw=0)
        series_all[id(ax)].append(d)
    for m, color in ((1, HARD_COLORS[0]), (2, HARD_COLORS[1])):
        d = data[f"hard{m}"]
        ax.plot(ROUNDS, d.mean(0), color=color, lw=2.0, ls="--",
                label=f"Hard-PDF E-NN+USLW member {m}")
        ax.fill_between(ROUNDS, d.min(0), d.max(0), color=color, alpha=0.18, lw=0)
        series_all[id(ax)].append(d)
    ax.set_yscale("log")
    lo = min(d.min() for d in series_all[id(ax)])
    hi = max(d.max() for d in series_all[id(ax)])
    ax.set_ylim(lo * 0.5, hi * 3.0)          # explicit limits (log-axis rule)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("Active-learning round (round 0 = 3 LHS points; round 100 = final refit)")
    ax.set_ylabel(ylabel + ", log scale")
    ax.set_title(f"{title}: 10-seed mean (solid) and min–max band (shaded)", fontsize=12)

axes[0].legend(fontsize=9.5, loc="upper right")
fig.suptitle("TOY_2D ten-seed experiment: 7 methods (5 lambdas + Hard-PDF members 1/2), "
             "seed groups 101–110", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=140)
plt.close(fig)
print(f"saved {OUT}")
