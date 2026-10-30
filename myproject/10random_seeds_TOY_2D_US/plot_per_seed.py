"""Per-seed comparison figures for the 10-seed TOY_2D US-acquisition experiment.

For each seed S in 101..110 (all data from summary_seeds.npz, computed on the
HPC from the 60 finished US-acquisition runs):

  figures/per_seed/s{S}_pdf_e.png   7 curves: 5 lambdas (solid) + hard-PDF
                                     member1/member2 (purple dashed)
  figures/per_seed/s{S}_ext_mse.png same 7 curves, extreme-region MSE

Usage: python -B plot_per_seed.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures" / "per_seed"
FIG.mkdir(parents=True, exist_ok=True)

LAMS = ["0", "1e-9", "1e-6", "0.001", "1"]
LAM_LABELS = ["λ=0（纯 MSE）", "λ=1e-9", "λ=1e-6", "λ=1e-3", "λ=1（η 全强度）"]
COLORS = ["#00441b", "#29a35a", "#b8d950", "#f4a23a", "#c4261e"]
HARD_COLORS = ["#6a3d9a", "#b08fd6"]
ROUNDS = np.arange(101)

summary = np.load(ROOT / "summary_seeds.npz")


def curves_for(S: int):
    out = {lam: (summary[f"s{S}_eta_{lam}_pdf_e"], summary[f"s{S}_eta_{lam}_ext_mse"])
           for lam in LAMS}
    hard = {k: summary[f"s{S}_hard_{k}"] for k in
            ("pdf_e_m1", "pdf_e_m2", "ext_mse_m1", "ext_mse_m2")}
    return out, hard


def draw(S, which, curves, hard):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    key = which
    series = []
    for lam, color, label in zip(LAMS, COLORS, LAM_LABELS):
        y = curves[lam][0] if which == "pdf_e" else curves[lam][1]
        ax.plot(ROUNDS, y, color=color, lw=1.9, label=label)
        series.append(y)
    for m, color in ((1, HARD_COLORS[0]), (2, HARD_COLORS[1])):
        y = hard[f"{key}_m{m}"]
        ax.plot(ROUNDS, y, color=color, lw=2.0, ls="--",
                label=f"PDF 硬约束 E-NN+US 成员{m}")
        series.append(y)
    ax.set_yscale("log")
    lo = min(s.min() for s in series)
    hi = max(s.max() for s in series)
    ax.set_ylim(lo * 0.5, hi * 3.0)          # explicit limits (log-axis rule)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlabel("迭代轮次（round 0 = 3 个 LHS 初始点；round 100 = 最终重拟合）")
    if which == "pdf_e":
        ax.set_ylabel(r"$e_{\log_{10}p}$（越小越好，对数轴）")
        ax.set_title(f"US 采集｜种子组 {S}（LHS 种子 {S}，成员种子基 {max(1,(S-101)*1000+1)}）："
                     "PDF 误差随迭代变化", fontsize=12)
    else:
        ax.set_ylabel("极端区域预测 MSE（真值两端 0.1%，对数轴）")
        ax.set_title(f"US 采集｜种子组 {S}：极端区域预测 MSE 随迭代变化", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"s{S}_{which}.png", dpi=130)
    plt.close(fig)


for S in range(101, 111):
    curves, hard = curves_for(S)
    draw(S, "pdf_e", curves, hard)
    draw(S, "ext_mse", curves, hard)
print(f"20 figures in {FIG}")
