"""Seed-averaged summary for the US-acquisition experiment, CN + EN versions.

Two panels (PDF error / extreme-region MSE); 7 methods; solid = 10-seed mean,
shaded = min..max band. All curves from summary_seeds.npz (HPC-computed).

Usage: python -B plot_seed_summary.py
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
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LAMS = ["0", "1e-9", "1e-6", "0.001", "1"]
ROUNDS = np.arange(101)
SEEDS = list(range(101, 111))
summary = np.load(ROOT / "summary_seeds.npz")

pdf = {lam: np.array([summary[f"s{S}_eta_{lam}_pdf_e"] for S in SEEDS]) for lam in LAMS}
ext = {lam: np.array([summary[f"s{S}_eta_{lam}_ext_mse"] for S in SEEDS]) for lam in LAMS}
for m in (1, 2):
    pdf[f"hard{m}"] = np.array([summary[f"s{S}_hard_pdf_e_m{m}"] for S in SEEDS])
    ext[f"hard{m}"] = np.array([summary[f"s{S}_hard_ext_mse_m{m}"] for S in SEEDS])

print("=== 10 种子统计（US 采集；PDF 误差 @ round 100）===")
print(f"{'方法':>10} {'均值':>10} {'下限':>10} {'上限':>10} {'逐种子中位的均值':>12}")
for name, d in [("λ=0", pdf["0"]), ("λ=1e-9", pdf["1e-9"]), ("λ=1e-6", pdf["1e-6"]),
                ("λ=1e-3", pdf["0.001"]), ("λ=1", pdf["1"]),
                ("hard-m1", pdf["hard1"]), ("hard-m2", pdf["hard2"])]:
    print(f"{name:>10} {d[:,100].mean():>10.4f} {d[:,100].min():>10.4f} "
          f"{d[:,100].max():>10.4f} {np.median(d[:,50:],axis=1).mean():>12.4f}")

CN = dict(lams=["λ=0（纯 MSE）", "λ=1e-9", "λ=1e-6", "λ=1e-3", "λ=1（η 全强度）"],
          hard="PDF 硬约束 E-NN+US 成员{m}",
          xlabel="迭代轮次（round 0 = 3 个 LHS 初始点；round 100 = 最终重拟合）",
          pdf_ylab=r"$e_{\log_{10}p}$（PDF 误差，越小越好）",
          ext_ylab="极端区域预测 MSE（真值两端 0.1%）",
          pdf_tit="PDF 误差：10 种子平均（实线）与上下限（阴影）",
          ext_tit="极端区域 MSE：10 种子平均（实线）与上下限（阴影）",
          suptit="TOY_2D 十种子实验（US 采集）：7 种方法，种子组 101–110",
          suffix="logscale")
EN = dict(lams=["λ=0 (pure MSE)", "λ=1e-9", "λ=1e-6", "λ=1e-3", "λ=1 (full-strength η)"],
          hard="Hard-PDF E-NN+US member {m}",
          xlabel="Active-learning round (round 0 = 3 LHS points; round 100 = final refit)",
          pdf_ylab=r"$e_{\log_{10}p}$ (PDF error, lower is better)",
          ext_ylab="Extreme-region MSE (top/bottom 0.1% of truth)",
          pdf_tit="PDF error: 10-seed mean (solid) and min\u2013max band (shaded)",
          ext_tit="Extreme-region MSE: 10-seed mean (solid) and min\u2013max band (shaded)",
          suptit="TOY_2D ten-seed experiment (US acquisition): 7 methods, seed groups 101\u2013110",
          suffix="logscale_en")
COLORS = ["#00441b", "#29a35a", "#b8d950", "#f4a23a", "#c4261e"]
HARD_COLORS = ["#6a3d9a", "#b08fd6"]

for L, fname in ((CN, "seed_summary_mean_range.png"), (EN, "seed_summary_mean_range_en.png")):
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.6))
    for ax, data, ylab, tit in [(axes[0], pdf, L["pdf_ylab"], L["pdf_tit"]),
                                (axes[1], ext, L["ext_ylab"], L["ext_tit"])]:
        series = []
        for lam, color, label in zip(LAMS, COLORS, L["lams"]):
            d = data[lam]
            ax.plot(ROUNDS, d.mean(0), color=color, lw=2.0, label=label)
            ax.fill_between(ROUNDS, d.min(0), d.max(0), color=color, alpha=0.18, lw=0)
            series.append(d)
        for m, color in ((1, HARD_COLORS[0]), (2, HARD_COLORS[1])):
            d = data[f"hard{m}"]
            ax.plot(ROUNDS, d.mean(0), color=color, lw=2.0, ls="--",
                    label=L["hard"].format(m=m))
            ax.fill_between(ROUNDS, d.min(0), d.max(0), color=color, alpha=0.18, lw=0)
            series.append(d)
        ax.set_yscale("log")
        lo = min(d.min() for d in series)
        hi = max(d.max() for d in series)
        ax.set_ylim(lo * 0.5, hi * 3.0)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(L["xlabel"])
        ax.set_ylabel(ylab + ("，对数轴" if L is CN else ", log scale"))
        ax.set_title(tit, fontsize=12)
    axes[0].legend(fontsize=9.5, loc="upper right")
    fig.suptitle(L["suptit"], fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)
    print(f"saved {FIG / fname}")
