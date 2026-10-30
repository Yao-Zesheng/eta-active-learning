"""Extreme-region prediction MSE vs round, per lambda.

Metric (user-specified): on the fixed 100,000 Gaussian MC set,
  y_high = 99.9% quantile of the true y  -> {x_high} = points with y >= y_high
  y_low  =  0.1% quantile of the true y  -> {x_low}  = points with y <= y_low
  extreme-region MSE = mean over the UNION of both regions of
                       (ensemble-mean prediction - true y)^2
Everything is recomputed from the saved per-round predictions
(round_XXX_predictions.npz holds both members' predictions on the same MC
set), so no model retraining is needed.

Usage: python -B plot_extreme_mse.py
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
OUT = ROOT / "results" / "extreme_region_mse.png"

ev = np.load(ROOT / "data" / "evaluation.npz")
y_mc = ev["y_mc"]
y_high = float(np.quantile(y_mc, 0.999))
y_low = float(np.quantile(y_mc, 0.001))
mask_hi = y_mc >= y_high
mask_lo = y_mc <= y_low
mask = mask_hi | mask_lo
print(f"y_high(99.9% 分位)={y_high:.6g}，上方 {mask_hi.sum()} 点；"
      f"y_low(0.1% 分位)={y_low:.6g}，下方 {mask_lo.sum()} 点；合计 {mask.sum()} 点")

CASES = [
    (0.0,   "uslw_seed101",           "λ=0（纯 MSE，官方基线）"),
    (1e-9,  "uslw_lam1e-9_seed101",   "λ=1e-9"),
    (1e-6,  "uslw_lam1e-6_seed101",   "λ=1e-6"),
    (1e-3,  "uslw_lam0.001_seed101",  "λ=1e-3"),
    (1.0,   "uslw_lam1_seed101",      "λ=1（η 全强度）"),
]
COLORS = ["#00441b", "#29a35a", "#b8d950", "#f4a23a", "#c4261e"]
ROUNDS = list(range(101))

# Hard-PDF constrained E-NN+USLW (TOY_2D_HARDPDF_5X200), same problem and
# initial design. The hard constraint holds for EACH MEMBER (not the
# ensemble mean), so per-member curves are plotted; extracted to the local
# npz by extract_hard_curves.py.
HARD_NPZ = ROOT / "results" / "hard_pdf_curves.npz"
hard = np.load(HARD_NPZ) if HARD_NPZ.exists() else None
HARD_STYLE = [("ext_mse_m1", "#6a3d9a", "PDF 硬约束 E-NN+USLW 成员1"),
              ("ext_mse_m2", "#b08fd6", "PDF 硬约束 E-NN+USLW 成员2")]

series = {}
for lam, name, label in CASES:
    vals = []
    for k in ROUNDS:
        pred = np.load(ROOT / "results" / name / f"round_{k:03d}_predictions.npz")["mc"].mean(0)
        err = pred[mask] - y_mc[mask]
        vals.append(float(np.mean(err ** 2)))
    series[lam] = np.array(vals)

print("\n=== 极端区域预测 MSE（两成员集成均值；真值两端 0.1% 合并区域）===")
print(f"{'λ':>7} {'@0':>10} {'@10':>10} {'@26':>10} {'@50':>10} {'@100':>10} "
      f"{'后51轮中位':>11} {'min':>10}")
for lam, name, label in CASES:
    v = series[lam]
    print(f"{lam:>7g} {v[0]:>10.3e} {v[10]:>10.3e} {v[26]:>10.3e} {v[50]:>10.3e} "
          f"{v[100]:>10.3e} {np.median(v[50:]):>11.3e} {v.min():>10.3e}")
if hard is not None:
    for key, tag in (("ext_mse_m1", "hard-m1"), ("ext_mse_m2", "hard-m2")):
        v = hard[key]
        print(f"{tag:>7} {v[0]:>10.3e} {v[10]:>10.3e} {v[26]:>10.3e} {v[50]:>10.3e} "
              f"{v[100]:>10.3e} {np.median(v[50:]):>11.3e} {v.min():>10.3e}")
        series["hard-" + key[-2:]] = v

fig, ax = plt.subplots(figsize=(11.5, 6.2))
for (lam, name, label), color in zip(CASES, COLORS):
    ax.plot(ROUNDS, series[lam], color=color, lw=1.9, label=label)
if hard is not None:
    for key, color, label in HARD_STYLE:
        ax.plot(hard["rounds"], hard[key], color=color, lw=2.0, ls="--", label=label)
ax.set_xlabel("迭代轮次（round 0 = 3 个 LHS 初始点；round 100 = 最终重拟合）")
ax.set_ylabel("极端区域预测 MSE（真值两端 0.1%，对数轴）")
ax.set_yscale("log")
lo = min(v.min() for v in series.values())
hi = max(v.max() for v in series.values())
ax.set_ylim(lo * 0.5, hi * 3.0)          # explicit limits: no auto junk on log axes
ax.set_xlim(0, 100)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10.5, loc="upper right")
ax.set_title(f"极端区域预测 MSE 随迭代变化（阈值 y_high={y_high:.4g} / y_low={y_low:.3g}；"
             f"共 {int(mask.sum())} 个 MC 点）", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
plt.close(fig)
print(f"\nsaved {OUT}")

# persist the curves for the report
np.savez_compressed(ROOT / "results" / "extreme_region_mse.npz",
                    rounds=np.array(ROUNDS),
                    **{f"lam{lam:g}": series[lam] for lam, _, _ in CASES},
                    y_high=y_high, y_low=y_low,
                    n_high=int(mask_hi.sum()), n_low=int(mask_lo.sum()))
