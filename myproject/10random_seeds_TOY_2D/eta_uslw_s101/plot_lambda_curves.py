"""Five e(iter) curves for the toy case: lambda in {0, 1e-9, 1e-6, 1e-3, 1}.

lambda=0 is the completed official run (results/uslw_seed101); the other four
are results/uslw_lam{g}_seed101 produced by run.py --lambd ... --device cuda.

Usage: python -B plot_lambda_curves.py
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
OUT = ROOT / "results" / "lam_sweep_curves_log.png"

CASES = [
    (0.0,   "uslw_seed101",           "λ=0（纯 MSE，官方基线）"),
    (1e-9,  "uslw_lam1e-9_seed101",   "λ=1e-9"),
    (1e-6,  "uslw_lam1e-6_seed101",   "λ=1e-6"),
    (1e-3,  "uslw_lam0.001_seed101",  "λ=1e-3"),
    (1.0,   "uslw_lam1_seed101",      "λ=1（η 全强度）"),
]
COLORS = ["#00441b", "#29a35a", "#b8d950", "#f4a23a", "#c4261e"]

ref = np.load(ROOT / "data" / "reference_law.npz")
thr99 = float(np.interp(0.99, ref["p"], ref["q"]))

# Hard-PDF constrained E-NN+USLW (TOY_2D_HARDPDF_5X200): same problem, same
# initial 3 LHS points (verified identical). The hard constraint guarantees
# EACH MEMBER's marginal — not the ensemble mean — so member-level curves
# (extracted to the local npz) are the correct comparison.
HARD_NPZ = ROOT / "results" / "hard_pdf_curves.npz"
hard = np.load(HARD_NPZ) if HARD_NPZ.exists() else None
HARD_STYLE = [("pdf_e_m1", "#6a3d9a", "PDF 硬约束 E-NN+USLW 成员1"),
              ("pdf_e_m2", "#b08fd6", "PDF 硬约束 E-NN+USLW 成员2")]

hist = {}
for lam, name, label in CASES:
    rows = json.loads((ROOT / "results" / name / "history.json").read_text())
    hist[lam] = rows

def e_series(rows):
    return np.array([r["pdf_e_mean"] for r in rows])

print("=== 汇总（round 0..100；query 共 100 个）===")
print(f"{'λ':>7} {'e@0':>7} {'e@10':>7} {'e@26':>7} {'e@50':>7} {'e@100':>7} "
      f"{'后51轮中位':>9} {'min e':>7} {'p99采样占比':>9} {'top1@100':>9} {'峰距@100':>8} {'负输出比@100':>10}")
stats = {}
for lam, name, label in CASES:
    rows = hist[lam]
    e = e_series(rows)
    obs = np.array([r["observed_y"] for r in rows[:100]])   # rounds 0..99 query
    hot = float(np.mean(obs >= thr99))
    last = rows[-1]
    stats[lam] = dict(e=e, hot=hot, top1=last["top1_rmse"],
                      peak=last["peak_distance_mean"], neg=last["negative_fraction"])
    print(f"{lam:>7g} {e[0]:>7.4f} {e[10]:>7.4f} {e[26]:>7.4f} {e[50]:>7.4f} {e[100]:>7.4f} "
          f"{np.median(e[50:]):>9.4f} {e.min():>7.4f} {hot:>9.0%} "
          f"{last['top1_rmse']:>9.4f} {last['peak_distance_mean']:>8.3f} "
          f"{last['negative_fraction']:>10.2%}")
if hard is not None:
    for key, tag in (("pdf_e_m1", "hard-m1"), ("pdf_e_m2", "hard-m2")):
        e = hard[key]
        print(f"{tag:>7} {e[0]:>7.4f} {e[10]:>7.4f} {e[26]:>7.4f} {e[50]:>7.4f} {e[100]:>7.4f} "
              f"{np.median(e[50:]):>9.4f} {e.min():>7.4f} {'—':>9} {'—':>9} {'—':>8} {'—':>10}")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), gridspec_kw={"width_ratios": [2.1, 1]})

ax = axes[0]
for (lam, name, label), color in zip(CASES, COLORS):
    e = e_series(hist[lam])
    ax.plot(np.arange(len(e)), e, color=color, lw=1.9, label=label)
if hard is not None:
    for key, color, label in HARD_STYLE:
        ax.plot(hard["rounds"], hard[key], color=color, lw=2.0, ls="--", label=label)
ax.set_xlabel("迭代轮次（round 0 = 3 个 LHS 初始点；round 100 = 最终重拟合）")
ax.set_ylabel(r"$e_{\log_{10}p}$（越小越好，对数轴）")
ax.set_yscale("log")
lo = min(e_series(hist[l]).min() for l, _, _ in CASES)
hi = max(e_series(hist[l]).max() for l, _, _ in CASES)
if hard is not None:
    for key, _, _ in HARD_STYLE:
        lo, hi = min(lo, hard[key].min()), max(hi, hard[key].max())
ax.set_ylim(lo * 0.5, hi * 2.0)          # explicit limits: no auto junk on log axes
ax.set_xlim(0, 100)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10.5, loc="upper right")
ax.set_title("玩具案例 λ 扫描：PDF 误差 e 随迭代的变化（2×MLP[5×200] + USLW + 逐轮换种子，仅 λ 不同）",
             fontsize=12.5)

ax2 = axes[1]
labels = [c[2].split("（")[0] for c in CASES]
colors2 = list(COLORS)
e100 = [e_series(hist[l])[-1] for l, _, _ in CASES]
med = [float(np.median(e_series(hist[l])[50:])) for l, _, _ in CASES]
if hard is not None:
    for key, color, _ in HARD_STYLE:
        labels.append(key[-2:].replace("m", "硬m"))
        colors2.append(color)
        e100.append(float(hard[key][-1]))
        med.append(float(np.median(hard[key][50:])))
xx = np.arange(len(labels))
b1 = ax2.bar(xx - 0.21, e100, width=0.42, color=colors2, alpha=0.9)
b2 = ax2.bar(xx + 0.21, med, width=0.42, color=colors2, alpha=0.45, hatch="//")
for bars, vals in ((b1, e100), (b2, med)):
    for r, v in zip(bars, vals):
        ax2.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:.3g}",
                 ha="center", fontsize=8.5)
ax2.set_yscale("log")
ax2.set_ylim(lo * 0.5, hi * 2.0)
ax2.set_ylabel("e（对数轴）", fontsize=10.5)
ax2.set_xticks(xx)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(alpha=0.3, axis="y", which="both")
ax2.legend([b1[0], b2[0]], ["e @ round 100", "后 51 轮中位数"], fontsize=9.5)
ax2.set_title("终值对比（深=终值，浅=后51轮中位）", fontsize=11.5)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
plt.close(fig)
print(f"\nsaved {OUT}")
