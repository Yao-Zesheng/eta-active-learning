"""One-time extraction of the Hard-PDF run's curves into this folder.

Reads (READ-ONLY) ../TOY_2D_HARDPDF_5X200/results/uslw_seed101 — the
hard-constrained E-NN + USLW run on the identical toy problem (same 3 LHS
initial points seed 101, same member-seed schedule, same candidate stream,
same 100,000-point Gaussian MC evaluation set, verified byte-identical here)
— and saves to results/hard_pdf_curves.npz, with PER-MEMBER statistics
(the hard constraint guarantees each member's marginal, NOT the ensemble
mean, so member-level curves are the correct comparison):

  pdf_e_m1/m2   : each member's log10-PDF error per round (history.json
                  pdf_e_members), same metric as our runs
  ext_mse_m1/m2 : each member's extreme-region MSE per round = mean squared
                  error over the union of the top-0.1% and bottom-0.1%
                  true-y quantile regions of the shared MC set
  (pdf_e_mean / ext_mse_mean kept only for reference, not plotted)

Usage: python -B extract_hard_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HARD = ROOT.parent / "TOY_2D_HARDPDF_5X200" / "results" / "uslw_seed101"
OUT = ROOT / "results" / "hard_pdf_curves.npz"

ev = np.load(ROOT / "data" / "evaluation.npz")
evh = np.load(ROOT.parent / "TOY_2D_HARDPDF_5X200" / "data" / "evaluation.npz")
assert np.array_equal(ev["x_mc"], evh["x_mc"]) and np.array_equal(ev["y_mc"], evh["y_mc"]), \
    "evaluation sets differ; curves would not be comparable"

y_mc = ev["y_mc"]
mask = (y_mc >= np.quantile(y_mc, 0.999)) | (y_mc <= np.quantile(y_mc, 0.001))

rows = json.loads((HARD / "history.json").read_text())
assert len(rows) == 101, "unexpected round count"
pdf_e_members = np.array([r["pdf_e_members"] for r in rows])        # (101, 2)
pdf_e_mean = np.array([r["pdf_e_mean"] for r in rows])

ext_mse = np.zeros((101, 2))
ext_mse_mean = np.zeros(101)
for k in range(101):
    mc = np.load(HARD / f"round_{k:03d}_predictions.npz")["mc"]     # (2, 100000)
    err = mc[:, mask] - y_mc[mask][None, :]
    ext_mse[k] = np.mean(err ** 2, axis=1)
    ext_mse_mean[k] = float(np.mean((mc.mean(0)[mask] - y_mc[mask]) ** 2))

np.savez_compressed(OUT, rounds=np.arange(101),
                    pdf_e_m1=pdf_e_members[:, 0], pdf_e_m2=pdf_e_members[:, 1],
                    pdf_e_mean=pdf_e_mean,
                    ext_mse_m1=ext_mse[:, 0], ext_mse_m2=ext_mse[:, 1],
                    ext_mse_mean=ext_mse_mean)
print(f"saved {OUT}")
for i in (1, 2):
    e, m = pdf_e_members[:, i - 1], ext_mse[:, i - 1]
    print(f"member{i} pdf_e : @0={e[0]:.4f} @26={e[26]:.4f} @50={e[50]:.4f} @100={e[100]:.4f} "
          f"median(last51)={np.median(e[50:]):.4f}")
    print(f"member{i} ext_mse: @0={m[0]:.3e} @26={m[26]:.3e} @50={m[50]:.3e} @100={m[100]:.3e} "
          f"median(last51)={np.median(m[50:]):.3e}")

