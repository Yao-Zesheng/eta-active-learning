# Hard-PDF + USLW with the ordinary baseline's MLP architecture

This is a new, independent run. Existing projects and previous results remain
read-only. The ordinary baseline is reused from
`../TOY_2D_ETA_LAMBDA0_USLW/results/uslw_seed101`.

## Run and reproduce

In the existing `pytorch` conda environment:

```powershell
python -B run.py --threads 8
# Only if continuing an interrupted run:
python -B run.py --threads 8 --resume
python -B audit_run.py
python -B render.py
```

Read `report.html` after rendering. The output includes two individual
Hard-PDF member error curves, the ordinary ensemble-mean curve, and four-column
maps at rounds 1–10, 20–90 every ten rounds, plus round 100. Each map displays
truth, Hard-PDF NN1, Hard-PDF NN2, and the ordinary ensemble mean. All maps share
one color scale, without output clipping. Curve values are also exported to CSV.

## What is matched

Both networks have topology `[2,200,200,200,200,200,1]`: five ReLU hidden layers,
all linear-layer biases included, 161601 trainable parameters per member.
The ordinary trainer's `make_mlp` and `seeded_member_init` are copied verbatim
in `mlp_repro.py`. Initial MLP parameters match exactly for a given seed.
Both MLPs use float32. The Hard-PDF probability transform casts vertex scores
to float64 and calculates the analytic triangle CDF and reference quantile in
float64; this does not approximate or soften the hard distribution constraint.

Both active learning runs start from three identical LHS points with seed 101,
use two members, restart with seeds `2*k+1,2*k+2`, and use the same 1000-candidate
Gaussian stream and USLW rule. Every selected point is newly queried from the
verified official toy oracle. The new Hard-PDF path is recomputed from scratch,
not retrained on the old selected points. Round 100 fits all 103 observations.

## What remains different

The Hard-PDF predictor uses a fixed 32-by-32 probability-space triangular mesh
and the transform `Q_ref(F_score(score(x)))`. This layer has no trainable
parameters. Its deployed score is an affine mesh interpolant, not the raw MLP
between vertices. Every valid member has the numerical reference marginal.
The ensemble mean used inside USLW is not itself guaranteed that marginal.

Hard-PDF retains its previous fitting procedure: Adam up to 1800 steps at
0.002, then LBFGS up to 450 iterations, with gradient clipping and best valid
training-MSE weights. It can stop at normalized RMSE `1e-4`. Constant-triangle
or globally degenerate trials are rejected, not smoothed into valid states.
The ordinary baseline uses Adam 1000 steps at 0.001 and final-step weights.
The network configurations therefore match, while optimization budgets and
the deployed function classes still differ.

## Validation and provenance

`preflight/validation.json` records matched parameters, all 202 admissible
initializations, and an independent uniform-rank check. Preflight timing
experiments compare CPU/GPU and precision choices; they are not AL results.
The full run uses CPU with eight threads. `source_sha256.json` records the
original source files; `audit_run.py` checks they remain unchanged, all 100
candidate argmaxes and observation labels, identical initial design, final
checkpoint reproduction and analytic-versus-compiled prediction consistency.

Evaluation uses the same independent 100000 Gaussian inputs and 201-by-201
plotting grid as before. The corrected fixed-bandwidth PDF metric is unchanged.
`render.py` verifies each member's PDF errors and RMSE from saved predictions
before producing the report. Per-round next states are committed before the
atomic history replacement, so resuming does not silently omit a queried label.
