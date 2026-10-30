# Official toy: lambda=0 E-NN + USLW

This directory is an independent experiment. All original projects are read-only.

## Reproduce

Use the `pytorch` conda environment (NumPy, SciPy, PyTorch, Matplotlib):

```powershell
python -B run.py
# Resume an interrupted run, or verify a finished run without retraining:
python -B run.py --resume
python -B compare.py
```

The completed comparison is `comparison/report.html`. PNG figures, the full
learning-curve CSV, summary statistics and audit results are next to it.
Per-round checkpoints, predictions and acquisition scores are under
`results/uslw_seed101`.

## Provenance and choices

- `mlp_repro.py` is a byte-identical copy of `../POIS_2D/mlp_repro.py`, the
  trainer used by `run_pois_lamsweep.py`. Its `lambd=0` branch is the existing
  pure-MSE baseline. `legacy_run_source.py` records the original runner and is
  not executed.
- The official toy replaces the Poiseuille system: independent Gaussian
  inputs with covariance `10 I` and a five-Gaussian deterministic oracle.
  `problem.py` and `acquisition.py` are unchanged copies of the previous toy
  experiment. See the previous preparation report for verification against all
  one million official data pairs.
- Two networks, five hidden layers of 200 ReLU units, full-batch Adam,
  learning rate 0.001, 1000 steps, final weights. No W1, auxiliary training
  labels, distribution pretraining, output clipping or early stopping.
- Input features are `2 Phi(x/sqrt(10))-1`. A fixed scalar reference maximum
  rescales the target; no reference PDF or quantile constraints enter training.
- Three LHS points, seed 101, exactly matching the hard-model experiment.
  Cold restarts use member seeds `2*k+1, 2*k+2` at acquired-label count `k`.
- At each round, 1000 candidates are drawn with `RandomState(n_observations)`
  in probability coordinates and transformed to the full Gaussian support.
  The copied selection routine maximizes `variance*p_X/KDE(mean)` over these
  candidates. This is a candidate argmax, not certified continuous optimization.
- Round 0 fits three observations. Rounds 1 through 100 fit 4 through 103.
  Round 100 performs the final refit without querying another point.
- Every round writes its model, predictions and next observation state before
  atomically committing history. Resume uses the state of the last committed
  round. Files in a deliberately separate `--name` can hold another run.

## Evaluation

Both trajectories use the same independent 100000 Gaussian test inputs and
201-by-201 plotting grid. The grid is only a visualization window, not a
truncation of the input distribution. The reference upper 1% and upper 0.1%
thresholds classify truth values, not the model's own rankings.

PDF error is the normalized integral of the absolute log10-density difference
on `[0, reference_max]`, floor `1e-16`. Both methods use the same fixed bandwidth
from the true test outputs, linear mass deposition and a 10-sigma Gaussian
kernel. The ordinary model's negative and out-of-range predictions are included
in the density estimate and its normalization. The deposition grid is extended
as needed, while the integration grid is identical. This is not numerically
interchangeable with Poiseuille metrics using different domains or bandwidths.

`compare.py` recomputes every PDF error and mapping RMSE, checks all 200 query
argmaxes and oracle labels, checks identical initial designs and candidate
streams, reproduces the ordinary final checkpoint, and validates the final
PDF evaluation against direct Gaussian summation.

## Limits of inference

The hard model has two hidden layers of width 48 with Tanh, a piecewise-linear
probability-conserving projection, float64, and a different Adam/LBFGS training
budget. The ordinary pipeline uses five layers of width 200 and float32.
This requested comparison therefore does not isolate the causal effect of the
PDF constraint. One initial design and member-seed schedule are not a statistical
benchmark. Inspect the full raw learning curves and final-21-round variation,
not just a favorable checkpoint.

Each hard member obeys the numerical reference marginal distribution; its
ensemble mean need not. A correct marginal PDF does not identify the spatial
input-output mapping.
