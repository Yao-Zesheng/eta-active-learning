"""Official toy data and verified query oracle, kept separate from the learner.

The five centers, variances and weights follow arXiv:2510.19161v1 F.1.
Its printed sqrt(variance) normalization differs from the released data.
The standard 2D Gaussian normalization (variance) reproduces toy.pth to float32
precision. The preparation command checks ALL official pairs before any run.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parents[1] / 'eta'
DATA = OFFICIAL / 'data/toy.pth'
STD = np.sqrt(10.)

def oracle(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    for a, v, c in [(1.5,.5,[2,2]), (1.5,.7,[-1,-1]), (1.,.3,[2,-2]),
                    (.5,.9,[0,1]), (1.25,.6,[.5,-.5])]:
        out += a/(2*np.pi*v)*np.exp(-np.sum((x-c)**2, axis=1)/(2*v))
    return out

def to_unit(x):
    return ndtr(np.asarray(x)/STD)

def from_unit(u):
    u = np.asarray(u)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError('Finite Gaussian queries need open-unit coordinates')
    return STD*ndtri(u)

def input_pdf(x):
    return np.exp(-np.sum(np.asarray(x)**2, axis=1)/20)/(20*np.pi)

def lhs(n=3, seed=101):
    rng = np.random.RandomState(seed)
    u = np.column_stack([(rng.permutation(n)+rng.rand(n))/n for _ in range(2)])
    return from_unit(u)

def prepare():
    folder = ROOT/'data'
    folder.mkdir(exist_ok=True)
    official = torch.load(DATA, weights_only=False, map_location='cpu')
    checks = {}
    for xkey, ykey in [('X_all','Y_all'),('in_grid','out_grid'),
                       ('x_train','y_train'),('x_train_e','y_train_e')]:
        y = official[ykey].numpy().ravel().astype(float)
        error = oracle(official[xkey].numpy())-y
        checks[xkey] = dict(n=len(y), max_abs_error=float(abs(error).max()),
                           rmse=float(np.sqrt(np.mean(error**2))))
        assert abs(error).max() < 4e-7, checks[xkey]
    # Only unpaired scalar outputs are passed into the learner's reference law.
    values = np.sort(official['Y_all'].numpy().astype(float))
    p = np.unique(np.r_[np.linspace(0,1,8193), np.geomspace(1e-7,.01,128),
                        1-np.geomspace(1e-7,.01,128)])
    q = np.quantile(values, p)
    keep = np.r_[True, np.diff(q)>0]
    p, q = p[keep], q[keep]
    assert p[0] == 0 and p[-1] == 1
    scale = q[-1]
    np.savez_compressed(folder/'reference_law.npz',p=p,q=q,scale=scale)
    # Independent evaluation: no coordinates or labels returned to fit/select.
    x_mc = np.random.RandomState(90210).normal(0,STD,(100000,2))
    axis = np.linspace(-8,8,201)
    xx, yy = np.meshgrid(axis,axis,indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    np.savez_compressed(folder/'evaluation.npz',x_mc=x_mc,y_mc=oracle(x_mc),
                        x_grid=grid,y_grid=oracle(grid),axis=axis)
    refcdf = np.interp(values,p*0+q,p)  # analytic CDF of the linear quantile law
    n = len(values)
    ks = max(np.max(np.arange(1,n+1)/n-refcdf),
             np.max(refcdf-np.arange(n)/n))
    report = dict(oracle_validation=checks, reference_knots=len(p),
                  reference_vs_official_empirical_cdf_sup=float(ks),scale=float(scale),
                  official_data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  source='https://arxiv.org/html/2510.19161v1#A6.SS1',
                  normalization='weight/(2*pi*variance); validated against toy.pth',
                  input_law='N(0,10 I), full unbounded support; no truncated Gaussian')
    (folder/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__ == '__main__':
    prepare()
