"""Legacy USLW KDE and candidate argmax, MMT conventions (dnosearch nls_MLP).

Candidates are UNIFORM in the design box [-6,6]^2 (linear from_unit), members
predict the RAW physical candidates (the nets consume raw x), and the USLW
output-density KDE is weighted by p_X at the candidates — exactly the original
dnosearch formula (gaussian_kde(mean, weights=px)); an unweighted KDE would
estimate the box-uniform output density, the wrong target. The p_X numerator
is in PHYSICAL x coordinates. No candidate truth is computed until after
argmax.
"""
import numpy as np
from scipy.stats import gaussian_kde
from problem import from_unit, to_unit, input_pdf, candidate_pdf

FLOOR = 1e-16

def kde(values, grid, weights=None):
    values = np.asarray(values,dtype=float).ravel()
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('KDE requires at least two finite values')
    if np.ptp(values) <= np.finfo(float).eps*max(1.,np.abs(values).max()):
        h = 1e-6*max(1.,abs(values[0]))
        result = np.exp(-.5*((np.asarray(grid)-values[0])/h)**2)/(np.sqrt(2*np.pi)*h)
    elif weights is None:
        result = gaussian_kde(values)(grid)
    else:
        w = np.asarray(weights,dtype=float).ravel()
        result = gaussian_kde(values,weights=w/w.sum())(grid)
    return np.maximum(result,FLOOR)

def select_query(x, compiled, scale, candidate_count=1000, acq='uslw'):
    candidates = from_unit(np.random.RandomState(len(x)).rand(candidate_count,2))
    u = to_unit(candidates)
    distance = np.linalg.norm(u[:,None]-to_unit(x)[None,:],axis=2)
    good = distance.min(axis=1)>1e-8
    candidates, u = candidates[good],u[good]
    members = np.array([m.predict(candidates)*scale for m in compiled])
    mu, variance = members.mean(0),members.var(0)
    padding = 1e-6*np.ptp(mu)+1e-12
    grid = np.linspace(mu.min()-padding,mu.max()+padding,10000)
    # importance weights input_pdf/proposal: uniform-box proposal -> p_X*area
    # (dnosearch weighted KDE); PIT-copula proposal -> weights == 1 (plain
    # KDE). One formula covers every convention; problem.py owns both densities.
    density = kde(mu,grid,weights=input_pdf(candidates)/candidate_pdf(candidates))
    if acq == 'us':
        scores = variance
    else:
        scores = variance*input_pdf(candidates)/np.maximum(np.interp(mu,grid,density),FLOOR)
    index = int(np.argmax(scores))
    return candidates[index:index+1], dict(candidates=candidates, members=members,
        mean=mu, variance=variance, scores=scores, selected_index=index)
