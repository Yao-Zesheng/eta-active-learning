"""Numerical integration of each actual compiled member's continuous output law.
The reference PDF is not substituted for the model. Output thresholds are
inverted through the compiled transform, then integrated over mesh triangles.
"""
import numpy as np
from scipy.stats import gaussian_kde
from model import triangle_cdf_numpy


def continuous_member_pdfs(compiled, scale, evaluation, grid, truth_pdf,
                           grid_predictions, bins=8192):
    curves, errors, metadata = [], [], []
    weights = evaluation['w']
    step = grid[1]-grid[0]
    for member, sampled_prediction in zip(compiled, grid_predictions):
        kde = gaussian_kde(sampled_prediction, weights=weights/weights.sum())
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))
        edges = np.linspace(member.q[0]*scale, member.q[-1]*scale, bins+1)
        lower = np.full(len(edges), member.knots[0])
        upper = np.full(len(edges), member.knots[-1])
        for _ in range(60):
            middle = (lower+upper)/2
            output = np.interp(member.cdf(middle), member.p, member.q)*scale
            go_right = output <= edges
            lower = np.where(go_right, middle, lower)
            upper = np.where(go_right, upper, middle)
        threshold = (lower+upper)/2
        threshold[0], threshold[-1] = member.knots[0], member.knots[-1]
        abc = member.scores[member.tri]
        cdf = triangle_cdf_numpy(threshold, abc, member.areas)
        survival = triangle_cdf_numpy(-threshold, -abc, member.areas)
        mass = np.where(cdf[:-1] < .5, np.diff(cdf), -np.diff(survival))
        if mass.min() < -1e-12:
            raise FloatingPointError('Nonmonotone continuous output probability')
        mass = np.maximum(mass, 0.)
        if abs(mass.sum()-1) > 1e-8:
            raise FloatingPointError('Continuous output probability does not sum to one')
        centers = (edges[:-1]+edges[1:])/2
        density = np.zeros(len(grid))
        for start in range(0, bins, 512):
            z = (grid[:, None]-centers[start:start+512])/bandwidth
            density += (np.exp(-.5*z*z)*mass[start:start+512]).sum(1)/(np.sqrt(2*np.pi)*bandwidth)
        density = np.maximum(density, 1e-16)
        error = float(np.abs(np.log10(density[:-1])-np.log10(truth_pdf[:-1])).sum()*step/scale)
        curves.append(density)
        errors.append(error)
        metadata.append(dict(output_bins=bins, probability_mass=float(mass.sum()),
                             bandwidth=bandwidth, method='compiled output inversion and exact triangle probability integration'))
    return np.array(curves), errors, metadata
