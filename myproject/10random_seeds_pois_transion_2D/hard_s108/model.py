"""Exact marginal-law architecture for a uniform two-dimensional input.

A neural net specifies vertex scores on a fixed triangular mesh. The deployed
score is the continuous affine interpolant, NOT the raw neural net between
vertices. Its CDF under the uniform input law is an area-weighted sum of exact
triangle CDFs. Q_ref(F_score(score(x))) has law Q_ref#Uniform for every admissible
parameter setting. Constant triangles are rejected, never silently smoothed.
No empirical ranks, KDE, soft sorting, or distribution penalty are used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from mlp_repro import make_mlp, seeded_member_init


def triangle_cdf_torch(t, abc):
    """CDF of an affine scalar at a uniform point in each triangle.

    abc: (n_triangles,3), sorted or unsorted vertex values. Repeated minima or
    maxima are supported exactly; three equal vertices (an atom) are invalid.
    """
    a, b, c = abc.sort(dim=-1).values.unbind(-1)
    if bool(torch.any(c == a)):
        raise ValueError("Constant triangle: score law has an atom; update invalid")
    t = t.reshape(-1, 1)
    z = torch.minimum(torch.maximum(t, a), c)
    dl, dr = (b-a)*(c-a), (c-a)*(c-b)
    # A zero denominator occurs only in an inactive branch. Replacing that
    # denominator by one is exact, unlike adding epsilon to active denominators.
    low = (z-a).square() / torch.where(dl == 0, torch.ones_like(dl), dl)
    high = 1 - (c-z).square() / torch.where(dr == 0, torch.ones_like(dr), dr)
    out = torch.where(z < b, low, high)
    return torch.where(t <= a, torch.zeros_like(out),
                       torch.where(t >= c, torch.ones_like(out), out))


def triangle_cdf_numpy(t, abc, weights, chunk=128):
    a, b, c = np.sort(abc, axis=1).T
    if np.any(c == a):
        raise ValueError("Constant triangle")
    dl, dr = (b-a)*(c-a), (c-a)*(c-b)
    dl, dr = np.where(dl == 0, 1., dl), np.where(dr == 0, 1., dr)
    result = []
    for start in range(0, len(t), chunk):
        v = np.asarray(t[start:start+chunk])[:, None]
        z = np.minimum(np.maximum(v, a), c)
        f = np.where(z < b, (z-a)**2/dl, 1-(c-z)**2/dr)
        f = np.where(v <= a, 0., np.where(v >= c, 1., f))
        result.append(f @ weights)
    return np.concatenate(result)


def mesh(na=32, nv=32):
    # Uniform probability coordinates; no truth-dependent knots or periodicity.
    ax = np.linspace(0, 1, na+1)
    ay = np.linspace(0, 1, nv+1)
    uu, vv = np.meshgrid(ax, ay, indexing="ij")
    vertices = np.column_stack((uu.ravel(), vv.ravel()))
    triangles, areas = [], []
    for i in range(na):
        for j in range(nv):
            a = i*(nv+1)+j
            b, c, d = a+nv+1, a+1, a+nv+2
            triangles.extend(((a, b, d), (a, d, c)))
            areas.extend([.5*(ax[i+1]-ax[i])/nv]*2)
    return ax, ay, vertices, np.array(triangles), np.array(areas)


def locate_numpy(x, ax, nv):
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-12) or np.any(x > 1+1e-12):
        raise ValueError("Input outside unit square")
    u = np.clip(x[:, 0], 0, 1)
    v = np.clip(x[:, 1], 0, 1)
    i = np.clip(np.searchsorted(ax, u, side="right")-1, 0, len(ax)-2)
    j = np.minimum((v*nv).astype(int), nv-1)
    p, q = (u-ax[i])/(ax[i+1]-ax[i]), v*nv-j
    a = i*(nv+1)+j
    b, c, d = a+nv+1, a+1, a+nv+2
    lower = q <= p
    ids = np.column_stack((a, np.where(lower, b, d), np.where(lower, d, c)))
    bary = np.column_stack((np.where(lower, 1-p, 1-q),
                            np.where(lower, p-q, p), np.where(lower, q, q-p)))
    return ids, bary


class ReferenceQuantile(nn.Module):
    def __init__(self, p, q):
        super().__init__()
        if np.any(np.diff(p) <= 0) or np.any(np.diff(q) <= 0):
            raise ValueError("Reference knots must be strictly increasing")
        self.register_buffer("p", torch.as_tensor(p, dtype=torch.float64))
        self.register_buffer("q", torch.as_tensor(q, dtype=torch.float64))

    def forward(self, u):
        u = u.clamp(0, 1)  # roundoff protection only; no tail truncation
        k = torch.searchsorted(self.p, u.contiguous(), right=True)-1
        k = k.clamp(0, len(self.p)-2)
        w = (u-self.p[k])/(self.p[k+1]-self.p[k])
        return self.q[k] + w*(self.q[k+1]-self.q[k])


class HardPDFNet(nn.Module):
    def __init__(self, p, q, seed=1, na=32, nv=32, width=200):
        super().__init__()
        self.na, self.nv, self.width, self.seed = na, nv, width, seed
        ax, ay, vertices, tri, areas = mesh(na, nv)
        self.ax_numpy = ax
        self.register_buffer("vertices", torch.tensor(vertices, dtype=torch.float64))
        self.register_buffer("triangles", torch.tensor(tri, dtype=torch.long))
        self.register_buffer("areas", torch.tensor(areas, dtype=torch.float64))
        self.quantile = ReferenceQuantile(p, q)
        # Exactly the ordinary baseline's MLP topology, activation, biases and
        # seeded initialization and float32 precision. The analytic CDF operates
        # on these vertex values cast to float64; no approximation of the CDF.
        self.layers = [2] + [width]*5 + [1]
        self.net = make_mlp(self.layers)
        seeded_member_init(self.net, seed)

    def vertex_scores(self):
        features = (2*self.vertices-1).to(dtype=next(self.net.parameters()).dtype)
        raw = self.net(features).squeeze(-1).double()
        # Positive affine normalization changes neither ranks nor final output.
        std = raw.std(unbiased=False)
        if not bool(torch.isfinite(std)) or float(std.detach()) < 1e-12:
            raise ValueError("Degenerate global score")
        return (raw-raw.mean())/std

    def locate(self, x):
        ids, bary = locate_numpy(x, self.ax_numpy, self.nv)
        return (torch.as_tensor(ids, device=self.vertices.device),
                torch.as_tensor(bary, device=self.vertices.device, dtype=torch.float64))

    def forward_located(self, located, return_rank=False):
        ids, bary = located
        vertices = self.vertex_scores()
        scores = (vertices[ids]*bary).sum(dim=1)
        u = triangle_cdf_torch(scores, vertices[self.triangles]) @ self.areas
        out = self.quantile(u)
        return (out, u) if return_rank else out

    def forward(self, x):
        return self.forward_located(self.locate(x))

    @torch.no_grad()
    def compile(self):
        return CompiledModel(self)


class CompiledModel:
    """Exact piecewise quadratic CDF representation for fast inference.

    Between consecutive vertex scores the CDF is quadratic. Its values at the
    endpoints and midpoint determine it exactly; this is not a fitted CDF.
    """
    def __init__(self, model):
        self.ax, self.nv = model.ax_numpy, model.nv
        self.scores = model.vertex_scores().cpu().numpy()
        self.tri = model.triangles.cpu().numpy()
        self.areas = model.areas.cpu().numpy()
        self.p = model.quantile.p.cpu().numpy()
        self.q = model.quantile.q.cpu().numpy()
        self.knots = np.unique(self.scores)
        mid = (self.knots[:-1]+self.knots[1:])/2
        vals = triangle_cdf_numpy(np.r_[self.knots, mid],
                                  self.scores[self.tri], self.areas)
        f = vals[:len(self.knots)]
        fm = vals[len(self.knots):]
        self.f0 = f[:-1]
        self.b = 2*(f[1:]+f[:-1]-2*fm)
        self.a = f[1:]-f[:-1]-self.b
        self.min_triangle_span = float(np.ptp(self.scores[self.tri], axis=1).min())

    def cdf(self, s):
        k = np.clip(np.searchsorted(self.knots, s, side="right")-1,
                    0, len(self.knots)-2)
        z = (s-self.knots[k])/(self.knots[k+1]-self.knots[k])
        out = self.f0[k]+z*(self.a[k]+z*self.b[k])
        return np.where(s <= self.knots[0], 0.,
                        np.where(s >= self.knots[-1], 1., np.clip(out, 0, 1)))

    def predict(self, x, return_rank=False):
        ids, bary = locate_numpy(x, self.ax, self.nv)
        s = (self.scores[ids]*bary).sum(axis=1)
        rank = self.cdf(s)
        y = np.interp(rank, self.p, self.q)
        return (y, rank) if return_rank else y
