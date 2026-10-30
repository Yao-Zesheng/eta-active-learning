"""Local, self-contained copy of the mil-project MLP ensemble machinery.

The code below mirrors dnosearch.core.mlp (make_mlp, _seeded_member_init,
EnsembleMLP) as used by mil_2d1d_tReTaoMean/pdf_workflow.py, so this repo no
longer imports from outside its own folder. Behaviour is line-for-line the
same; the only additions are (a) an optional `device` that actually moves the
nets (needed for the lambda-sweep on GPU) and (b) the W1 (quantile) penalty
option in training, used only when lambd > 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_mlp(layers, activation=nn.ReLU):
    modules = []
    for i, (a, b) in enumerate(zip(layers[:-1], layers[1:])):
        modules.append(nn.Linear(a, b))
        if i < len(layers) - 2:
            modules.append(activation())
    return nn.Sequential(*modules)


def seeded_member_init(net, member_seed=None):
    """Legacy fixed-init draw order, copied from dnosearch.core.mlp."""
    rng = np.random.RandomState(member_seed)
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                std = np.float32(np.sqrt(2.0 / (layer.in_features + layer.out_features)))
                w = rng.randn(layer.in_features, layer.out_features).astype(np.float32) * std
                if member_seed is not None:
                    w = np.clip(w, -2 * std, 2 * std)
                else:
                    bad = np.abs(w) > 2 * std
                    while bad.any():
                        w[bad] = rng.randn(bad.sum()) * std
                        bad = np.abs(w) > 2 * std
                    w /= 0.8796256610342398
                layer.weight.copy_(torch.from_numpy(w.T.copy()))
                layer.bias.zero_()


def _tail_index_refresh(Yaux, quantiles):
    """train_utils.update_idx: indices realizing the target scalar quantiles."""
    _, indices = torch.sort(Yaux)
    positions = (quantiles * (Yaux.size(0) - 1)).floor().long()
    return indices[positions]


class MLPEnsemble:
    """Two-or-more MLPs trained on the same data; mil's EnsembleMLP equivalent.

    Inputs Theta are already scaled by the caller; Y is physical and is
    normalized by dno_y_transform. Training is full-batch Adam(lr, eps=1e-8)
    for `epochs` steps, final model returned (no best-epoch selection), as in
    mil. With lambd > 0 the loss becomes MSE + lambd * W1(quantiles) on a
    large auxiliary input set X_aux against reference quantiles Ytail_ref,
    with the tail-index refresh every `omega` steps (eta-recipe).
    """

    def __init__(self, Theta, Y, layers, lr, epochs, N, DNO_Y_transform,
                 DNO_Y_itransform, member_seeds, device="cpu",
                 X_aux=None, Ytail_ref=None, quantiles=None,
                 lambd=0.0, omega=50):
        if N < 1 or epochs < 1:
            raise ValueError("N and epochs must be positive")
        if member_seeds is not None and len(member_seeds) != N:
            raise ValueError("member_seeds must contain one seed per network")
        self.N, self.epochs = N, epochs
        self.Theta = np.asarray(Theta)
        self.Y = np.asarray(DNO_Y_transform(Y)).reshape(-1, 1)
        self.DNO_Y_itransform = DNO_Y_itransform
        self.device = torch.device(device)
        self.model, self.loss_history = [], []
        x = torch.as_tensor(self.Theta, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(self.Y, dtype=torch.float32, device=self.device)
        use_w1 = lambd > 0.0
        if use_w1:
            x_aux = torch.as_tensor(np.asarray(X_aux), dtype=torch.float32, device=self.device)
            ytail_ref = torch.as_tensor(np.asarray(Ytail_ref), dtype=torch.float32, device=self.device)
            quantiles = torch.as_tensor(quantiles, dtype=torch.float32, device=self.device)
        for i in range(N):
            net = make_mlp(layers)
            seeded_member_init(net, int(member_seeds[i]))
            net.to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-8)
            if use_w1:
                net.eval()
                with torch.no_grad():
                    yaux = net(x_aux).squeeze(-1)
                    tail_idx = _tail_index_refresh(yaux, quantiles)
                net.train()
            losses = []
            for step in range(1, epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = (net(x) - y).square().mean()
                if use_w1:
                    tail_out = net(x_aux[tail_idx]).squeeze(-1)
                    loss = loss + lambd * torch.mean(torch.abs(ytail_ref - tail_out))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if use_w1 and step % omega == 0:
                    net.eval()
                    with torch.no_grad():
                        yaux = net(x_aux).squeeze(-1)
                        tail_idx = _tail_index_refresh(yaux, quantiles)
                    net.train()
            net.eval()
            self.model.append(net)
            self.loss_history.append(losses)

    def predict_all(self, Thetanew):
        x = torch.as_tensor(np.atleast_2d(Thetanew), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return np.column_stack([m(x).cpu().numpy().ravel() for m in self.model])

    def predict(self, Thetanew):
        values = self.DNO_Y_itransform(self.predict_all(Thetanew).astype(float))
        return values.mean(1)[:, None], values.var(1)[:, None]

    predict_noiseless = predict

    def training(self):
        return np.asarray(self.loss_history).T
