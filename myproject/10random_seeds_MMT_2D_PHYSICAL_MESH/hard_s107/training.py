from copy import deepcopy
import time
import numpy as np
import torch
SCALE = 1.0

def fit(model, x, y, stage, out, adam_steps, lbfgs_steps, lr=.002, stop_at_goal=True):
    """Only supervised output MSE; best state also selected only by this MSE."""
    located = model.locate(x)
    target = torch.tensor(y, dtype=torch.float64, device=model.vertices.device)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    best_loss, best_state, trace = float('inf'), None, []
    started = time.perf_counter()
    # y is normalized by the fixed reference maximum before training.
    goal = 1e-8
    def record(loss, step, phase, force=False):
        nonlocal best_loss, best_state
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite supervised loss')
        if value < best_loss:
            best_loss = value
            best_state = deepcopy(model.net.state_dict())
        if step % 100 == 0 or force:
            trace.append(dict(phase=phase, step=step, mse=value,
                              rmse_normalized=np.sqrt(value),
                              elapsed=time.perf_counter()-started))
        return value
    adam_done = 0
    rejected_steps = []
    for step in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        try:
            pred = model.forward_located(located)
        except ValueError as exc:
            if 'Constant triangle' not in str(exc) and 'Degenerate global score' not in str(exc):
                raise
            if best_state is None:
                raise
            rejected_steps.append(dict(phase='adam',step=step,reason=str(exc)))
            model.net.load_state_dict(best_state)
            # Reset momentum as well as step size after an inadmissible step.
            lr *= .5
            opt = torch.optim.Adam(model.net.parameters(), lr=lr)
            if len(rejected_steps)>=10:
                break
            continue
        loss = (pred-target).square().mean()
        value = record(loss, step, 'adam')
        if stop_at_goal and value <= goal:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 10.)
        opt.step()
        adam_done += 1
    with torch.no_grad():
        try:
            record((model.forward_located(located)-target).square().mean(),
                   adam_done, 'adam', True)
        except ValueError as exc:
            if 'Constant triangle' not in str(exc) and 'Degenerate global score' not in str(exc):
                raise
            rejected_steps.append(dict(phase='adam_final',step=adam_done,reason=str(exc)))
    model.net.load_state_dict(best_state)
    closures = 0
    if (best_loss > goal or not stop_at_goal) and lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.net.parameters(), lr=.8,
                    max_iter=lbfgs_steps, max_eval=lbfgs_steps*2,
                    tolerance_grad=1e-11, tolerance_change=1e-14,
                    history_size=50, line_search_fn='strong_wolfe')
        def closure():
            nonlocal closures
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_located(located)
            loss = (pred-target).square().mean()
            record(loss, closures, 'lbfgs')
            loss.backward()
            closures += 1
            return loss
        try:
            optimizer.step(closure)
            with torch.no_grad():
                record((model.forward_located(located)-target).square().mean(),
                       closures, 'lbfgs', True)
        except ValueError as exc:
            if 'Constant triangle' not in str(exc) and 'Degenerate global score' not in str(exc):
                raise
            # A line-search trial left the admissible set. Terminate this phase
            # and restore the best VALID state; never regularize away an atom.
            rejected_steps.append(dict(phase='lbfgs_trial',step=closures,reason=str(exc)))
            print(f'{stage} member={model.seed}: rejected invalid L-BFGS trial; restoring best valid MSE state',flush=True)
    model.net.load_state_dict(best_state)
    with torch.no_grad():
        prediction = model.forward_located(located).cpu().numpy()
    elapsed = time.perf_counter()-started
    result = dict(stage=stage, seed=model.seed, n=len(x),
                  mse=float(np.mean((prediction-y)**2)),
                  rmse_normalized=float(np.sqrt(np.mean((prediction-y)**2))),
                  max_abs_error_normalized=float(np.max(abs(prediction-y))),
                  elapsed_seconds=elapsed, adam_updates=adam_done, lbfgs_evaluations=closures,
                  stop_at_goal=stop_at_goal,
                  rejected_invalid_steps=rejected_steps,
                  trace=trace)
    torch.save(dict(state_dict=model.state_dict(), seed=model.seed,
                    na=model.na, nv=model.nv, width=model.width,
                    layers=model.layers, activation='ReLU',
                    mesh_type=model.mesh_type, score_features=model.score_features),
               out/f'{stage}_member{model.seed}.pt')
    print(f'{stage} member={model.seed} n={len(x)} normalized_RMSE={result["rmse_normalized"]:.5g} '
          f'max={result["max_abs_error_normalized"]:.5g} time={elapsed:.1f}s', flush=True)
    return result


