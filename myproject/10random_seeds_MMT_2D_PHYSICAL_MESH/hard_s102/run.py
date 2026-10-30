"""Two MLPs, exact reference law, MSE-only fitting, 100 sequential USLW queries."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'mpl_cache'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import json
import time
import numpy as np
import torch
from scipy.stats import kstest
from scipy.stats import gaussian_kde
from model import HardPDFNet
from problem import ROOT, prepare, from_unit, oracle, to_unit, weighted_quantiles, law_copula
from training import fit
from acquisition import select_query
from continuous_pdf import continuous_member_pdfs

def smoothed_pdf(values, grid, w=None):
    """nls-style density: scipy gaussian_kde (Scott bandwidth) on a fixed window,
    floored at 1e-16. The MMT evaluation set is a UNIFORM grid, so every call
    must pass the input-law weights w (dnosearch ptheta weights). Predictions
    outside the window decay to the floor automatically - no bounds check."""
    values = np.asarray(values, dtype=float).ravel()
    if w is None:
        kde = gaussian_kde(values)
    else:
        w = np.asarray(w, dtype=float).ravel()
        kde = gaussian_kde(values, weights=w/w.sum())
    return np.maximum(kde.evaluate(np.asarray(grid, dtype=float)), 1e-16)

def metrics(pred, truth, threshold, w=None):
    """With w: input-law weighted distribution metrics (uniform grid +
    input-law weights). Without w: plain unweighted domain metrics."""
    pred=np.asarray(pred,float);truth=np.asarray(truth,float)
    err = pred-truth
    extreme = truth>=threshold
    predicted = pred>=threshold
    if w is None: w=np.ones(len(truth))
    w=w/np.sum(w)
    overlap = np.sum(w[extreme & predicted])
    return dict(rmse=float(np.sqrt(np.sum(w*err**2))),
                relative_l2=float(np.sqrt(np.sum(w*err**2))/np.sqrt(np.sum(w*truth**2))),
                r2=float(1-np.sum(w*err**2)/np.sum(w*(truth-np.sum(w*truth))**2)),
                mae=float(np.sum(w*abs(err))),max_abs=float(abs(err).max()),
                top1_rmse=float(np.sqrt(np.sum(w[extreme]*err[extreme]**2)/np.sum(w[extreme]))),
                top1_recall=float(overlap/max(1e-300,np.sum(w[extreme]))),
                top1_precision=float(overlap/max(1e-300,np.sum(w[predicted]))))

def evaluate(compiled,scale,evaluation,ref,observed_x,observed_y):
    # model-input coordinates: the INPUT-LAW PIT copula, so the probability
    # mesh measures the input law and the exact constraint targets the
    # weighted reference law (TOY precedent: to_unit was ndtr there)
    umc,ugrid = law_copula(evaluation['x_mc']),law_copula(evaluation['x_grid'])
    w_mc = evaluation['w']  # normalized input-law weights of the uniform grid
    pairs = [m.predict(umc,True) for m in compiled]
    mc = np.array([p[0]*scale for p in pairs])
    grid = np.array([m.predict(ugrid)*scale for m in compiled])
    threshold = float(np.interp(.99,ref['p'],ref['q']))
    row = metrics(mc.mean(0),evaluation['y_mc'],threshold,w_mc)
    row.update({f'grid_{k}':v for k,v in metrics(grid.mean(0),evaluation['y_grid'],threshold).items()})
    train = np.array([m.predict(law_copula(observed_x))*scale for m in compiled])
    row['train_mse_members'] = np.mean((train-observed_y)**2,axis=1).tolist()
    row['train_mse_mean'] = float(np.mean((train.mean(0)-observed_y)**2))
    row['disagreement_rms'] = float(np.sqrt(np.mean((mc[0]-mc[1])**2)))
    # rank uniformity under the INPUT LAW: the grid is uniform, so resample it
    # by the input-law weights before the KS test (fixed seed, reproducible)
    rng_rank = np.random.RandomState(20261028)
    idx = rng_rank.choice(len(w_mc), size=4096, replace=True, p=w_mc/w_mc.sum())
    row['rank_ks_members'] = [float(kstest(p[1][idx],'uniform').statistic) for p in pairs]
    row['member_metrics'] = [metrics(a,evaluation['y_mc'],threshold,w_mc) for a in mc]
    qt = np.linspace(0,1,10001)
    qref = np.interp(qt,ref['p'],ref['q'])
    row['mean_w1_vs_reference'] = float(np.mean(abs(weighted_quantiles(mc.mean(0),w_mc,qt)-qref)))
    row['member_w1_vs_reference'] = [float(np.mean(abs(weighted_quantiles(a,w_mc,qt)-qref))) for a in mc]
    # nls-style PDF-error window (fixed 1000-point grid; dnosearch nls uses
    # np.linspace(0,1,1000) the same way).
    # TODO(6): the LOWER bound 0 assumes y >= 0; change it for negative-capable y.
    # The UPPER bound only needs to safely cover the truth support.
    y_int = np.linspace(0.0, max(scale, float(evaluation['y_mc'].max())), 1000)
    dy = y_int[1]-y_int[0]
    density_true = smoothed_pdf(evaluation['y_mc'], y_int, w_mc)
    densities = np.array([smoothed_pdf(a, y_int, w_mc) for a in [*mc,mc.mean(0)]])
    # Left-rectangle integral of |log10 density difference| over the window.
    errors = (np.sum(abs(np.log10(densities[:,:-1])-
                     np.log10(density_true[:-1])),axis=1)*dy/scale)
    row['pdf_e_members'] = errors[:2].tolist()
    row['pdf_e_mean'] = float(errors[2])
    row['pdf_evaluation_binning'] = 'gaussian_kde (Scott bandwidth, p_X-weighted)'
    row['pdf_l1_mean'] = float(np.sum(abs(densities[2]-density_true))*dy)
    peak = evaluation['x_grid'][evaluation['y_grid'].argmax()]
    peaks = evaluation['x_grid'][np.argmax(np.r_[grid,grid.mean(0,keepdims=True)],axis=1)]
    row['true_peak_grid'] = peak.tolist()
    row['predicted_peaks_grid'] = peaks.tolist()
    row['peak_distance_mean'] = float(np.linalg.norm(peaks[2]-peak))
    arrays = dict(mc=mc,grid=grid,ranks=np.array([p[1] for p in pairs]),
                  pdf_grid=y_int,pdf_true=density_true,pdf_predictions=densities,
                  train_predictions=train,threshold=threshold)
    # Retain the historical grid-KDE metrics and add member-wise integration
    # of the actual continuous model. The ensemble average has no hard-law guarantee.
    continuous_pdf, continuous_errors, continuous_meta = continuous_member_pdfs(
        compiled, scale, evaluation, y_int, density_true, mc)
    row['pdf_e_members_continuous'] = continuous_errors
    row['pdf_e_members_grid_kde'] = row['pdf_e_members'].copy()
    row['pdf_e_continuous_metadata'] = continuous_meta
    arrays['pdf_members_continuous'] = continuous_pdf
    order = np.argsort(evaluation['y_mc'])
    lo, hi = np.interp([.001, .999], np.cumsum(w_mc[order])/w_mc.sum(), evaluation['y_mc'][order])
    both = (evaluation['y_mc'] <= lo) | (evaluation['y_mc'] >= hi)
    upper = evaluation['y_mc'] >= hi
    row['extreme_mse_members'] = [float(np.average((a[both]-evaluation['y_mc'][both])**2, weights=w_mc[both])) for a in mc]
    row['upper_tail_mse_members'] = [float(np.average((a[upper]-evaluation['y_mc'][upper])**2, weights=w_mc[upper])) for a in mc]
    return row,arrays

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--adam',type=int,default=1800)
    parser.add_argument('--lbfgs',type=int,default=450)
    parser.add_argument('--mesh',type=int,default=32)
    parser.add_argument('--width',type=int,default=230)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--acq',default='uslw',choices=['us','uslw'],help='acquisition: us = variance only, uslw = variance*pX/KDE(mean)')
    parser.add_argument('--name',default='uslw_seed101')
    parser.add_argument('--lhs-seed',type=int,default=101,help='RandomState seed of the 3-point LHS initial design')
    parser.add_argument('--seed-base',type=int,default=1,help='member seed base: round k uses [base+2k, base+2k+1]')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    if not (ROOT/'data/reference_law.npz').exists(): prepare()
    ref=np.load(ROOT/'data/reference_law.npz')
    evaluation=np.load(ROOT/'data/evaluation.npz')
    scale=float(ref['scale'])
    out=ROOT/'results'/args.name
    out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config.pop('resume')
    config.update(initial_points=3,lhs_seed=args.lhs_seed,candidates=1000,
                  layers=[2]+[args.width]*5+[1],activation='ReLU',
                  initialization='same mlp_repro.seeded_member_init as ordinary baseline',
                  precision='float32 MLP as ordinary baseline; float64 exact probability transform',
                  parameter_count_per_member=2*args.width+args.width+4*(args.width*args.width+args.width)+args.width+1,
                  restart=f'cold each round; seeds {args.seed_base}+2k,{args.seed_base}+2k+1; k=number of acquired labels',
                  objective='MSE(y/scale) with scale=1 (raw y, mmt_mlp convention)',
                  input_distribution='N(0,I) density on design domain [-6,6]^2; LHS/candidates box-uniform; hard probability mesh integrates the box-conditioned law',
                  score_features='physical x/6 at mesh vertices, float32 MLP',
                  mesh_geometry='32x32 physical-uniform intervals mapped to nonuniform Gaussian probability coordinates; exact triangle probability areas',
                  reference_knots=len(ref['p']),
                  pdf_evaluation='legacy grid-KDE metrics retained; pdf_e_members_continuous integrates each actual compiled member using the same prediction KDE bandwidth',
                  selection=f'argmax over 1000 box-uniform candidates, acq={args.acq}: sigma^2(x) or sigma^2(x)*p_X(x)/weighted-KDE(mu(x))')
    if (out/'config.json').exists():
        assert args.resume,'Existing run: use --resume to avoid overwrite'
        assert json.loads((out/'config.json').read_text())==config
    else:(out/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];start=0
    rng_init=np.random.RandomState(args.lhs_seed)
    u_init=np.column_stack([(rng_init.permutation(3)+rng_init.rand(3))/3 for _ in range(2)])
    if args.resume and (out/'history.json').exists():
        rows=json.loads((out/'history.json').read_text())
        state=np.load(out/f'round_{len(rows)-1:03d}_next_state.npz');x,y=state['x'],state['y']
        start=len(rows)
    else:
        x=from_unit(u_init);y=oracle(x)
        np.savez_compressed(out/'initial_lhs.npz',x=x,u=to_unit(x),y=y)
    began=time.perf_counter()
    for k in range(start,args.iterations+1):
        logs=[];models=[]
        for seed in [args.seed_base+2*k,args.seed_base+2*k+1]:
            model=HardPDFNet(ref['p'],ref['q']/scale,seed=seed,
                            na=args.mesh,nv=args.mesh,width=args.width).to(args.device)
            logs.append(fit(model,law_copula(x),y/scale,f'round_{k:03d}',out,
                            args.adam,args.lbfgs))
            models.append(model)
        compiled=[m.compile() for m in models]
        row,arrays=evaluate(compiled,scale,evaluation,ref,x,y)
        row.update(iteration=k,n=len(x),training=logs,elapsed_this_session=time.perf_counter()-began)
        if k:
            previous=np.load(out/f'round_{k-1:03d}_selection.npz')
            query=x[-1:]
            before=previous['members'][:,int(previous['selected_index'])]
            after=np.array([m.predict(law_copula(query))[0]*scale for m in compiled])
            row['previous_query_variance_before']=float(before.var())
            row['previous_query_variance_after']=float(after.var())
            row['previous_query_abs_error_before']=float(abs(before.mean()-y[-1]))
            row['previous_query_abs_error_after']=float(abs(after.mean()-y[-1]))
        np.savez_compressed(out/f'round_{k:03d}_predictions.npz',**arrays)
        if k<args.iterations:
            query,selection=select_query(x,compiled,scale,acq=args.acq)
            # This is the only call to the oracle for an actively selected label.
            observed=oracle(query)
            np.savez_compressed(out/f'round_{k:03d}_selection.npz',**selection,
                                query=query,observed=observed)
            row.update(query=query[0].tolist(),observed_y=float(observed[0]),
                       selected_index=selection['selected_index'],
                       acquisition_max=float(selection['scores'].max()))
            x=np.vstack((x,query));y=np.r_[y,observed]
        np.savez_compressed(out/f'round_{k:03d}_next_state.npz',x=x,y=y)
        rows.append(row)
        np.savez_compressed(out/'observations.npz',x=x,y=y)
        temporary=out/'history.tmp'
        temporary.write_text(json.dumps(rows,indent=2),encoding='utf8')
        temporary.replace(out/'history.json')
        print(f'COMPLETED queries={k} n={row["n"]} RMSE={row["rmse"]:.6f} '
              f'R2={row["r2"]:.5f} top1_RMSE={row["top1_rmse"]:.5f} '
              f'peak_error={row["peak_distance_mean"]:.3f} e_continuous_members={row["pdf_e_members_continuous"]} '
              f'e_grid_members={row["pdf_e_members"]} extreme_mse_members={row["extreme_mse_members"]}',flush=True)
    print('FINISHED',out,flush=True)

if __name__=='__main__':main()
