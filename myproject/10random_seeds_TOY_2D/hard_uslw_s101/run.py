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
from scipy.ndimage import gaussian_filter1d
from model import HardPDFNet
from problem import ROOT, prepare, lhs, oracle, to_unit
from training import fit
from acquisition import select_query

def smoothed_pdf(values, edges, bandwidth):
    # Linear mass deposition preserves each observation's first moment and
    # avoids the bias of snapping the large near-zero probability mass to a bin.
    dx=edges[1]-edges[0]
    centers=(edges[1:]+edges[:-1])/2
    z=(np.asarray(values)-centers[0])/dx
    lo=np.floor(z).astype(int);w=z-lo
    if np.any(lo<0) or np.any(lo>=len(centers)-1):
        raise ValueError('PDF grid must include every observation with padding')
    mass=(np.bincount(lo,weights=1-w,minlength=len(centers))+
          np.bincount(lo+1,weights=w,minlength=len(centers)))
    hist=mass/(len(values)*dx)
    # 4-sigma truncation creates artificial zeros relevant to log-density error.
    # 10 sigma retains tails well below the 1e-16 density floor.
    return gaussian_filter1d(hist.astype(float),bandwidth/np.diff(edges)[0],mode='constant',truncate=10.)

def metrics(pred, truth, threshold):
    err = pred-truth
    extreme = truth>=threshold
    predicted = pred>=threshold
    overlap = np.sum(extreme & predicted)
    return dict(rmse=float(np.sqrt(np.mean(err**2))),
                relative_l2=float(np.linalg.norm(err)/np.linalg.norm(truth)),
                r2=float(1-np.mean(err**2)/np.var(truth)),
                mae=float(np.mean(abs(err))),max_abs=float(abs(err).max()),
                top1_rmse=float(np.sqrt(np.mean(err[extreme]**2))),
                top1_recall=float(overlap/max(1,extreme.sum())),
                top1_precision=float(overlap/max(1,predicted.sum())))

def evaluate(compiled,scale,evaluation,ref,observed_x,observed_y):
    umc,ugrid = to_unit(evaluation['x_mc']),to_unit(evaluation['x_grid'])
    pairs = [m.predict(umc,True) for m in compiled]
    mc = np.array([p[0]*scale for p in pairs])
    grid = np.array([m.predict(ugrid)*scale for m in compiled])
    threshold = float(np.interp(.99,ref['p'],ref['q']))
    row = metrics(mc.mean(0),evaluation['y_mc'],threshold)
    row.update({f'grid_{k}':v for k,v in metrics(grid.mean(0),evaluation['y_grid'],threshold).items()})
    train = np.array([m.predict(to_unit(observed_x))*scale for m in compiled])
    row['train_mse_members'] = np.mean((train-observed_y)**2,axis=1).tolist()
    row['train_mse_mean'] = float(np.mean((train.mean(0)-observed_y)**2))
    row['disagreement_rms'] = float(np.sqrt(np.mean((mc[0]-mc[1])**2)))
    row['rank_ks_members'] = [float(kstest(p[1],'uniform').statistic) for p in pairs]
    row['member_metrics'] = [metrics(a,evaluation['y_mc'],threshold) for a in mc]
    qt = np.linspace(0,1,10001)
    row['mean_w1_vs_reference'] = float(np.mean(abs(np.quantile(mc.mean(0),qt)-np.interp(qt,ref['p'],ref['q']))))
    row['member_w1_vs_reference'] = [float(np.mean(abs(np.quantile(a,qt)-np.interp(qt,ref['p'],ref['q'])))) for a in mc]
    edges = np.linspace(-.1,.7,4097)
    centers = (edges[1:]+edges[:-1])/2
    h = evaluation['y_mc'].std(ddof=1)*len(evaluation['y_mc'])**(-.2)
    density_true = smoothed_pdf(evaluation['y_mc'],edges,h)
    densities = np.array([smoothed_pdf(a,edges,h) for a in [*mc,mc.mean(0)]])
    region = (centers>=0)&(centers<=scale)
    step = centers[1]-centers[0]
    # Dimensionless normalized-y integral over [0, reference max], log10/floor.
    errors = (np.sum(abs(np.log10(np.maximum(densities[:,region],1e-16))-
                     np.log10(np.maximum(density_true[region],1e-16))),axis=1)*step/scale)
    row['pdf_e_members'] = errors[:2].tolist()
    row['pdf_e_mean'] = float(errors[2])
    row['pdf_evaluation_kernel_truncate_sigma'] = 10.
    row['pdf_evaluation_binning'] = 'linear mass deposition'
    row['pdf_l1_mean'] = float(np.sum(abs(densities[2]-density_true))*step)
    peak = evaluation['x_grid'][evaluation['y_grid'].argmax()]
    peaks = evaluation['x_grid'][np.argmax(np.r_[grid,grid.mean(0,keepdims=True)],axis=1)]
    row['true_peak_grid'] = peak.tolist()
    row['predicted_peaks_grid'] = peaks.tolist()
    row['peak_distance_mean'] = float(np.linalg.norm(peaks[2]-peak))
    arrays = dict(mc=mc,grid=grid,ranks=np.array([p[1] for p in pairs]),
                  pdf_grid=centers,pdf_true=density_true,pdf_predictions=densities,
                  train_predictions=train,threshold=threshold,bandwidth=h)
    return row,arrays

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--adam',type=int,default=1800)
    parser.add_argument('--lbfgs',type=int,default=450)
    parser.add_argument('--mesh',type=int,default=32)
    parser.add_argument('--width',type=int,default=200)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--name',default='uslw_seed101')
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
    config.update(initial_points=3,lhs_seed=101,candidates=1000,
                  layers=[2]+[args.width]*5+[1],activation='ReLU',
                  initialization='same mlp_repro.seeded_member_init as ordinary baseline',
                  precision='float32 MLP as ordinary baseline; float64 exact probability transform',
                  parameter_count_per_member=2*args.width+args.width+4*(args.width*args.width+args.width)+args.width+1,
                  restart='cold each round; seeds 2*k+1,2*k+2; k=number of acquired labels',
                  objective='MSE(y/reference_max)',input_distribution='N(0,10I)',
                  selection='argmax of sigma^2(x)*p_X(x)/KDE(mu(x)) over 1000 Gaussian candidates')
    if (out/'config.json').exists():
        assert args.resume,'Existing run: use --resume to avoid overwrite'
        assert json.loads((out/'config.json').read_text())==config
    else:(out/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];x=lhs();y=oracle(x);start=0
    if args.resume and (out/'history.json').exists():
        rows=json.loads((out/'history.json').read_text())
        state=np.load(out/f'round_{len(rows)-1:03d}_next_state.npz');x,y=state['x'],state['y']
        start=len(rows)
    np.savez_compressed(out/'initial_lhs.npz',x=lhs(),u=to_unit(lhs()),y=oracle(lhs()))
    began=time.perf_counter()
    for k in range(start,args.iterations+1):
        logs=[];models=[]
        for seed in [2*k+1,2*k+2]:
            model=HardPDFNet(ref['p'],ref['q']/scale,seed=seed,
                            na=args.mesh,nv=args.mesh,width=args.width).to(args.device)
            logs.append(fit(model,to_unit(x),y/scale,f'round_{k:03d}',out,
                            args.adam,args.lbfgs))
            models.append(model)
        compiled=[m.compile() for m in models]
        row,arrays=evaluate(compiled,scale,evaluation,ref,x,y)
        row.update(iteration=k,n=len(x),training=logs,elapsed_this_session=time.perf_counter()-began)
        if k:
            previous=np.load(out/f'round_{k-1:03d}_selection.npz')
            query=x[-1:]
            before=previous['members'][:,int(previous['selected_index'])]
            after=np.array([m.predict(to_unit(query))[0]*scale for m in compiled])
            row['previous_query_variance_before']=float(before.var())
            row['previous_query_variance_after']=float(after.var())
            row['previous_query_abs_error_before']=float(abs(before.mean()-y[-1]))
            row['previous_query_abs_error_after']=float(abs(after.mean()-y[-1]))
        np.savez_compressed(out/f'round_{k:03d}_predictions.npz',**arrays)
        if k<args.iterations:
            query,selection=select_query(x,compiled,scale)
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
              f'peak_error={row["peak_distance_mean"]:.3f} e={row["pdf_e_mean"]:.5f}',flush=True)
    print('FINISHED',out,flush=True)

if __name__=='__main__':main()
