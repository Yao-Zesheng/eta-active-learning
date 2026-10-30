"""Independent lambda=0 USLW run; only this experiment directory is writable."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from mlp_repro import MLPEnsemble
from problem import ROOT, STD, from_unit, oracle, to_unit
from acquisition import select_query

HARD = ROOT.parent/'TOY_2D_HARDPDF/results/uslw_seed101'
LAYERS=[2,200,200,200,200,200,1]

def pdf(values, centers, bandwidth):
    """Same linear deposition/10 sigma kernel, extended for unbounded outputs."""
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all()
    dx=.8/4096
    z=(values-centers[0])/dx
    lo=np.floor(z).astype(np.int64); w=z-lo
    left=min(0,int(lo.min()))-2
    right=max(len(centers)-1,int(lo.max())+1)+2
    mass=(np.bincount(lo-left,weights=1-w,minlength=right-left+1)+
          np.bincount(lo+1-left,weights=w,minlength=right-left+1))
    smooth=gaussian_filter1d(mass/(len(values)*dx),bandwidth/dx,
                            mode='constant',truncate=10.)
    return smooth[-left:len(centers)-left]

def metrics(pred,truth,threshold):
    error=pred-truth; tail=truth>=threshold; predicted=pred>=threshold
    overlap=np.sum(tail & predicted)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
        relative_l2=float(np.linalg.norm(error)/np.linalg.norm(truth)),
        r2=float(1-np.mean(error**2)/np.var(truth)),
        mae=float(np.mean(abs(error))),max_abs=float(abs(error).max()),
        top1_rmse=float(np.sqrt(np.mean(error[tail]**2))),
        top1_recall=float(overlap/tail.sum()),
        top1_precision=float(overlap/max(1,predicted.sum())))

class Member:
    def __init__(self,net): self.net=net
    def predict(self,u):
        dev=next(self.net.parameters()).device
        chunks=[]
        with torch.no_grad():
            for batch in np.array_split(u,max(1,int(np.ceil(len(u)/4096)))):
                chunks.append(self.net(torch.as_tensor(2*batch-1,dtype=torch.float32,device=dev)).cpu().numpy().ravel())
        return np.concatenate(chunks).astype(float)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--name',default='uslw_seed101')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--lambd',type=float,default=0.,help='W1 (quantile) penalty weight; 0 = pure MSE')
    parser.add_argument('--lhs-seed',type=int,default=101,help='RandomState seed of the 3-point LHS initial design')
    parser.add_argument('--seed-base',type=int,default=1,help='member seed base: round k uses [base+2k, base+2k+1]')
    parser.add_argument('--device',default='cpu',help='cpu (lambda=0 default) or cuda for W1 runs')
    parser.add_argument('--threads',type=int,default=1,help='torch intra-op threads (W1 aux forwards benefit from >1)')
    parser.add_argument('--acq',default='uslw',choices=['us','uslw'],help='acquisition: us = variance only, uslw = variance*pX/KDE(mean)')
    parser.add_argument('--x-aux',type=int,default=200000,help='auxiliary input size for the W1 term')
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    out=ROOT/'results'/args.name;out.mkdir(parents=True,exist_ok=True)
    config=dict(iterations=args.iterations,epochs=args.epochs,lambd=args.lambd,layers=LAYERS,
        learning_rate=.001,optimizer='Adam eps=1e-8, final weights, no best selection',
        seeds=f'{args.seed_base}+2k,{args.seed_base}+2k+1; cold restart, k=number of acquired labels',
        initial_n=3,lhs_seed=args.lhs_seed,candidates=1000,input_law='N(0,10 I)',
        input_transform='2*Phi(x/sqrt(10))-1',output_transform='y/reference_max',
        device=args.device,threads=1,dtype='float32',
        selection=f'copied candidate argmax, acq={args.acq}: variance or variance*p_X/KDE(mean), physical Gaussian p_X',
        endpoint='100 queries then refit both members on all 103 observations',
        pdf_metric='shared fixed bandwidth; linear deposition, 10 sigma, log10 floor 1e-16; no clipping')
    if args.lambd>0:
        config.update(w1=dict(aux_points=args.x_aux,aux_seed=20261020,omega=50,
            reference='data/reference_law.npz quantile knots (p, q/scale)',
            loss='MSE + lambd * mean|model quantiles - reference quantiles|'))
    if (out/'config.json').exists():
        saved=json.loads((out/'config.json').read_text())
        # backward compatible: every key of a previous config must be unchanged
        assert args.resume and all(config.get(key)==value for key,value in saved.items())
    else:
        (out/'config.json').write_text(json.dumps(config,indent=2))
    ref=np.load(ROOT/'data/reference_law.npz'); ev=np.load(ROOT/'data/evaluation.npz')
    scale=float(ref['scale']); threshold=float(np.interp(.99,ref['p'],ref['q']))
    threshold001=float(np.interp(.999,ref['p'],ref['q']))
    edges=np.linspace(-.1,.7,4097);centers=(edges[1:]+edges[:-1])/2
    bandwidth=ev['y_mc'].std(ddof=1)*len(ev['y_mc'])**(-.2)
    truthpdf=pdf(ev['y_mc'],centers,bandwidth)
    mask=(centers>=0)&(centers<=scale)
    # Check compatibility against the saved hard-model metric BEFORE any fitting.
    # (Only when the original sibling project exists; skipped in copied cases/HPC.)
    if (HARD/'round_000_predictions.npz').exists():
        hard0=np.load(HARD/'round_000_predictions.npz')
        check=pdf(hard0['mc'].mean(0),centers,bandwidth)
        assert np.max(abs(check-hard0['pdf_predictions'][2]))<1e-8
    rng_init=np.random.RandomState(args.lhs_seed)
    u_init=np.column_stack([(rng_init.permutation(3)+rng_init.rand(3))/3 for _ in range(2)])
    x=from_unit(u_init);y=oracle(x);rows=[]
    # W1 ingredients for lambd>0: auxiliary inputs from N(0,10I) in feature
    # coordinates, and the reference quantile law (p, q/scale) in z-space.
    x_aux=ytail_ref=quantiles=None
    if args.lambd>0:
        rng_aux=np.random.default_rng(20261020)
        x_aux=2*to_unit(rng_aux.normal(0,STD,(args.x_aux,2)))-1
        quantiles=torch.from_numpy(ref['p'].astype(np.float32))
        ytail_ref=torch.from_numpy((ref['q']/scale).astype(np.float32))
    if args.resume and (out/'history.json').exists():
        rows=json.loads((out/'history.json').read_text())
        state=np.load(out/f'round_{len(rows)-1:03d}_next_state.npz');x,y=state['x'],state['y']
    np.savez_compressed(out/'initial_lhs.npz',x=x,y=y,u=to_unit(x))
    start=time.perf_counter()
    for k in range(len(rows),args.iterations+1):
        began=time.perf_counter()
        ensemble=MLPEnsemble(2*to_unit(x)-1,y,LAYERS,.001,args.epochs,2,
            lambda a:a/scale,lambda a:a*scale,[args.seed_base+2*k,args.seed_base+2*k+1],device=device,lambd=args.lambd,
            X_aux=x_aux,Ytail_ref=ytail_ref,quantiles=quantiles,omega=50)
        members=[Member(net) for net in ensemble.model]
        mc=np.array([m.predict(to_unit(ev['x_mc']))*scale for m in members])
        grid=np.array([m.predict(to_unit(ev['x_grid']))*scale for m in members])
        train=np.array([m.predict(to_unit(x))*scale for m in members])
        row=metrics(mc.mean(0),ev['y_mc'],threshold)
        row.update({'grid_'+key:value for key,value in metrics(grid.mean(0),ev['y_grid'],threshold).items()})
        row.update(iteration=k,n=len(x),train_mse_members=np.mean((train-y)**2,axis=1).tolist(),
            train_mse_mean=float(np.mean((train.mean(0)-y)**2)),
            disagreement_rms=float(np.sqrt(np.mean((mc[0]-mc[1])**2))),
            top001_rmse=float(np.sqrt(np.mean((mc.mean(0)[ev['y_mc']>=threshold001]-ev['y_mc'][ev['y_mc']>=threshold001])**2))),
            negative_fraction=float(np.mean(mc.mean(0)<0)),prediction_min=float(mc.mean(0).min()),
            prediction_max=float(mc.mean(0).max()))
        densities=np.array([pdf(a,centers,bandwidth) for a in [*mc,mc.mean(0)]])
        e=np.sum(abs(np.log10(np.maximum(densities[:,mask],1e-16))-
            np.log10(np.maximum(truthpdf[mask],1e-16))),axis=1)*(.8/4096)/scale
        row.update(pdf_e_mean=float(e[2]),pdf_e_members=e[:2].tolist())
        truepeak=ev['x_grid'][ev['y_grid'].argmax()]
        predpeak=ev['x_grid'][grid.mean(0).argmax()]
        row.update(peak_distance_mean=float(np.linalg.norm(predpeak-truepeak)),predicted_peak=predpeak.tolist())
        if k:
            prev=np.load(out/f'round_{k-1:03d}_selection.npz')
            before=prev['members'][:,int(prev['selected_index'])]
            row.update(previous_query_variance_before=float(before.var()),
                       previous_query_variance_after=float(train[:,-1].var()))
        torch.save(dict(states=[m.state_dict() for m in ensemble.model],x=x,y=y,
            scale=scale,layers=LAYERS,seeds=[args.seed_base+2*k,args.seed_base+2*k+1],losses=ensemble.loss_history),out/f'round_{k:03d}.pt')
        np.savez_compressed(out/f'round_{k:03d}_predictions.npz',mc=mc,grid=grid,
            train_predictions=train,pdf_grid=centers,pdf_true=truthpdf,pdf_predictions=densities,bandwidth=bandwidth)
        if k<args.iterations:
            query,selection=select_query(x,members,scale,acq=args.acq)
            observed=oracle(query)
            np.savez_compressed(out/f'round_{k:03d}_selection.npz',**selection,query=query,observed=observed)
            row.update(query=query[0].tolist(),observed_y=float(observed[0]),
                       selected_index=selection['selected_index'],acquisition_max=float(selection['scores'].max()))
            x=np.vstack([x,query]);y=np.r_[y,observed]
        row['seconds']=time.perf_counter()-began
        np.savez_compressed(out/f'round_{k:03d}_next_state.npz',x=x,y=y)
        rows.append(row)
        temporary=out/'history.tmp';temporary.write_text(json.dumps(rows,indent=2))
        temporary.replace(out/'history.json')
        np.savez_compressed(out/'observations.npz',x=x,y=y)
        print(f'queries={k} n={row["n"]} RMSE={row["rmse"]:.6f} e={row["pdf_e_mean"]:.6f} top1={row["top1_rmse"]:.6f} seconds={row["seconds"]:.1f}',flush=True)
    print(f'FINISHED in {time.perf_counter()-start:.1f}s: {out}',flush=True)

if __name__=='__main__':main()
