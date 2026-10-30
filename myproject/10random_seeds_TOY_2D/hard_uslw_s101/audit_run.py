"""Validate independent acquisition and final 5x200 checkpoints without edits."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from model import HardPDFNet
from problem import lhs,oracle,to_unit,from_unit,input_pdf
from acquisition import kde
ROOT=Path(__file__).resolve().parent
out=ROOT/'results/uslw_seed101'
torch.set_num_threads(8)
h=json.loads((out/'history.json').read_text());assert len(h)==101
ref=np.load(ROOT/'data/reference_law.npz');ev=np.load(ROOT/'data/evaluation.npz');scale=float(ref['scale'])
obs=np.load(out/'observations.npz');assert obs['x'].shape==(103,2)
assert np.array_equal(obs['x'][:3],lhs())
assert np.max(abs(oracle(obs['x'])-obs['y']))<1e-12
base=ROOT.parent/'TOY_2D_ETA_LAMBDA0_USLW/results/uslw_seed101'
assert np.array_equal(obs['x'][:3],np.load(base/'initial_lhs.npz')['x'])
max_score_error=0.
for k,row in enumerate(h):
    assert row['iteration']==k and row['n']==k+3
    assert all(log['seed']==2*k+j+1 and log['n']==k+3 for j,log in enumerate(row['training']))
    if k==100:continue
    s=np.load(out/f'round_{k:03d}_selection.npz');idx=int(s['selected_index'])
    assert idx==int(s['scores'].argmax())
    assert np.array_equal(s['query'][0],s['candidates'][idx])
    assert np.array_equal(s['query'][0],obs['x'][k+3])
    candidates=from_unit(np.random.RandomState(k+3).rand(1000,2))
    assert np.array_equal(candidates,s['candidates'])
    assert np.max(abs(oracle(s['query'])-s['observed']))<1e-12
    if k in [0,49,99]:
        mu=s['members'].mean(0);pad=1e-6*np.ptp(mu)+1e-12
        grid=np.linspace(mu.min()-pad,mu.max()+pad,10000)
        scores=s['members'].var(0)*input_pdf(candidates)/np.maximum(np.interp(mu,grid,kde(mu,grid)),1e-16)
        max_score_error=max(max_score_error,float(np.max(abs(scores-s['scores']))))
assert max_score_error<1e-12
saved=np.load(out/'round_100_predictions.npz');errors=[];direct_errors=[];counts=[];spans=[]
for j,seed in enumerate([201,202]):
    checkpoint=torch.load(out/f'round_100_member{seed}.pt',map_location='cpu',weights_only=False)
    assert checkpoint['layers']==[2]+[200]*5+[1] and checkpoint['activation']=='ReLU'
    m=HardPDFNet(ref['p'],ref['q']/scale,seed=seed);m.load_state_dict(checkpoint['state_dict'])
    counts.append(sum(p.numel() for p in m.net.parameters()));assert counts[-1]==161601
    c=m.compile();u=to_unit(ev['x_mc'])
    errors.append(float(np.max(abs(c.predict(u)*scale-saved['mc'][j]))))
    with torch.no_grad():direct=m(u[:400]).numpy()*scale
    direct_errors.append(float(np.max(abs(direct-saved['mc'][j,:400]))))
    spans.append(c.min_triangle_span)
assert max(errors)<1e-10 and max(direct_errors)<1e-8 and min(spans)>0
sources=json.loads((ROOT/'source_sha256.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in sources.items())
report=dict(completed_queries=100,final_observations=103,initial_design_matches_baseline=True,
    layers=[2]+[200]*5+[1],activation='ReLU',parameters_per_member=counts,
    candidate_argmax_checks=100,max_recomputed_acquisition_error=max_score_error,
    final_checkpoint_prediction_max_errors=errors,analytic_vs_compiled_max_errors=direct_errors,
    final_min_triangle_spans=spans,original_sources_unchanged=True,
    rejected_trials=sum(len(log['rejected_invalid_steps']) for row in h for log in row['training']),
    total_training_seconds=sum(log['elapsed_seconds'] for row in h for log in row['training']))
(ROOT/'run_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
(ROOT/'code_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
