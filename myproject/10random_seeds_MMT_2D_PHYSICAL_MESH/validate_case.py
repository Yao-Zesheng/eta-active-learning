"""Checks the production case against the previously tested prototype."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parent;CASE=ROOT/'hard_s101';sys.path.insert(0,str(CASE))
from model import HardPDFNet,locate_numpy
from problem import law_copula
from acquisition import select_query
from continuous_pdf import continuous_member_pdfs
from scipy.stats import gaussian_kde
from scipy.special import ndtri,ndtr
OLD=ROOT.with_name('10random_seeds_MMT_2D_QREF_TAIL');D=OLD/'diagnostics/spike_mechanism';torch.set_num_threads(8)
r=np.load(CASE/'data/reference_law.npz');e=np.load(CASE/'data/evaluation.npz');assert len(r['p'])==9506
n=HardPDFNet(r['p'],r['q'],seed=13);ck=torch.load(D/'physical_features_physical32/fixed_labels_member13.pt',map_location='cpu',weights_only=False);n.load_state_dict(ck['state_dict']);c=n.compile();pred=c.predict(law_copula(e['x_mc']));expected=np.load(D/'physical_features_physical32_curves.npz')['mc'];delta=float(abs(pred-expected).max());assert delta<1e-11
u=np.random.RandomState(319).uniform(.0001,.9999,(256,2));ids,bary=n.locate(u)
with torch.no_grad():a,rank=n.forward_located((ids,bary),True)
b,rank2=c.predict(u,True);assert abs(a.numpy()-b).max()<1e-10;assert abs(rank.numpy()-rank2).max()<1e-12
assert abs(float(n.areas.sum())-1)<1e-14;assert float(n.areas.min())>0
# Exercise unequal axis counts, boundary points, vertices and barycentric reconstruction.
z=HardPDFNet(r['p'],r['q'],seed=7,na=16,nv=24);pts=np.r_[u,[[0,0],[1,1],[0,1],[1,0]],z.vertices.numpy()];ii,bb=locate_numpy(pts,z.ax_numpy,z.nv,z.ay_numpy);reconstructed=(z.vertices.numpy()[ii]*bb[:,:,None]).sum(1);assert abs(reconstructed-pts).max()<1e-14;assert bb.min()>-1e-14
state=np.load(OLD/'hard_s101/results/uslw_seed101_refined/round_006_next_state.npz');x=state['x'][:9];query,selection=select_query(x,[c,c],1,acq='uslw');assert selection['candidates'].shape[1]==2 and np.isfinite(selection['scores']).all()
g=np.linspace(0,1,1000);truth=np.maximum(gaussian_kde(e['y_mc'],weights=e['w'])(g),1e-16);pdf,ee,meta=continuous_member_pdfs([c],1,e,g,truth,pred[None]);assert pdf.shape==(1,1000) and np.isfinite(ee).all()
report={'prototype_prediction_max_difference':delta,'torch_compiled_prediction_max_difference':float(abs(a.numpy()-b).max()),'torch_compiled_rank_max_difference':float(abs(rank.numpy()-rank2).max()),'probability_area_sum':float(n.areas.sum()),'minimum_probability_area':float(n.areas.min()),'rectangular_mesh_barycentric_max_difference':float(abs(reconstructed-pts).max()),'reference_knots':len(r['p']),'continuous_pdf_error_prototype_checkpoint':ee,'continuous_pdf_probability_mass':meta[0]['probability_mass'],'original_solver_and_acquisition_identical':{f:hashlib.sha256((CASE/f).read_bytes()).hexdigest()==hashlib.sha256((OLD/'hard_s101'/f).read_bytes()).hexdigest() for f in ['acquisition.py','solver/evaluate_mmt.m']}}
(ROOT/'diagnostics/validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
