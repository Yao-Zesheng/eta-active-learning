"""Audit the moving-CDF parameter gradient: scalar cancellation and real checkpoints.
Detach is used only to isolate the two paths in this test, never for training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path(__file__).resolve().parent;C=R/'hard_s101';sys.path.insert(0,str(C))
from model import HardPDFNet,triangle_cdf_torch,ReferenceQuantile,mesh,locate_numpy
from problem import law_copula
D=R/'diagnostics/moving_cdf_gradient_audit';D.mkdir(exist_ok=True);torch.set_num_threads(8)
report={'scalar_shift_test':[],'checkpoints':[]}
# Translation is added AFTER score normalization, so this tests the CDF itself.
rng=np.random.RandomState(130);ax,ay,verts,tri,area=mesh(3,3);base=torch.tensor(rng.normal(size=len(verts)),dtype=torch.float64);coords=rng.uniform(.15,.85,(4,2));ids,bary=locate_numpy(coords,ax,3,ay);ids=torch.tensor(ids);bary=torch.tensor(bary);tri=torch.tensor(tri);area=torch.tensor(area)
for i in range(len(coords)):
 theta=torch.tensor(.37,dtype=torch.float64,requires_grad=True);scores=base+theta;t=(scores[ids[i]]*bary[i]).sum()[None];abc=scores[tri]
 full=(triangle_cdf_torch(t,abc)@area)[0];local=(triangle_cdf_torch(t,abc.detach())@area)[0];cdf=(triangle_cdf_torch(t.detach(),abc)@area)[0]
 gf=float(torch.autograd.grad(full,theta,retain_graph=True)[0]);gl=float(torch.autograd.grad(local,theta,retain_graph=True)[0]);gc=float(torch.autograd.grad(cdf,theta)[0]);report['scalar_shift_test'].append({'query':i,'rank':float(full.detach()),'full_derivative':gf,'argument_path_only':gl,'CDF_parameter_path_only':gc,'sum_two_paths':gl+gc});assert abs(gf)<1e-12 and abs(gl+gc)<1e-12 and gl>0
O=C/'results/uslw_seed101_physical';ref=np.load(C/'data/reference_law.npz')
for k,m in [(6,1),(6,2),(100,1),(100,2)]:
 seed=2*k+m;n=HardPDFNet(ref['p'],ref['q'],seed=seed);n.load_state_dict(torch.load(O/f'round_{k:03d}_member{seed}.pt',map_location='cpu',weights_only=False)['state_dict']);st=np.load(O/f'round_{k:03d}_next_state.npz');count=3+k;x,y=st['x'][:count],st['y'][:count];ids,ba=n.locate(law_copula(x));s=n.vertex_scores();t=(s[ids]*ba).sum(1);abc=s[n.triangles];params=list(n.net.parameters());target=torch.tensor(y)
 def grads(targ,table):
  u=triangle_cdf_torch(targ,table)@n.areas;pred=n.quantile(u);L=(pred-target).square().mean();g=torch.autograd.grad(L,params,retain_graph=True);return float(L.detach()),torch.cat([v.flatten().double() for v in g])
 loss,gfull=grads(t,abc);_,glocal=grads(t,abc.detach());_,gcdf=grads(t.detach(),abc);norm=float(gfull.norm());res=float((gfull-glocal-gcdf).norm());item={'round':k,'member':m,'training_points':count,'MSE':loss,'full_gradient_L2':norm,'argument_path_L2':float(glocal.norm()),'CDF_parameter_path_L2':float(gcdf.norm()),'relative_sum_residual':res/norm,'relative_error_if_CDF_path_omitted':float((glocal-gfull).norm())/norm};report['checkpoints'].append(item);assert item['relative_sum_residual']<1e-4;print(json.dumps(item),flush=True)
# Parameter-space finite differences: actual checkpoint evaluated in float64.
n=HardPDFNet(ref['p'],ref['q'],seed=13);n.load_state_dict(torch.load(O/'round_006_member13.pt',map_location='cpu',weights_only=False)['state_dict']);n=n.double();st=np.load(O/'round_006_next_state.npz');located=n.locate(law_copula(st['x'][:9]));target=torch.tensor(st['y'][:9]);params=list(n.net.parameters());L=(n.forward_located(located)-target).square().mean();g=torch.autograd.grad(L,params);gnorm=torch.sqrt(sum(a.square().sum() for a in g));direction=[a/gnorm for a in g];original=[a.detach().clone() for a in params];tests=[]
for eps in [1e-6,1e-7,1e-8]:
 vals=[]
 for sign in [1,-1]:
  with torch.no_grad():
   for par,b,d in zip(params,original,direction):par.copy_(b+sign*eps*d)
   vals.append(float((n.forward_located(located)-target).square().mean()))
 fd=(vals[0]-vals[1])/(2*eps);tests.append({'epsilon':eps,'finite_difference':fd,'autograd_directional':float(gnorm),'relative_error':abs(fd-float(gnorm))/float(gnorm)})
with torch.no_grad():
 for par,b in zip(params,original):par.copy_(b)
report['float64_actual_network_parameter_finite_differences']=tests
report['training_source_matches_run_snapshots']={f:hashlib.sha256((C/f).read_bytes()).digest()==hashlib.sha256((O/(Path(f).stem+'_used.py')).read_bytes()).digest() for f in ['model.py','training.py','run.py']}
(D/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
