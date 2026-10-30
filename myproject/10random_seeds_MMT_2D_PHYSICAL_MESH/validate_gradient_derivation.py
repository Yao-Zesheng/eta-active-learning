"""Validate analytic hard-PDF MSE gradients against autograd and finite differences.
No optimizer steps or training-state mutations are performed.
"""
from pathlib import Path
import os,sys,json
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np,torch
R=Path(__file__).resolve().parent;C=R/'hard_s101';sys.path.insert(0,str(C))
from model import triangle_cdf_torch,ReferenceQuantile,HardPDFNet,locate_numpy,mesh
from problem import law_copula
D=R/'diagnostics/gradient_derivation';D.mkdir(exist_ok=True);torch.set_num_threads(8)

def analytic(scores,tri,areas,B,p,q,targets):
    sorted_ids=np.take_along_axis(tri,np.argsort(scores[tri],axis=1),axis=1)
    a,b,c=scores[sorted_ids].T;t=B@scores
    psi=np.zeros((len(t),len(scores)));f=np.zeros(len(t));ranks=np.zeros(len(t))
    for i,tt in enumerate(t):
        low=(tt>a)&(tt<b);high=(tt>=b)&(tt<c)
        H=np.zeros(len(a));da=np.zeros(len(a));db=da.copy();dc=da.copy();ft=da.copy()
        l=low;d=(b[l]-a[l])*(c[l]-a[l]);F=(tt-a[l])**2/d
        H[l]=F;ft[l]=2*(tt-a[l])/d
        da[l]=-ft[l]+F*(1/(b[l]-a[l])+1/(c[l]-a[l]));db[l]=-F/(b[l]-a[l]);dc[l]=-F/(c[l]-a[l])
        h=high;d=(c[h]-a[h])*(c[h]-b[h]);S=(c[h]-tt)**2/d
        H[h]=1-S;ft[h]=2*(c[h]-tt)/d
        da[h]=-S/(c[h]-a[h]);db[h]=-S/(c[h]-b[h]);dc[h]=-ft[h]+S*(1/(c[h]-a[h])+1/(c[h]-b[h]))
        H[tt>=c]=1
        ranks[i]=H@areas;f[i]=ft@areas
        for j,g in enumerate([da,db,dc]):np.add.at(psi[i],sorted_ids[:,j],areas*g)
    J=f[:,None]*B+psi
    ki=np.clip(np.searchsorted(p,ranks,side='right')-1,0,len(p)-2);Qprime=np.diff(q)[ki]/np.diff(p)[ki];pred=np.interp(ranks,p,q)
    upstream=2/len(t)*(pred-targets)*Qprime
    return {'J':J,'local_J':f[:,None]*B,'global_J':psi,'grad_s':J.T@upstream,'local_grad_s':(f[:,None]*B).T@upstream,'rank':ranks,'pred':pred,'f':f,'Qprime':Qprime}

def matrix(ids,bary,count):
    B=np.zeros((len(ids),count))
    for i in range(len(ids)):np.add.at(B[i],ids[i],bary[i])
    return B

# Float64 score-level check away from branch boundaries.
rng=np.random.RandomState(942);ax,ay,verts,tri,areas=mesh(3,3);pts=rng.uniform(.1,.9,(7,2));ids,bary=locate_numpy(pts,ax,3,ay);B=matrix(ids,bary,len(verts));raw=rng.normal(size=len(verts));p=np.linspace(0,1,31);q=.003+.12*p**1.7;target=rng.uniform(.01,.09,len(pts))
r=torch.tensor(raw,dtype=torch.float64,requires_grad=True);s=(r-r.mean())/r.std(unbiased=False);tt=torch.tensor(B,dtype=torch.float64)@s;rank=triangle_cdf_torch(tt,s[torch.tensor(tri)])@torch.tensor(areas);prediction=ReferenceQuantile(p,q)(rank);loss=((prediction-torch.tensor(target))**2).mean();gs,gr=torch.autograd.grad(loss,(s,r),retain_graph=True)
a=analytic(s.detach().numpy(),tri,areas,B,p,q,target);sigma=float(r.detach().std(unbiased=False));sn=s.detach().numpy();gr_manual=(a['grad_s']-a['grad_s'].mean()-sn*np.mean(a['grad_s']*sn))/sigma
v=rng.normal(size=len(raw));v/=np.linalg.norm(v)
def L(rr):
 z=torch.tensor(rr,dtype=torch.float64);z=(z-z.mean())/z.std(unbiased=False);t=torch.tensor(B)@z;u=triangle_cdf_torch(t,z[torch.tensor(tri)])@torch.tensor(areas);y=ReferenceQuantile(p,q)(u);return float(((y-torch.tensor(target))**2).mean())
eps=1e-6;fd=(L(raw+eps*v)-L(raw-eps*v))/(2*eps);direction=float(gr_manual@v)
report={'float64_score_check':{'MSE':float(loss.detach()),'grad_s_max_abs_error':float(abs(gs.numpy()-a['grad_s']).max()),'grad_raw_max_abs_error':float(abs(gr.numpy()-gr_manual).max()),'finite_difference_directional':fd,'analytic_directional':direction,'directional_abs_error':abs(fd-direction),'relative_error_if_global_CDF_term_omitted':float(np.linalg.norm(a['local_grad_s']-a['grad_s'])/np.linalg.norm(a['grad_s'])),'shift_invariance_sum_grad_s':float(a['grad_s'].sum()),'scale_invariance_dot_grad_s_scores':float(a['grad_s']@sn)}}
assert report['float64_score_check']['grad_s_max_abs_error']<1e-12
assert report['float64_score_check']['grad_raw_max_abs_error']<1e-12
assert abs(fd-direction)<1e-9
# Actual trained physical-mesh network, original float32 weights.
ref=np.load(C/'data/reference_law.npz');O=C/'results/uslw_seed101_physical';ck=torch.load(O/'round_006_member13.pt',map_location='cpu',weights_only=False);net=HardPDFNet(ref['p'],ref['q'],seed=13);net.load_state_dict(ck['state_dict']);state=np.load(O/'round_006_next_state.npz');x,y=state['x'][:9],state['y'][:9];located=net.locate(law_copula(x));s=net.vertex_scores();s.retain_grad();ids,bar=located;sc=(s[ids]*bar).sum(1);u=triangle_cdf_torch(sc,s[net.triangles])@net.areas;yp=net.quantile(u);ell=(yp-torch.tensor(y)).square().mean();params=list(net.net.parameters());actual=torch.autograd.grad(ell,params,retain_graph=True);gs_actual=torch.autograd.grad(ell,s,retain_graph=True)[0];A=analytic(s.detach().numpy(),net.triangles.numpy(),net.areas.numpy(),matrix(ids.numpy(),bar.numpy(),len(s)),ref['p'],ref['q'],y);chain=torch.autograd.grad(s,params,grad_outputs=torch.tensor(A['grad_s']),retain_graph=True);g1=torch.cat([g.ravel() for g in actual]);g2=torch.cat([g.ravel() for g in chain]);report['actual_round6_member1']={'labels':len(y),'parameters':sum(t.numel() for t in params),'MSE':float(ell.detach()),'grad_s_max_abs_error':float(np.max(abs(gs_actual.numpy()-A['grad_s']))),'MLP_parameter_grad_max_abs_error':float((g1-g2).abs().max()),'MLP_parameter_grad_relative_L2_error':float(torch.linalg.vector_norm(g1-g2)/torch.linalg.vector_norm(g1))}
# Gradient of an output at the current global-score maximum vertex.
j=int(s.detach().argmax());locations=net.locate(net.vertices[j:j+1].numpy());peak=net.forward_located(locations);gpeak=torch.autograd.grad(peak.sum(),params);report['current_maximum_vertex']={'vertex':j,'prediction':float(peak.detach()[0]),'output_gradient_parameter_max_abs':float(max(g.abs().max() for g in gpeak))}
(D/'gradient_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
