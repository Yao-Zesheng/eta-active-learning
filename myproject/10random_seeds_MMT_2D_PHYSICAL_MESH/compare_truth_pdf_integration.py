"""Truth-grid PDF audit: spatial interpolation/integration versus weighted KDE.
No learner predictions or Q_ref are used. Both interpolation coordinate choices
are checked. 'Truth' here means an interpolant of the 100x100 solver data, not
an independently solved continuum response.
"""
from pathlib import Path
import sys,json,csv,time
import numpy as np
from scipy.special import ndtr
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;CASE=ROOT/'hard_s101';sys.path.insert(0,str(CASE))
from model import triangle_cdf_numpy
E=np.load(CASE/'data/evaluation.npz');X,Y,W=E['x_mc'],E['y_mc'],E['w'];ax=np.unique(X[:,0]);ay=np.unique(X[:,1]);N,M=len(ax),len(ay)
xx,yy=np.meshgrid(ax,ay,indexing='ij');assert np.array_equal(X,np.column_stack((xx.ravel(),yy.ravel())))
i,j=np.meshgrid(np.arange(N-1),np.arange(M-1),indexing='ij');v=(i*M+j).ravel();tri=np.concatenate([np.column_stack((v,v+M,v+M+1)),np.column_stack((v,v+M+1,v+1))]);abc=Y[tri];assert np.all(np.ptp(abc,axis=1)>0)
plo,phi=ndtr(-6),ndtr(6);pa=(ndtr(ax)-plo)/(phi-plo);pb=(ndtr(ay)-plo)/(phi-plo);areas=.5*(np.diff(pa)[:,None]*np.diff(pb)[None,:]).ravel();areas=np.r_[areas,areas];assert abs(areas.sum()-1)<1e-14
D=ROOT/'diagnostics/truth_pdf_integration_vs_kde';D.mkdir(exist_ok=True)
kde=gaussian_kde(Y,weights=W/W.sum());h=float(np.sqrt(kde.covariance[0,0]));thresholds=np.array([0.,.005,.02,.04,.06,.08,.10,.11,float(Y.max())]);report={'source':'100x100 physical-uniform MATLAB truth grid','nodes':len(Y),'physical_spacing':float(ax[1]-ax[0]),'input_law':'box-conditioned N(0,I), consistent with normalized grid weights','triangle_count':len(tri),'constant_triangles':0,'kde_bandwidth':h,'kde_effective_sample_size':float(kde.neff),'grid_values_min_max':[float(Y.min()),float(Y.max())],'kde_probability_below_zero':float(kde.integrate_box_1d(-np.inf,0)),'kde_probability_beyond_grid_max':float(kde.integrate_box_1d(Y.max(),np.inf)),'convergence':[]}
# Common extended output axis; padding retains Gaussian mass outside support.
step=float(Y.max()/16384);pad=int(np.ceil(12*h/step));centers=(np.arange(-pad,16384+pad)+.5)*step;density_kde=kde(centers)
# Probability-coordinate piecewise-affine interpolant: exact triangle CDF/SF.
def prob_integral(bins):
 edges=np.linspace(0,Y.max(),bins+1);cdf=triangle_cdf_numpy(edges,abc,areas);sf=triangle_cdf_numpy(-edges,-abc,areas);mass=np.where(cdf[:-1]<.5,np.diff(cdf),-np.diff(sf));assert mass.min()>-1e-12;mass=np.maximum(mass,0);dy=edges[1]-edges[0];padding=int(np.ceil(12*h/dy));z=(np.arange(-padding,bins+padding)+.5)*dy;pdf=gaussian_filter1d(np.pad(mass,(padding,padding))/dy,h/dy,mode='constant',truncate=12.);return z,pdf,mass
for bins in [8192,16384]:
 z,pdf,mass=prob_integral(bins);report['convergence'].append({'method':'probability-coordinate affine triangles','output_bins':bins,'total_mass':float(mass.sum())});print('probability integral complete',bins,flush=True)
 if bins==16384:prob_pdf=pdf;assert np.array_equal(z,centers)
 else:prob_coarse=(z,pdf)
# Physical-coordinate affine interpolant with true Gaussian density integrated
# in each physical triangle using product Gauss-Legendre/Duffy quadrature.
def physical_integral(order):
 points,weights=leggauss(order);s=(points+1)/2;wg=weights/2;u,z=np.meshgrid(s,s,indexing='ij');gw=(wg[:,None]*wg[None,:]*(1-u)).ravel();u=u.ravel();z=z.ravel();bary=np.column_stack(((1-u)*(1-z),u,(1-u)*z));mass=np.zeros(len(centers));prob=0.;mean=0.;second=0.
 for start in range(0,len(tri),128):
  ids=tri[start:start+128];verts=X[ids];values=Y[ids];xy=np.einsum('qv,tvd->tqd',bary,verts);output=np.einsum('qv,tv->tq',bary,values);dx=verts[:,1]-verts[:,0];dz=verts[:,2]-verts[:,0];jac=np.abs(dx[:,0]*dz[:,1]-dx[:,1]*dz[:,0]);density=np.exp(-.5*(xy*xy).sum(2))/(2*np.pi*(phi-plo)**2);ww=jac[:,None]*gw[None,:]*density;prob+=float(ww.sum());mean+=float((ww*output).sum());second+=float((ww*output**2).sum())
  location=(output-centers[0])/step;left=np.floor(location).astype(int);fraction=location-left;assert left.min()>=0 and left.max()+1<len(mass)
  mass+=np.bincount(left.ravel(),weights=(ww*(1-fraction)).ravel(),minlength=len(mass));mass+=np.bincount((left+1).ravel(),weights=(ww*fraction).ravel(),minlength=len(mass))
 pdf=gaussian_filter1d(mass/step,h/step,mode='constant',truncate=12.)
 return pdf,mass,{'method':'physical-coordinate affine triangles, Gaussian-weighted integration','Gauss_order':order,'total_mass':prob,'mean':mean,'variance':second-mean**2}
phys_curves={}
for order in [6,12,24]:
 pdf,mass,meta=physical_integral(order);phys_curves[order]=pdf;report['convergence'].append(meta);print('physical integral complete',order,flush=True)
phys_pdf=phys_curves[24]
# Exact unsmoothed PDF of the probability-coordinate interpolant. Positive terms
# avoid subtracting nearly equal CDFs in the extreme tail.
plotx=np.linspace(0,.14,2501);raw=np.zeros(len(plotx));a,b,c=np.sort(abc,axis=1).T;dl=(b-a)*(c-a);dr=(c-a)*(c-b);dl=np.where(dl==0,1.,dl);dr=np.where(dr==0,1.,dr)
for st in range(0,len(plotx),64):
 q=plotx[st:st+64,None];val=np.where((q>a)&(q<b),2*(q-a)/dl,0.)+np.where((q>=b)&(q<c),2*(c-q)/dr,0.);raw[st:st+64]=(val*areas).sum(1)
# Same e grid/window as historical runs, plus full-window L1 and support diagnostics.
yint=np.linspace(0,1,1000);truth_kde=np.maximum(kde(yint),1e-16)
def metrics(pdf):
 den=np.maximum(np.interp(yint,centers,pdf,left=0,right=0),1e-16);de=np.abs(np.log10(den[:-1])-np.log10(truth_kde[:-1]))/999
 regions=[(0,.005),(.005,.02),(.02,.06),(.06,.10),(.10,float(Y.max())),(float(Y.max()),1)]
 return {'e_vs_original_KDE_0_to_1':float(de.sum()),'L1_full_extended_window':float(np.trapz(abs(pdf-density_kde),centers)),'negative_probability':float(np.sum(pdf[centers<0])*step),'e_components':[{'y_interval':[lo,hi],'contribution':float(de[(yint[:-1]>=lo)&(yint[:-1]<hi)].sum())} for lo,hi in regions]}
report['probability_interpolant_smoothed']=metrics(prob_pdf);report['physical_interpolant_smoothed']=metrics(phys_pdf)
report['convergence_checks']={'physical_order_12_vs_24_L1':float(np.trapz(abs(phys_curves[12]-phys_pdf),centers)),'physical_order_6_vs_24_L1':float(np.trapz(abs(phys_curves[6]-phys_pdf),centers)),'probability_output_bins_8192_vs_16384_L1':float(np.trapz(abs(np.interp(centers,*prob_coarse,left=0,right=0)-prob_pdf),centers)),'two_interpolation_coordinates_smoothed_L1':float(np.trapz(abs(phys_pdf-prob_pdf),centers))}
report['tail_probabilities']=[]
sf=triangle_cdf_numpy(-thresholds,-abc,areas)
for t,ss in zip(thresholds,sf):
 report['tail_probabilities'].append({'threshold':float(t),'original_grid_point_mass':float(W[Y>t].sum()),'probability_interpolant_unsmoothed':float(ss),'grid_KDE':float(np.sum(W*ndtr((Y-t)/h))),'physical_interpolant_smoothed':float(np.sum(phys_pdf[centers>t])*step),'probability_interpolant_smoothed':float(np.sum(prob_pdf[centers>t])*step)})
report['means']={'original_weighted_grid':float(W@Y),'probability_interpolant':float(np.sum(areas*abc.mean(1))),'physical_interpolant':report['convergence'][-1]['mean']}
np.savez_compressed(D/'truth_pdf_curves.npz',y=centers,kde=density_kde,physical_continuous_smoothed=phys_pdf,probability_continuous_smoothed=prob_pdf,raw_y=plotx,probability_continuous_unsmoothed=raw)
with (D/'truth_pdf_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.writer(f);writer.writerow(['y','original_weighted_KDE','physical_interpolation_integral_same_Gaussian_smoothing','probability_interpolation_integral_same_Gaussian_smoothing']);writer.writerows(zip(centers,density_kde,phys_pdf,prob_pdf))
(D/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
# Figures distinguish raw laws, smoothing, and interpolation coordinates.
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(13.5,9));fig.subplots_adjust(top=.88,bottom=.10,hspace=.34,wspace=.25)
for ax in axs[0]:
 ax.plot(centers,density_kde,color='#e67e22',label='原10000真值：加权 KDE',lw=2)
 ax.plot(centers,phys_pdf,color='#0072B2',label='物理空间插值＋连续积分＋同带宽平滑',lw=1.9)
 ax.plot(centers,prob_pdf,color='#009E73',ls='--',label='概率空间插值＋连续积分＋同带宽平滑',lw=1.7)
 ax.set(xlabel='输出 y',ylabel='概率密度');ax.grid(alpha=.2)
axs[0,0].set(xlim=(-.008,.065),title='主体：统一高斯带宽 h=0.002725');axs[0,0].axvline(0,color='gray',ls=':');axs[0,0].legend(fontsize=8.3)
axs[0,1].set(xlim=(.06,.14),ylim=(1e-17,.1),yscale='log',title='极低概率尾部：对数坐标');axs[0,1].axvline(Y.max(),color='gray',ls=':',label='原真值最大值');axs[0,1].axhline(1e-16,color='gray',ls=':',lw=.8)
axs[1,0].plot(plotx,raw,color='#777777',lw=1.1,label='连续积分原始 PDF（概率空间三角插值）');axs[1,0].plot(centers,prob_pdf,color='#009E73',lw=2,label='该分布经过高斯平滑');axs[1,0].plot(centers,density_kde,color='#e67e22',lw=1.8,ls='--',label='原加权 KDE');axs[1,0].set(xlim=(-.004,.04),ylim=(0,150),xlabel='输出 y',ylabel='概率密度',title='原始密度与平滑密度不是同一个对象');axs[1,0].legend(fontsize=8.3);axs[1,0].grid(alpha=.2)
rel=(phys_pdf-density_kde)/np.maximum(density_kde,1e-16);sel=(centers>=0)&(centers<=.12)&(density_kde>1e-14);axs[1,1].plot(centers[sel],100*rel[sel],color='#0072B2',label='物理插值积分相对 KDE 的差异');axs[1,1].axhline(0,color='gray',lw=.8);axs[1,1].set(xlabel='输出 y',ylabel='相对差异（%）',title='同带宽下的剩余差异：空间积分与点权重近似');axs[1,1].grid(alpha=.2)
fig.suptitle('MMT真实数据：100×100网格的连续积分 PDF 与原加权 KDE',fontsize=16,y=.97);fig.text(.5,.925,'所有曲线均来自真实求解值，不使用神经网络。连续积分依赖插值假设；没有新增MMT求解。',ha='center',fontsize=10);fig.text(.5,.035,'上排统一高斯平滑带宽，避免把平滑差异误认为积分差异。未平滑曲线在左下图单独展示。',ha='center',fontsize=10)
fig.savefig(D/'truth_pdf_comparison.png',dpi=190);fig.savefig(D/'truth_pdf_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
for ax in axs:
 ax.plot(centers,density_kde,color='#e67e22',lw=2,label='原真实值加权 KDE');ax.plot(centers,phys_pdf,color='#0072B2',lw=2,label='物理插值连续积分，同带宽平滑');ax.set(xlabel='输出 y',ylabel='概率密度');ax.grid(alpha=.2)
axs[0].set(xlim=(-.008,.065),title='主体与非负边界');axs[0].axvline(0,color='gray',ls=':');axs[0].legend(fontsize=9)
axs[1].set(xlim=(.06,.14),ylim=(1e-17,.1),yscale='log',title='上尾与支持域外的平滑延伸');axs[1].axvline(Y.max(),color='gray',ls=':',label='网格真值最大值');axs[1].legend(fontsize=9)
fig.savefig(D/'truth_pdf_main.png',dpi=190);fig.savefig(D/'truth_pdf_main.pdf');plt.close(fig)
