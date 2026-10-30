"""Exact unsmoothed PDF of physical-affine truth triangles under Gaussian input.
Coarea formula: integrate the Gaussian density along each output level segment.
This avoids histograms/quadrature when obtaining the raw continuous truth PDF.
"""
from pathlib import Path
import numpy as np,json
from scipy.special import ndtr
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;D=R/'diagnostics/truth_pdf_integration_vs_kde';E=np.load(R/'hard_s101/data/evaluation.npz');X,Y,W=E['x_mc'],E['y_mc'],E['w'];z=np.load(D/'truth_pdf_curves.npz');analysis=json.loads((D/'analysis.json').read_text());N=len(np.unique(X[:,0]));M=len(np.unique(X[:,1]));i,j=np.meshgrid(np.arange(N-1),np.arange(M-1),indexing='ij');v=(i*M+j).ravel();tri=np.concatenate([np.column_stack((v,v+M,v+M+1)),np.column_stack((v,v+M+1,v+1))]);vals=Y[tri];order=np.argsort(vals,axis=1);yy=np.take_along_axis(vals,order,axis=1);verts=np.take_along_axis(X[tri],order[:,:,None],axis=1);a,b,c=yy.T;A,B,C=verts[:,0],verts[:,1],verts[:,2];e1,e2=B-A,C-A;dy1,dy2=b-a,c-a;det=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0];gx=(dy1*e2[:,1]-dy2*e1[:,1])/det;gy=(e1[:,0]*dy2-e2[:,0]*dy1)/det;norm=np.hypot(gx,gy);assert np.all(norm>0);normal=np.column_stack((gx,gy))/norm[:,None];tangent=np.column_stack((-gy,gx))/norm[:,None];normalA=np.sum(A*normal,axis=1);prob_norm=(ndtr(6)-ndtr(-6))**2

def raw_pdf(t):
 result=np.zeros(len(t))
 for start in range(0,len(t),48):
  ts=t[start:start+48];ii,jj=np.where((ts[:,None]>a[None,:])&(ts[:,None]<c[None,:]));tt=ts[ii];alpha=(tt-a[jj])/(c[jj]-a[jj]);P=A[jj]+alpha[:,None]*(C[jj]-A[jj]);low=tt<b[jj];base=np.where(low[:,None],A[jj],B[jj]);end=np.where(low[:,None],B[jj],C[jj]);ya=np.where(low,a[jj],b[jj]);yc=np.where(low,b[jj],c[jj]);beta=(tt-ya)/(yc-ya);Q=base+beta[:,None]*(end-base);l1=np.sum(P*tangent[jj],axis=1);l2=np.sum(Q*tangent[jj],axis=1);lmin,lmax=np.minimum(l1,l2),np.maximum(l1,l2);line_probability=np.where(lmin>=0,ndtr(-lmin)-ndtr(-lmax),ndtr(lmax)-ndtr(lmin));d=normalA[jj]+(tt-a[jj])/norm[jj];contribution=np.exp(-.5*d*d)/np.sqrt(2*np.pi)/norm[jj]*line_probability/prob_norm;result[start:start+len(ts)]=np.bincount(ii,weights=contribution,minlength=len(ts))
 return result
plotx=np.unique(np.r_[np.linspace(0,.003,3001),np.linspace(.003,.065,4001),np.linspace(.065,.14,2501),Y.min(),Y.max()]);raw=raw_pdf(plotx);raw_coarse_mass=float(np.trapz(raw,plotx));np.savez_compressed(D/'physical_continuous_unsmoothed.npz',y=plotx,pdf=raw);(D/'coarea_validation.json').write_text(json.dumps({'method':'exact Gaussian line integrals over physical affine triangles','plot_grid_trapezoidal_mass':raw_coarse_mass,'not_normalized_or_rescaled':True,'triangles':len(tri),'raw_pdf_support':[float(Y.min()),float(Y.max())]},indent=2),encoding='utf8')
k=gaussian_kde(Y,weights=W);h=analysis['kde_bandwidth'];kline=k(plotx)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(13.5,9));fig.subplots_adjust(top=.88,bottom=.10,hspace=.34,wspace=.24)
for ax in axs[0]:
 ax.plot(plotx,kline,color='#e67e22',lw=2,label='原10000真值：加权 KDE')
 ax.plot(plotx,raw,color='#0072B2',lw=1.6,label='真实网格插值的连续积分 PDF（未平滑）')
 ax.set(xlabel='输出 y',ylabel='概率密度');ax.grid(alpha=.2)
axs[0,0].set(xlim=(0,.065),ylim=(0,105),title='未平滑的连续积分 vs KDE：主体');axs[0,0].legend(fontsize=8.5)
axs[0,1].set(xlim=(.06,.14),ylim=(1e-18,.1),yscale='log',title='未平滑的连续积分 vs KDE：上尾');axs[0,1].axvline(Y.max(),color='gray',ls=':',label='网格真值最大值');axs[0,1].legend(fontsize=8.5)
for ax in axs[1]:
 ax.plot(z['y'],z['kde'],color='#e67e22',lw=2,label='原加权 KDE')
 ax.plot(z['y'],z['physical_continuous_smoothed'],color='#0072B2',lw=1.9,label='连续积分＋相同高斯平滑')
 ax.set(xlabel='输出 y',ylabel='概率密度');ax.grid(alpha=.2)
axs[1,0].set(xlim=(-.008,.065),title='统一平滑后：主体几乎重合');axs[1,0].axvline(0,color='gray',ls=':');axs[1,0].legend(fontsize=8.5)
axs[1,1].set(xlim=(.09,.14),ylim=(1e-18,.0001),yscale='log',title='统一平滑后：剩余差异集中在极尾');axs[1,1].axvline(Y.max(),color='gray',ls=':');axs[1,1].legend(fontsize=8.5)
fig.suptitle('MMT真实系统100×100采样：连续积分与KDE的区别',fontsize=16,y=.97);fig.text(.5,.925,'上排：直接比较两种PDF；下排：给连续积分结果加相同带宽，单独检查空间积分差异。',ha='center',fontsize=10);fig.text(.5,.035,'连续积分对象：100×100真值在物理空间的分片线性插值，按高斯输入概率积分；不是新增求解的精确连续真值。',ha='center',fontsize=9.5)
fig.savefig(D/'truth_pdf_raw_and_smoothed.png',dpi=190);fig.savefig(D/'truth_pdf_raw_and_smoothed.pdf');plt.close(fig)
print('raw plot-grid integral',raw_coarse_mass,'raw max',raw.max());r=analysis['physical_interpolant_smoothed'];part=sum(k['contribution'] for k in r['e_components'] if k['y_interval'][0]>=.1);print('e',r['e_vs_original_KDE_0_to_1'],'fraction from y>=.1',part/r['e_vs_original_KDE_0_to_1'])
