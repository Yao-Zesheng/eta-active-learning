from pathlib import Path
import json,numpy as np
from scipy.stats import gaussian_kde
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('10random_seeds_MMT_2D_PHYSICAL_MESH');D=R/'diagnostics/truth_pdf_integration_vs_kde';e=np.load(R/'hard_s101/data/evaluation.npz');x,y,w=e['x_mc'],e['y_mc'],e['w'];a=json.loads((D/'analysis.json').read_text());z=np.load(D/'truth_pdf_curves.npz');h=a['kde_bandwidth'];factors=np.ones(len(w))
for j in [0,1]:factors[(x[:,j]==x[:,j].min())|(x[:,j]==x[:,j].max())]*=.5
wt=w*factors;wt/=wt.sum();mu=wt@y;var=np.sum(wt*(y-mu)**2)/(1-np.sum(wt**2));kt=gaussian_kde(y,weights=wt,bw_method=h/np.sqrt(var));assert abs(float(np.sqrt(kt.covariance[0,0]))-h)<1e-15
k=gaussian_kde(y,weights=w);g=np.linspace(0,1,1000);p=np.maximum(np.interp(g,z['y'],z['physical_continuous_smoothed'],left=0,right=0),1e-16);old=np.maximum(k(g),1e-16);trap=np.maximum(kt(g),1e-16);err=abs(np.log10(trap[:-1])-np.log10(p[:-1]))/999;old_err=abs(np.log10(old[:-1])-np.log10(p[:-1]))/999
r={'original_point_weight_KDE_vs_physical_integral_e':float(old_err.sum()),'trapezoidal_boundary_weight_KDE_vs_physical_integral_e':float(err.sum()),'e_reduction_fraction':float(1-err.sum()/old_err.sum()),'edge_node_weight_multiplier':.5,'corner_node_weight_multiplier':.25,'Gaussian_bandwidth_unchanged':h,'original_KDE_tail_probability_Y_above_.11':float(np.sum(w*ndtr((y-.11)/h))),'trapezoid_KDE_tail_probability_Y_above_.11':float(np.sum(wt*ndtr((y-.11)/h))),'physical_integral_tail_probability_Y_above_.11':next(t['physical_interpolant_smoothed'] for t in a['tail_probabilities'] if t['threshold']==.11)}
(D/'boundary_weight_control.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42});fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');line=np.linspace(.085,.14,1501);old=k(line);new=kt(line);phys=np.interp(line,z['y'],z['physical_continuous_smoothed'])
axs[0].semilogy(line,old,color='#e67e22',label='原 KDE：边界与内部节点同样计权',lw=2);axs[0].semilogy(line,phys,color='#0072B2',label='物理插值连续积分：同带宽平滑',lw=2);axs[0].semilogy(line,new,color='#009E73',ls='--',label='KDE 改用梯形边界权重：其余不变',lw=2);axs[0].set(ylim=(1e-17,.001),xlabel='输出 y',ylabel='概率密度',title='控制实验：边界半权重后，上尾明显接近');axs[0].legend(fontsize=8.5);axs[0].grid(alpha=.2)
axs[1].plot(line,100*(old/np.maximum(phys,1e-100)-1),color='#e67e22',label='原 KDE');axs[1].plot(line,100*(new/np.maximum(phys,1e-100)-1),color='#009E73',ls='--',label='梯形边界权重 KDE');axs[1].axhline(0,color='gray',lw=.8);axs[1].set(xlim=(.09,.125),ylim=(-10,250),xlabel='输出 y',ylabel='相对连续积分的差异（%）',title='保留同一高斯带宽，单独改变节点积分权重');axs[1].legend();axs[1].grid(alpha=.2);fig.savefig(D/'truth_pdf_boundary_weights.png',dpi=190);fig.savefig(D/'truth_pdf_boundary_weights.pdf');plt.close(fig)
