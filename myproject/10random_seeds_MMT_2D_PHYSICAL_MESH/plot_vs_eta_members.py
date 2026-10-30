"""Hard-PDF members are evaluated individually; no averaging of the two networks.
Eta references remain the previously requested ten-seed ensemble-error means.
PDF estimators differ and are explicitly labeled in the figure and metadata.
"""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OLD=ROOT.with_name('10random_seeds_MMT_2D');RUN=ROOT/'hard_s101/results/uslw_seed101_physical';OUT=ROOT/'figures/eta_tenseed_uslw_members';OUT.mkdir(parents=True,exist_ok=True)
summary=np.load(OLD/'summary_seeds_uslw.npz');rows=json.loads((RUN/'history.json').read_text());assert len(rows)==101;t=np.arange(101)
LAMS=['0','1e-9','1e-6','0.001','1'];LABELS=[r'$\lambda=0$',r'$\lambda=10^{-9}$',r'$\lambda=10^{-6}$',r'$\lambda=10^{-3}$',r'$\lambda=1$'];COLORS=['#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7']
eta={lam:{m:np.array([summary[f's{s}_eta_{lam}_{m}'] for s in range(101,111)]) for m in ['pdf_e','ext_mse']} for lam in LAMS}
pdf=np.array([r['pdf_e_members_continuous'] for r in rows]);grid=np.array([r['pdf_e_members_grid_kde'] for r in rows]);ext=np.array([r['extreme_mse_members'] for r in rows]);train=np.array([r['train_mse_members'] for r in rows])
assert pdf.shape==ext.shape==(101,2);assert (pdf>0).all() and (ext>0).all()
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'pdf.fonttype':42})
for bands,suffix in [(False,'members_vs_eta_means'),(True,'members_vs_eta_mean_range')]:
 fig,axs=plt.subplots(1,2,figsize=(15.5,6.6));fig.subplots_adjust(top=.84,bottom=.30,left=.075,right=.98,wspace=.23)
 for j,metric in enumerate(['pdf_e','ext_mse']):
  ax=axs[j]
  for lam,label,color in zip(LAMS,LABELS,COLORS):
   a=eta[lam][metric];ax.plot(t,a.mean(0),color=color,lw=1.9,label=f'η-learning {label}：原10种子平均')
   if bands:ax.fill_between(t,a.min(0),a.max(0),color=color,alpha=.09,lw=0)
  a=pdf if j==0 else ext
  for m,col,ls in [(0,'#111111','-'),(1,'#D55E00','--')]:ax.plot(t,a[:,m],color=col,lw=2.3,ls=ls,label=f'新 hard-PDF 成员{m+1}：seed101',zorder=5)
  ax.set(xlim=(0,100),yscale='log',xlabel='主动学习迭代轮数（初始3个标签）',ylabel='PDF 误差 e' if j==0 else '极端区域加权 MSE');ax.grid(alpha=.22,which='both')
  ax.set_title('PDF：η沿用原网格KDE；hard使用连续模型积分' if j==0 else '极端区 MSE：相同评估点、阈值和权重')
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,fontsize=9.5,frameon=False)
 fig.suptitle('MMT / USLW：新 hard-PDF 两成员分别对比 η-learning 十种子平均',fontsize=16,y=.97)
 fig.text(.5,.91,'hard-PDF 的成员1、成员2分别计算误差；不计算集成预测，不对两个成员的指标取平均。',ha='center',fontsize=10)
 fig.text(.5,.055,'η曲线保留原十种子汇总（各组集成预测指标）。PDF估计方法不同，仅作趋势参考，不作为统一数值精度的排名。',ha='center',fontsize=9.5,color='#444444')
 fig.savefig(OUT/f'{suffix}.png',dpi=210);fig.savefig(OUT/f'{suffix}.pdf');plt.close(fig)
# Independent member-wise explanation for the previously high KDE curves.
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
for m,ax in enumerate(axs):
 ax.plot(t,pdf[:,m],color='#0072B2',lw=2,label='连续模型积分：本次主图采用');ax.plot(t,grid[:,m],color='#E69F00',lw=1.7,label='原网格 KDE：上一张对比图采用');ax.axhline(1e-3,ls=':',color='gray',label='1e-3');ax.set(yscale='log',xlim=(0,100),xlabel='迭代轮数',ylabel='PDF 误差 e',title=f'成员{m+1}：同一网络，两种PDF评估方法');ax.legend(fontsize=9);ax.grid(alpha=.2,which='both')
fig.savefig(OUT/'individual_pdf_estimators.png',dpi=190);fig.savefig(OUT/'individual_pdf_estimators.pdf');plt.close(fig)
columns={'round':t,'training_labels':t+3}
for lam in LAMS:
 for metric in ['pdf_e','ext_mse']:columns[f'eta_lambda_{lam}_{metric}_original_10seed_mean']=eta[lam][metric].mean(0)
for m in [0,1]:
 for label,a in [('pdf_continuous',pdf),('pdf_grid_kde',grid),('extreme_mse',ext),('training_mse',train)]:columns[f'hard_seed101_member{m+1}_{label}']=a[:,m]
with (OUT/'individual_member_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.writer(f);writer.writerow(columns);writer.writerows(zip(*columns.values()))
np.savez_compressed(OUT/'individual_member_curves.npz',**columns)
report={'hard_aggregation':'none; each member evaluated separately','hard_seed':101,'eta_reference':'original USLW 10-seed mean of ensemble prediction errors for each lambda','main_pdf_methods':{'hard':'continuous compiled model probability integration','eta':'original weighted grid gaussian_kde'},'main_extreme_mse':'same original truth thresholds, evaluation points and input-law weights','round100_members':[{'member':m+1,'pdf_continuous':float(pdf[-1,m]),'pdf_grid_kde':float(grid[-1,m]),'extreme_mse':float(ext[-1,m]),'training_mse':float(train[-1,m])} for m in [0,1]]}
(OUT/'individual_member_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
