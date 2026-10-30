"""Compare the physical-mesh seed-101 campaign with prior MMT USLW eta ten-seed means.
All main-panel PDF values use the historical weighted grid KDE. Continuous-law
PDF values are exported and shown only in a separate, explicitly labeled plot.
"""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OLD=ROOT.with_name('10random_seeds_MMT_2D')
RUN=ROOT/'hard_s101/results/uslw_seed101_physical'
OUT=ROOT/'figures/eta_tenseed_uslw_comparison';OUT.mkdir(parents=True,exist_ok=True)
LAMS=['0','1e-9','1e-6','0.001','1'];LABELS=[r'$\lambda=0$',r'$\lambda=10^{-9}$',r'$\lambda=10^{-6}$',r'$\lambda=10^{-3}$',r'$\lambda=1$']
COLORS=['#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7'];SEEDS=range(101,111);t=np.arange(101)
summary=np.load(OLD/'summary_seeds_uslw.npz');rows=json.loads((RUN/'history.json').read_text());assert len(rows)==101
new_ev=np.load(ROOT/'hard_s101/data/evaluation.npz');old_ev=np.load(OLD/'eta_s101/data/evaluation.npz')
for key in ['x_mc','y_mc','w']:assert np.array_equal(new_ev[key],old_ev[key]),key
truth,w=new_ev['y_mc'],new_ev['w'];order=np.argsort(truth);lo,hi=np.interp([.001,.999],np.cumsum(w[order])/w.sum(),truth[order]);mask=(truth<=lo)|(truth>=hi)
eta={}
for lam in LAMS:
 eta[lam]={metric:np.array([summary[f's{s}_eta_{lam}_{metric}'] for s in SEEDS]) for metric in ['pdf_e','ext_mse']}
 for a in eta[lam].values():assert a.shape==(10,101) and np.isfinite(a).all() and (a>0).all()
 # Source validation against stored histories, not just chart exports.
 folder='uslw_seed101' if lam=='0' else f'uslw_lam{lam}_seed101'
 hist=json.loads((OLD/'eta_s101/results'/folder/'history.json').read_text())
 assert np.allclose(eta[lam]['pdf_e'][0],[r['pdf_e_mean'] for r in hist],rtol=1e-13,atol=0)
new_pdf_ensemble=np.array([r['pdf_e_mean'] for r in rows])
new_pdf_members=np.array([r['pdf_e_members_grid_kde'] for r in rows])
new_pdf_continuous=np.array([r['pdf_e_members_continuous'] for r in rows])
new_ext_members=np.array([r['extreme_mse_members'] for r in rows]);new_ext_ensemble=[]
for k in range(101):
 pred=np.load(RUN/f'round_{k:03d}_predictions.npz')['mc'];err=(pred.mean(0)[mask]-truth[mask])**2;new_ext_ensemble.append(float(np.average(err,weights=w[mask])))
 check=np.array([np.average((p[mask]-truth[mask])**2,weights=w[mask]) for p in pred]);assert np.allclose(check,new_ext_members[k],rtol=1e-12,atol=1e-18)
new_ext_ensemble=np.array(new_ext_ensemble)
arrays={'round':t,'new_pdf_ensemble_grid_kde':new_pdf_ensemble,'new_pdf_members_grid_kde':new_pdf_members,'new_pdf_members_continuous':new_pdf_continuous,'new_ext_ensemble':new_ext_ensemble,'new_ext_members':new_ext_members}
for lam in LAMS:
 for metric,a in eta[lam].items():arrays[f'eta_{lam}_{metric}_all_seeds']=a
np.savez_compressed(OUT/'comparison_curves.npz',**arrays)
columns={'round':t,'training_labels':t+3}
for lam in LAMS:
 for metric,a in eta[lam].items():
  for stat,f in [('mean',lambda a:a.mean(0)),('min',lambda a:a.min(0)),('max',lambda a:a.max(0))]:columns[f'eta_lambda_{lam}_{metric}_10seed_{stat}']=f(a)
columns.update(new_hard_pdf_grid_kde_ensemble=new_pdf_ensemble,new_hard_pdf_grid_kde_member_average=new_pdf_members.mean(1),new_hard_pdf_continuous_member_average=new_pdf_continuous.mean(1),new_hard_ext_mse_ensemble=new_ext_ensemble,new_hard_ext_mse_member_average=new_ext_members.mean(1))
with (OUT/'comparison_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.writer(f);writer.writerow(columns);writer.writerows(zip(*columns.values()))

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'pdf.fonttype':42,'ps.fonttype':42})
for bands,suffix in [(False,'mean_curves'),(True,'mean_and_range')]:
 fig,axs=plt.subplots(1,2,figsize=(15.5,6.4));fig.subplots_adjust(top=.84,bottom=.28,left=.075,right=.98,wspace=.23)
 for j,metric in enumerate(['pdf_e','ext_mse']):
  ax=axs[j]
  for lam,label,color in zip(LAMS,LABELS,COLORS):
   a=eta[lam][metric];ax.plot(t,a.mean(0),color=color,lw=1.9,label=f'η-learning {label}：10种子平均',zorder=2)
   if bands:ax.fill_between(t,a.min(0),a.max(0),color=color,alpha=.09,lw=0,zorder=1)
  ens=new_pdf_ensemble if j==0 else new_ext_ensemble;member=new_pdf_members.mean(1) if j==0 else new_ext_members.mean(1)
  ax.plot(t,ens,color='black',lw=2.8,label='新 hard-PDF：seed101，集成预测',zorder=5)
  ax.plot(t,member,color='#707070',lw=1.7,ls='--',label='新 hard-PDF：seed101，成员指标平均',zorder=4)
  ax.set(yscale='log',xlim=(0,100),xlabel='主动学习迭代轮数（初始3个标签）')
  ax.grid(alpha=.22,which='both');ax.set_ylabel('PDF 误差 e：相同网格 KDE 口径' if j==0 else '极端区域加权 MSE：相同阈值与权重')
  ax.set_title('PDF误差：沿用原评估方法' if j==0 else '极端区域预测误差')
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.085),ncol=3,fontsize=9.5,frameon=False,columnspacing=1.7)
 fig.suptitle('MMT / USLW：物理坐标 hard-PDF 与原 η-learning 十随机种子平均曲线',fontsize=16,y=.97)
 fig.text(.5,.91,'η-learning：种子101–110，各组先计算集成预测误差，再取十组平均；新 hard-PDF：仅种子101。',ha='center',fontsize=10)
 fig.text(.5,.035,'黑实线与彩色线均使用集成预测指标；灰虚线对应此前报告的两成员指标平均。PDF连续积分结果另图展示。',ha='center',fontsize=9.5,color='#444444')
 fig.savefig(OUT/f'{suffix}.png',dpi=210);fig.savefig(OUT/f'{suffix}.pdf');plt.close(fig)
# Keep the continuous-law diagnostic visible without presenting it as the same estimator as eta's PDF curves.
fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained')
ax.plot(t,new_pdf_continuous.mean(1),color='#0072B2',lw=2.4,label='连续模型积分：成员指标平均')
ax.fill_between(t,new_pdf_continuous.min(1),new_pdf_continuous.max(1),color='#0072B2',alpha=.12)
ax.plot(t,new_pdf_members.mean(1),color='#E69F00',lw=1.8,label='原网格 KDE：成员指标平均')
ax.plot(t,new_pdf_ensemble,color='black',lw=1.5,ls='--',label='原网格 KDE：集成预测指标')
ax.axhline(1e-3,color='gray',ls=':',label='1e-3');ax.set(yscale='log',xlim=(0,100),xlabel='迭代轮数',ylabel='PDF 误差 e',title='新物理网格 hard-PDF / seed101：评估方式诊断（不与η曲线混用）');ax.legend(fontsize=10);ax.grid(alpha=.2,which='both');fig.savefig(OUT/'new_hard_pdf_metric_diagnostic.png',dpi=200);fig.savefig(OUT/'new_hard_pdf_metric_diagnostic.pdf');plt.close(fig)
report={'acquisition':'USLW','old_seeds':list(SEEDS),'new_seed':101,'new_model':'physical x/6 features, 32 physical-uniform node intervals, 9506 Q_ref knots','main_comparison':'same weighted grid-KDE PDF estimator and same extreme-region thresholds; eta ten-seed mean versus new hard single seed','extreme_thresholds':{'lo':float(lo),'hi':float(hi),'lower_mass':float(w[truth<=lo].sum()),'upper_mass':float(w[truth>=hi].sum())},'round100':{}}
for lam in LAMS:report['round100'][f'eta_lambda_{lam}']={'pdf_grid_KDE_10seed_mean':float(eta[lam]['pdf_e'][:,-1].mean()),'extreme_mse_10seed_mean':float(eta[lam]['ext_mse'][:,-1].mean())}
report['round100']['new_hard_seed101']={'pdf_grid_KDE_ensemble':float(new_pdf_ensemble[-1]),'pdf_grid_KDE_member_average':float(new_pdf_members[-1].mean()),'pdf_continuous_member_average':float(new_pdf_continuous[-1].mean()),'extreme_mse_ensemble':float(new_ext_ensemble[-1]),'extreme_mse_member_average':float(new_ext_members[-1].mean())}
(OUT/'comparison_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report['round100'],indent=2));print('Saved',OUT)
