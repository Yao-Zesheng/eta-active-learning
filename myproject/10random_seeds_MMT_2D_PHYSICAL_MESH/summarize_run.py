"""Generate progress/completion artifacts; safe to rerun while training."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'hard_s101/results/uslw_seed101_physical';D=ROOT/'diagnostics';D.mkdir(exist_ok=True)
rows=json.loads((OUT/'history.json').read_text());k=len(rows)-1
report={'last_completed_iteration':k,'labels':rows[-1]['n'],'continuous_pdf_e_members':rows[-1]['pdf_e_members_continuous'],'grid_pdf_e_members':rows[-1]['pdf_e_members'],'extreme_mse_members':rows[-1]['extreme_mse_members'],'upper_tail_mse_members':rows[-1]['upper_tail_mse_members'],'train_mse_members':rows[-1]['train_mse_members']}
oldroot=ROOT.with_name('10random_seeds_MMT_2D_QREF_TAIL');oldout=oldroot/'hard_s101/results/uslw_seed101_refined';oldrows=json.loads((oldout/'history.json').read_text());ev=np.load(ROOT/'hard_s101/data/evaluation.npz');y,w=ev['y_mc'],ev['w'];ids=np.argsort(y);lo,hi=np.interp([.001,.999],np.cumsum(w[ids])/w.sum(),y[ids]);mask=(y<=lo)|(y>=hi)
oldext=[]
for j in range(k+1):
 p=np.load(oldout/f'round_{j:03d}_predictions.npz')['mc'];oldext.append([float(np.average((a[mask]-y[mask])**2,weights=w[mask])) for a in p])
report['previous_probability_mesh_same_round']={'grid_pdf_e_members':oldrows[k]['pdf_e_members'],'extreme_mse_members':oldext[-1],'train_mse_members':oldrows[k]['train_mse_members']}
(D/'progress_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');t=np.arange(k+1)
for key,label in [('pdf_e_members_continuous','连续模型积分'),('pdf_e_members','原网格 KDE')]:
 a=np.array([r[key] for r in rows]);axs[0].plot(t,a.mean(1),label=label)
axs[0].axhline(1e-3,color='gray',ls=':');axs[0].set(title='新模型：两种 PDF 评估',yscale='log',xlabel='迭代轮数',ylabel='两成员平均 e');axs[0].legend()
for a,label in [(np.array(oldext),'原概率坐标与网格'),(np.array([r['extreme_mse_members'] for r in rows]),'物理坐标与物理均匀节点')]:axs[1].plot(t,a.mean(1),label=label)
axs[1].set(title='同轮次极端区 MSE 对比',yscale='log',xlabel='迭代轮数',ylabel='两成员平均 MSE');axs[1].legend()
a=np.array([r['train_mse_members'] for r in rows]);axs[2].plot(t,a[:,0],label='成员1');axs[2].plot(t,a[:,1],label='成员2');axs[2].set(title='新模型训练 MSE',yscale='log',xlabel='迭代轮数');axs[2].legend()
for ax in axs:ax.grid(alpha=.2)
fig.savefig(D/'progress_curves.png',dpi=170);plt.close(fig);print(json.dumps(report,indent=2))
