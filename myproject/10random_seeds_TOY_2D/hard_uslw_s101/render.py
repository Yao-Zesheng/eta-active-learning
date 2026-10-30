"""Render saved Hard-PDF members individually; ordinary ensemble uses its mean."""
from pathlib import Path
import os,json,csv,hashlib
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROUNDS=list(range(1,11))+list(range(20,101,10))
HARD=ROOT/'results/uslw_seed101'
BASE=ROOT.parent/'TOY_2D_ETA_LAMBDA0_USLW/results/uslw_seed101'
manifest={}
def track(path):
    manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path
h=json.loads(track(HARD/'history.json').read_text())
b=json.loads(track(BASE/'history.json').read_text())
ev=np.load(track(ROOT.parent/'TOY_2D_ETA_LAMBDA0_USLW/data/evaluation.npz'))
ref=np.load(track(ROOT.parent/'TOY_2D_ETA_LAMBDA0_USLW/data/reference_law.npz'))
axis=ev['axis'];truth=ev['y_grid'];shape=(len(axis),len(axis));scale=float(ref['scale'])
data={};rows=[];pdf_check=0.
for k in range(101):
    assert h[k]['iteration']==b[k]['iteration']==k
    assert h[k]['n']==b[k]['n']==k+3
    p=np.load(track(HARD/f'round_{k:03d}_predictions.npz'))
    mask=(p['pdf_grid']>=0)&(p['pdf_grid']<=scale)
    errors=np.sum(abs(np.log10(np.maximum(p['pdf_predictions'][:2,mask],1e-16))-
        np.log10(np.maximum(p['pdf_true'][mask],1e-16))),axis=1)*(.8/4096)/scale
    pdf_check=max(pdf_check,float(np.max(abs(errors-np.array(h[k]['pdf_e_members'])))))
    rmse=np.sqrt(np.mean((p['mc']-ev['y_mc'])**2,axis=1))
    assert np.allclose(rmse,[m['rmse'] for m in h[k]['member_metrics']],rtol=0,atol=1e-12)
    rows.append([k,k+3,*h[k]['pdf_e_members'],b[k]['pdf_e_mean'],*rmse,b[k]['rmse'],
        *[m['top1_rmse'] for m in h[k]['member_metrics']],b[k]['top1_rmse'],
        *h[k]['train_mse_members'],b[k]['train_mse_mean']])
    if k in ROUNDS:
        q=np.load(track(BASE/f'round_{k:03d}_predictions.npz'))
        data[k]=[p['grid'][0],p['grid'][1],q['grid'].mean(0)]
        assert all(a.shape==truth.shape and np.isfinite(a).all() for a in data[k])
assert pdf_check<1e-8
arr=np.array(rows)
plt.rcParams.update({'font.size':10,'figure.dpi':150})
labels=['Hard-PDF NN1','Hard-PDF NN2','Ordinary E-NN mean (lambda=0)']
colors=['#d66b18','#8144b5','#247ca6']
def curves(ax,columns,title,log=True):
    for col,label,color in zip(columns,labels,colors):ax.plot(arr[:,0],arr[:,col],label=label,color=color,lw=1.3)
    ax.set(title=title,xlabel='Acquired observations k (training n=3+k)',xlim=(0,100))
    if log:ax.set_yscale('log')
    ax.grid(alpha=.25)
fig,axs=plt.subplots(1,2,figsize=(13,4.3),layout='constrained')
for j in range(2):axs[0].plot(arr[:,0],arr[:,2+j],label=labels[j],color=colors[j],lw=1.4)
axs[0].set(title='Hard-PDF: individual member PDF errors',xlabel='Acquired observations k',ylabel='PDF error e',xlim=(0,100),ylim=(0,max(.02,float(arr[:,2:4].max())*1.1)))
axs[0].grid(alpha=.25);axs[0].legend()
curves(axs[1],[2,3,4],'PDF error: individual Hard-PDF members vs ordinary mean');axs[1].legend(fontsize=8)
fig.savefig(ROOT/'pdf_member_curves.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,cols,title in zip(axs.flat,[[2,3,4],[5,6,7],[8,9,10],[11,12,13]],
        ['PDF error e','Global RMSE under Gaussian input law','True upper 1% RMSE','Observed-point MSE']):curves(ax,cols,title)
axs[0,0].legend(fontsize=8);fig.savefig(ROOT/'all_member_error_curves.png');plt.close(fig)
vmin=min(0.,*(v.min() for triplet in data.values() for v in triplet))
vmax=max(truth.max(),*(v.max() for triplet in data.values() for v in triplet))
norm=Normalize(vmin,vmax)
def maps(rounds,path):
    fig,axs=plt.subplots(len(rounds),4,figsize=(17,4.1*len(rounds)),squeeze=False,layout='constrained')
    for i,k in enumerate(rounds):
        for j,(values,title) in enumerate(zip([truth,*data[k]],['Truth',*labels])):
            ax=axs[i,j]
            im=ax.pcolormesh(axis,axis,values.reshape(shape).T,shading='auto',cmap='viridis',norm=norm)
            extra=f'Iteration {k} | n={k+3}'
            if j:extra+=f' | RMSE={arr[k,4+j]:.4f}'
            ax.set(title=title+'\n'+extra,xlabel='x1',ylabel='x2',aspect='equal',xticks=[-8,-4,0,4,8],yticks=[-8,-4,0,4,8])
            ax.title.set_fontsize(9)
        fig.colorbar(im,ax=axs[i,:].tolist(),label='y(x) | fixed scale for all rounds',shrink=.8)
    fig.savefig(path);plt.close(fig)
for k in ROUNDS:maps([k],ROOT/f'iteration_{k:03d}.png')
groups=[ROUNDS[i:i+6] for i in range(0,len(ROUNDS),6)]
for group in groups:maps(group,ROOT/f'overview_{group[0]:03d}_{group[-1]:03d}.png')
with (ROOT/'member_error_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['iteration','n','hard_nn1_pdf_e','hard_nn2_pdf_e','ordinary_mean_pdf_e',
        'hard_nn1_rmse','hard_nn2_rmse','ordinary_mean_rmse','hard_nn1_top1_rmse','hard_nn2_top1_rmse','ordinary_mean_top1_rmse',
        'hard_nn1_train_mse','hard_nn2_train_mse','ordinary_mean_train_mse']);writer.writerows(rows)
cards=''.join(f'<section id="k{k}"><h2>第 {k} 轮：{k+3} 个观测</h2><a href="iteration_{k:03d}.png"><img loading="lazy" src="iteration_{k:03d}.png"></a></section>' for k in ROUNDS)
summary={}
for j,name in enumerate(['hard_nn1','hard_nn2','ordinary_mean']):
    keys={'pdf_e':2+j,'rmse':5+j,'top1_rmse':8+j,'train_mse':11+j}
    summary[name]={key:float(arr[-1,col]) for key,col in keys.items()}
    summary[name]['last21']={key:dict(median=float(np.median(arr[80:,col])),min=float(arr[80:,col].min()),max=float(arr[80:,col].max())) for key,col in keys.items()}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
table=''.join('<tr><td>'+key+'</td>'+''.join(f'<td>{summary[name][key]:.7g}</td>' for name in summary)+'</tr>' for key in ['pdf_e','rmse','top1_rmse','train_mse'])
nav=' '.join(f'<a href="#k{k}">{k}</a>' for k in ROUNDS)
links=' · '.join(f'<a href="overview_{g[0]:03d}_{g[-1]:03d}.png">{", ".join(map(str,g))} 轮总览</a>' for g in groups)
html=f'''<!DOCTYPE html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hard-PDF 两成员独立比较</title>
<style>body{{font:17px/1.65 system-ui;max-width:1700px;margin:25px auto;padding:0 20px;color:#24364a}}img{{width:100%}}a{{color:#17649a}}nav{{position:sticky;top:0;background:#edf3f8;padding:10px;z-index:2}}nav a{{padding:4px 10px;display:inline-block}}section{{scroll-margin-top:90px;border-bottom:1px solid #ddd}}</style>
<h1>Hard-PDF NN1、NN2 分别展示；普通 E-NN 展示均值</h1>
<p>新版 Hard-PDF 已从 3 个初始 LHS 点（seed=101）重新独立执行 100 次 USLW 查询，并用全部 103 个观测完成最终重训。以下 PDF 误差和云图均采用它的两个单独成员；普通 E-NN（λ=0）沿用已完成的基线，展示两成员平均预测。旧工程和旧结果没有修改。</p>
<p><strong>两种模型的 MLP 均为 2→200→200→200→200→200→1、5 个 ReLU 隐藏层，每成员 161601 个参数，包含相同偏置、单精度和初始化规则。</strong>Hard-PDF 另有不含可训练参数的 32×32 三角网格及解析 CDF/参考分位数变换；此变换采用双精度。最终部署的 Hard-PDF 函数因此仍不同于直接 MLP。</p>
<p>仍存在的训练差别：Hard-PDF 使用 Adam 最多 1800 步、学习率 0.002，随后 LBFGS 最多 450 步，以训练 MSE 选择有效参数；达到归一化 RMSE 10⁻⁴ 时提前结束。普通模型使用 Adam 1000 步、学习率 0.001、最后一步参数。因而本次统一了网络配置，仍不是完全相同优化器预算的消融实验。</p>
<p>每轮冷启动，NN1、NN2 是成员槽位，第 k 轮初始化种子为 2k+1、2k+2；第 k 轮训练点数为 3+k。USLW 继续使用成员均值和方差选择 1000 个高斯候选点中的最大值；两个单成员的 PDF 约束不意味着用于选点的均值也受同样约束。</p>
<h2>第 100 轮结果</h2><table border="1" cellpadding="8" style="border-collapse:collapse"><tr><th>指标</th><th>Hard-PDF NN1</th><th>Hard-PDF NN2</th><th>普通 E-NN 均值</th></tr>{table}</table><p><a href="summary.json">最终及第 80—100 轮统计</a> · <a href="run_audit.json">采样与模型复核</a></p>
<h2>完整运行后的结果解读</h2>
<p>本次新版 Hard-PDF 的两个成员在总体 RMSE 和 PDF 误差上均优于普通 E-NN 均值，但最高 1% 区域误差仍较大。第 80—100 轮总体 RMSE 中位数分别为 {summary['hard_nn1']['last21']['rmse']['median']:.6f}、{summary['hard_nn2']['last21']['rmse']['median']:.6f}，普通模型为 {summary['ordinary_mean']['last21']['rmse']['median']:.6f}；相应最高 1% 区域 RMSE 中位数为 {summary['hard_nn1']['last21']['top1_rmse']['median']:.6f}、{summary['hard_nn2']['last21']['top1_rmse']['median']:.6f} 和 {summary['ordinary_mean']['last21']['top1_rmse']['median']:.6f}。因此，整体预测改善不等于最极端区域也最优。</p>
<p>曲线显示新版约在第 50—60 轮出现明显改善。第 100 轮两个成员的最高峰网格位置均为 {h[-1]['predicted_peaks_grid'][0]}，真实最高峰为 {h[-1]['true_peak_grid']}；普通模型最终网格最高峰位于真实峰位置。Hard-PDF 仍使用 32×32 分片网格，单纯增加 MLP 层数与宽度并没有增加该空间表示的分辨率。这个残余峰位偏差与极端区误差同时存在，需进一步做网格分辨率实验才能判断其因果贡献。</p>
<p>两个 Hard-PDF 成员的最终全域分歧 RMS 为 {h[-1]['disagreement_rms']:.6f}，已较小，但相似预测也可能共享局部偏差。PDF 误差仍是对数密度估计误差；每成员满足数值参考边缘分布，不代表这项有限样本估计误差为零，也不保证峰的位置正确。本次是一个初始设计和种子序列的实验，且优化预算仍不同，不能据此断言所有问题上硬约束都更好。</p>
<h2>各成员 PDF 误差：0—100 轮</h2><img src="pdf_member_curves.png">
<p>左图用线性纵轴展示两个成员的小误差；右图用对数纵轴与普通模型均值对比。误差口径与此前一致：同一测试集、固定带宽、log₁₀ 密度差积分及 10⁻¹⁶ 密度下限。数值参考分布与有限样本 PDF 估计使成员误差不严格为零。</p>
<h2>PDF、全域、极端区和训练点误差</h2><img src="all_member_error_curves.png"><p><a href="member_error_curves.csv">下载全部曲线数值</a></p>
<h2>18 个指定迭代及第 100 轮的四列云图</h2><p>从左到右：真实值｜Hard-PDF NN1｜Hard-PDF NN2｜普通 E-NN 均值。全部图共用色标 [{vmin:.6f}, {vmax:.6f}]，保留负值与超出真实峰值的预测。标题 RMSE 按高斯输入分布的独立测试集计算。</p>
<p>{links}</p><nav>跳转：{nav}</nav>{cards}</html>'''
(ROOT/'report.html').write_text(html,encoding='utf-8')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
audit=dict(rounds=ROUNDS,curves_iterations=list(range(101)),hard_maps='individual members 0 and 1',
    ordinary_maps='mean over both members',max_pdf_e_check_error=pdf_check,common_vmin=float(vmin),common_vmax=float(vmax),
    member_pdf_ranges=[dict(min=float(arr[:,j].min()),max=float(arr[:,j].max())) for j in [2,3]],
    source_hashes_unchanged=True,source_sha256=manifest)
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},indent=2))
print('DONE: 19 four-column maps, 4 overview maps, 2 curve figures, CSV, report.html')
