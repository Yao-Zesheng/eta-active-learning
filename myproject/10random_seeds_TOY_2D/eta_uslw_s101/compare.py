"""Audit both independent trajectories and render a common-metric comparison."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import json, hashlib, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.stats import norm
from run import HARD, pdf, Member, LAYERS
from problem import oracle, to_unit, from_unit, input_pdf
from acquisition import kde
from mlp_repro import make_mlp

OUT=ROOT/'results/uslw_seed101'
DEST=ROOT/'comparison';DEST.mkdir(exist_ok=True)

def main():
    torch.set_num_threads(1)
    hard=json.loads((HARD/'history.json').read_text())
    ordinary=json.loads((OUT/'history.json').read_text())
    assert len(hard)==len(ordinary)==101
    ev=np.load(ROOT/'data/evaluation.npz');ref=np.load(ROOT/'data/reference_law.npz')
    scale=float(ref['scale']);threshold001=float(np.interp(.999,ref['p'],ref['q']))
    initial=np.load(OUT/'initial_lhs.npz')
    assert np.array_equal(initial['x'],np.load(HARD/'initial_lhs.npz')['x'])
    prediction={}; histories=[hard,ordinary];folders=[HARD,OUT]
    density_error=0.; scores_error=0.; direct_error=0.
    audit=dict(initial_identical=True,rounds=101,queries=100,final_n=103,
               lambda_zero=True,raw_predictions_not_clipped=True)
    for name,rows,folder in zip(['hard','ordinary'],histories,folders):
        for k,row in enumerate(rows):
            assert row['iteration']==k and row['n']==k+3
            p=np.load(folder/f'round_{k:03d}_predictions.npz')
            mean=p['mc'].mean(0)
            assert abs(np.sqrt(np.mean((mean-ev['y_mc'])**2))-row['rmse'])<1e-12
            row['top001_rmse']=float(np.sqrt(np.mean((mean[ev['y_mc']>=threshold001]-ev['y_mc'][ev['y_mc']>=threshold001])**2)))
            d=pdf(mean,p['pdf_grid'],float(p['bandwidth']))
            mask=(p['pdf_grid']>=0)&(p['pdf_grid']<=scale)
            error=np.sum(abs(np.log10(np.maximum(d[mask],1e-16))-
                    np.log10(np.maximum(p['pdf_true'][mask],1e-16))))*(.8/4096)/scale
            density_error=max(density_error,abs(error-row['pdf_e_mean']))
            assert abs(error-row['pdf_e_mean'])<1e-8
            if k<100:
                s=np.load(folder/f'round_{k:03d}_selection.npz')
                idx=int(s['selected_index'])
                assert idx==int(np.argmax(s['scores']))
                assert np.array_equal(s['query'][0],s['candidates'][idx])
                assert np.max(abs(oracle(s['query'])-s['observed']))<1e-12
                assert np.allclose(s['candidates'],from_unit(np.random.RandomState(k+3).rand(1000,2)),rtol=0,atol=0)
            if k==100:
                prediction[name]={key:p[key] for key in p.files}
                # Direct Gaussian sum validates PDF evaluation for negative raw outputs too.
                centers=p['pdf_grid'];test=centers[mask][::40]
                exact=np.array([np.mean(norm.pdf((a-mean)/float(p['bandwidth'])))/float(p['bandwidth']) for a in test])
                approx=d[mask][::40]
                direct_error=max(direct_error,float(np.max(abs(approx-exact)/np.maximum(exact,1e-12))))
        observations=np.load(folder/'observations.npz')
        assert observations['x'].shape==(103,2)
        assert np.max(abs(oracle(observations['x'])-observations['y']))<1e-12
    # Independently rebuild final checkpoint and reproduce all final MC predictions.
    checkpoint=torch.load(OUT/'round_100.pt',weights_only=False,map_location='cpu')
    assert len(checkpoint['x'])==103 and checkpoint['seeds']==[201,202]
    reproduced=[]
    for state in checkpoint['states']:
        net=make_mlp(LAYERS);net.load_state_dict(state);net.eval()
        reproduced.append(Member(net).predict(to_unit(ev['x_mc']))*scale)
    reproduction_error=float(np.max(abs(np.array(reproduced)-prediction['ordinary']['mc'])))
    assert reproduction_error==0
    # Recompute acquisition scores independently for representative rounds.
    for k in [0,49,99]:
        s=np.load(OUT/f'round_{k:03d}_selection.npz');mu=s['members'].mean(0)
        padding=1e-6*np.ptp(mu)+1e-12; grid=np.linspace(mu.min()-padding,mu.max()+padding,10000)
        recomputed=s['members'].var(0)*input_pdf(s['candidates'])/np.maximum(np.interp(mu,grid,kde(mu,grid)),1e-16)
        scores_error=max(scores_error,float(np.max(abs(recomputed-s['scores']))))
    audit.update(max_pdf_e_recomputation_error=density_error,
                 direct_gaussian_pdf_max_relative_error=direct_error,
                 checkpoint_prediction_max_error=reproduction_error,
                 acquisition_recomputation_max_error=scores_error)
    assert direct_error<1e-3 and scores_error<1e-12
    (DEST/'audit.json').write_text(json.dumps(audit,indent=2))
    colors=['#e07820','#2678ad'];labels=['Hard PDF + USLW','MSE (lambda=0) + USLW']
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    fields=[('pdf_e_mean','PDF error e (common fixed bandwidth)'),('rmse','Global RMSE under p_X'),
            ('top1_rmse','True upper 1% RMSE'),('top001_rmse','True upper 0.1% RMSE'),
            ('train_mse_mean','Observed-point MSE'),('disagreement_rms','Member disagreement RMS')]
    for ax,(key,title) in zip(axs.flat,fields):
        for rows,label,color in zip(histories,labels,colors):
            ax.plot(range(101),[r[key] for r in rows],label=label,color=color,lw=1.3)
        ax.set(title=title,xlabel='Acquired observations (initial n=3)',yscale='log',xlim=(0,100))
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Independent active learning | same initial LHS seed 101 | 100 queries + final refit')
    fig.savefig(DEST/'learning_curves.png');plt.close(fig)
    axis=ev['axis'];shape=(len(axis),len(axis));truth=ev['y_grid']
    means=[prediction[n]['grid'].mean(0) for n in ['hard','ordinary']]
    fields_y=[truth,*means];vmin=min(0.,*(float(v.min()) for v in fields_y));vmax=max(float(v.max()) for v in fields_y)
    fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    fig.get_layout_engine().set(h_pad=.12,hspace=.12)
    for ax,values,title in zip(axs[0],fields_y,['Truth',*labels]):
        im=ax.pcolormesh(axis,axis,values.reshape(shape).T,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.set(title=title,ylabel='x2',aspect='equal');ax.grid(False)
        ax.plot(2,-2,'r+',ms=10)
    fig.colorbar(im,ax=axs[0].tolist(),label='y(x), common color scale',shrink=.85)
    error_max=max(float(abs(v-truth).max()) for v in means)
    axs[1,0].axis('off')
    text='100 newly queried labels; final n = 103\n\nBoth models: raw ensemble mean\nNo clipping of negative outputs\nRed cross: true grid maximum (2, -2)\n\nNetworks / training budgets differ.\nThis compares the requested pipelines,\nnot a controlled architecture ablation.'
    axs[1,0].text(.02,.95,text,va='top',transform=axs[1,0].transAxes,linespacing=1.7)
    for ax,values,label in zip(axs[1,1:],means,labels):
        im=ax.pcolormesh(axis,axis,abs(values-truth).reshape(shape).T,shading='auto',cmap='magma',vmin=0,vmax=error_max)
        ax.set(title=label+' | absolute error',xlabel='x1',ylabel='x2',aspect='equal');ax.grid(False)
    fig.colorbar(im,ax=axs[1,1:].tolist(),label='Absolute error, common color scale',shrink=.85)
    fig.savefig(DEST/'final_maps.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    ph=prediction['hard'];axs[0].semilogy(ph['pdf_grid'],np.maximum(ph['pdf_true'],1e-16),'k',label='Truth')
    for name,label,color in zip(['hard','ordinary'],labels,colors):
        p=prediction[name];axs[0].semilogy(p['pdf_grid'],np.maximum(p['pdf_predictions'][2],1e-16),label=label,color=color)
    axs[0].set(xlim=(0,scale),ylim=(1e-3,100),xlabel='y',ylabel='PDF',title='Final PDFs (same smoothing)');axs[0].legend(fontsize=8)
    for ax,folder,label in zip(axs[1:],folders,labels):
        obs=np.load(folder/'observations.npz')
        ax.contour(axis,axis,truth.reshape(shape).T,levels=[.1,.3,.433,.515],colors='gray',linewidths=.8)
        sc=ax.scatter(obs['x'][3:,0],obs['x'][3:,1],c=np.arange(1,101),s=18,cmap='plasma')
        ax.scatter(obs['x'][:3,0],obs['x'][:3,1],marker='x',color='black',s=60,label='Initial LHS')
        ax.set(xlim=(-8,8),ylim=(-8,8),aspect='equal',title=label+' | selected points',xlabel='x1',ylabel='x2')
    fig.colorbar(sc,ax=axs[1:].tolist(),label='Query number',shrink=.8)
    fig.savefig(DEST/'pdf_and_sampling.png');plt.close(fig)
    keys=['pdf_e_mean','rmse','relative_l2','r2','top1_rmse','top001_rmse','top1_recall','grid_rmse','grid_max_abs','train_mse_mean','disagreement_rms','peak_distance_mean']
    summary={name:dict(final={key:rows[-1][key] for key in keys},
        last21={key:dict(min=float(min(r[key] for r in rows[80:])),median=float(np.median([r[key] for r in rows[80:]])),max=float(max(r[key] for r in rows[80:]))) for key in ['pdf_e_mean','rmse','top1_rmse']},
        total_seconds=float(sum(r['seconds'] for r in rows)) if 'seconds' in rows[0] else None,
        query_variance_decreased=sum(r['previous_query_variance_after']<r['previous_query_variance_before'] for r in rows[1:])) for name,rows in zip(['hard','ordinary'],histories)}
    summary['ordinary']['final'].update(negative_fraction=ordinary[-1]['negative_fraction'],prediction_min=ordinary[-1]['prediction_min'],prediction_max=ordinary[-1]['prediction_max'])
    for name,folder in zip(['hard','ordinary'],folders):
        obs=np.load(folder/'observations.npz'); values=obs['y'];points=obs['x']
        extreme=values[3:]>=float(np.interp(.99,ref['p'],ref['q']))
        near_peak=np.linalg.norm(points[3:]-[2,-2],axis=1)<.5
        summary[name]['sampling']=dict(max_observed_y=float(values.max()),
            upper1_observations=int(np.sum(values>=float(np.interp(.99,ref['p'],ref['q'])))),
            upper001_observations=int(np.sum(values>=threshold001)),
            first_upper1_query=int(np.flatnonzero(extreme)[0]+1) if extreme.any() else None,
            first_query_within_half_unit_of_true_peak=int(np.flatnonzero(near_peak)[0]+1) if near_peak.any() else None)
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2))
    with (DEST/'learning_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['iteration','n']+[name+'_'+key for name in ['hard','ordinary'] for key in keys])
        for k in range(101): writer.writerow([k,k+3]+[rows[k][key] for rows in histories for key in keys])
    table=''.join(f'<tr><td>{key}</td><td>{hard[-1][key]:.7g}</td><td>{ordinary[-1][key]:.7g}</td></tr>' for key in keys)
    stability=''
    for key in ['pdf_e_mean','rmse','top1_rmse']:
        cells=[]
        for name in ['hard','ordinary']:
            stats=summary[name]['last21'][key]
            cells.append(f'<td>{stats["median"]:.6g} [{stats["min"]:.6g}, {stats["max"]:.6g}]</td>')
        stability+=f'<tr><td>{key}</td>'+''.join(cells)+'</tr>'
    report=f'''<!DOCTYPE html><html lang="zh"><meta charset="utf-8"><title>Toy 普通与硬约束 E-NN 对比</title>
    <style>body{{font:17px/1.7 system-ui;max-width:1400px;margin:35px auto;padding:0 25px;color:#203040}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px 18px}}code{{background:#eee}}</style>
    <h1>官方 toy：普通 E-NN（λ=0）与硬约束 E-NN 的独立 USLW 主动学习</h1>
    <p>两种方法均从相同的 3 个 LHS 点（seed=101）出发，独立选择 100 个新点，最终使用全部 103 个观测重新训练。横轴 0 表示初始 3 点。普通算法来自 POIS_2D/mlp_repro.py 的原样副本，λ=0，无 W1、无预训练、无输出截断。候选集为每轮 1000 个高斯样本，选择 USLW 最大值，并非连续全局优化的证明。</p>
    <p>共同输入为完整 N(0,10I)，模型输入用 2Φ(x/√10)−1，y 用共同常数 {scale:.9f} 缩放。普通模型没有读取参考分位数来训练；参考分位数仅供评估极端区。测试使用相同的独立 100000 个输入，云图为 [-8,8]² 的 201×201 网格。没有用测试误差挑选训练轮次。</p>
    <p>PDF 误差 e 是两成员均值的输出分布，在 y∈[0,{scale:.6f}] 上积分 |log₁₀ p̂−log₁₀ p| 并除以 {scale:.6f}。两种方法共用真实测试输出确定的固定带宽 0.01159545、密度下限 10⁻¹⁶、线性质量分配和 10σ 核。普通模型的负值和越界值保留并参与归一化，未丢弃或截断。本指标与旧 Poiseuille 工程不同范围/带宽下的 e 不能直接比较。</p>
    <table><tr><th>最终指标</th><th>硬约束</th><th>普通 λ=0</th></tr>{table}</table>
    <p>本次结果中，普通模型在真实输入概率分布下的 RMSE 和极端区误差更小，硬约束模型的最终 PDF 误差略小、均匀网格 RMSE 更小。普通模型测试预测中有 {ordinary[-1]['negative_fraction']:.2%} 为负，最小值 {ordinary[-1]['prediction_min']:.6f}，体现了远离峰值区域时的非物理外推；本报告保留这些原始输出。这也解释了概率加权测试与均匀网格评价的排序不同。普通方法第 {summary['ordinary']['sampling']['first_query_within_half_unit_of_true_peak']} 次查询进入最高峰半径 0.5 内，硬约束为第 {summary['hard']['sampling']['first_query_within_half_unit_of_true_peak']} 次；但这只是本次采样轨迹的观测，不能作为普遍的因果结论。</p>
    <h2>第 80—100 轮：中位数 [最小值, 最大值]</h2>
    <table><tr><th>指标</th><th>硬约束</th><th>普通 λ=0</th></tr>{stability}</table>
    <h2>全部 101 个训练状态的原始误差曲线</h2><img src="learning_curves.png">
    <h2>第 100 次采样后：真实值、预测均值、绝对误差</h2><img src="final_maps.png">
    <h2>PDF 与各自的采样轨迹</h2><img src="pdf_and_sampling.png">
    <p>解释边界：普通模型为 5×200 ReLU、Adam 1000 步、float32；硬约束模型为 2×48 Tanh + 32×32 分片概率投影、Adam 1800 步 + LBFGS、float64。二者均每轮冷启动种子 2k+1、2k+2。本次只能比较这两套配置及单次初始设计，不能分离网络容量、优化器和 PDF 约束的贡献。硬约束保证的是每个成员相对于数值参考分布的边缘分布，成员均值没有同样的严格保证；边缘 PDF 正确也不等于 x→y 映射正确。</p>
    <p><a href="summary.json">最终及后 21 轮统计</a> · <a href="learning_curves.csv">完整曲线数值</a> · <a href="audit.json">复核结果</a></p></html>'''
    (DEST/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2));print('AUDIT',json.dumps(audit))

if __name__=='__main__':main()
