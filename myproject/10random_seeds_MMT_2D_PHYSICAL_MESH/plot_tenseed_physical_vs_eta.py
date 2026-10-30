"""Ten-seed comparison: physical-mesh hard-PDF (seeds 101-110) vs the prior
MMT USLW eta-lambda ten-seed campaign.

Main panels use the SAME estimators on both sides: the historical weighted
grid-KDE PDF error and the input-law-weighted two-tail extreme MSE. The
continuous-law PDF metric of the new model is exported and shown only in a
separate, explicitly labeled diagnostic figure.
Outputs one Chinese and one English comparison figure.
"""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD = ROOT.with_name('10random_seeds_MMT_2D')
OUT = ROOT / 'figures/tenseed_uslw_comparison'; OUT.mkdir(parents=True, exist_ok=True)
LAMS = ['0', '1e-9', '1e-6', '0.001', '1']
LABELS = [r'$\lambda=0$', r'$\lambda=10^{-9}$', r'$\lambda=10^{-6}$', r'$\lambda=10^{-3}$', r'$\lambda=1$']
COLORS = ['#0072B2', '#56B4E9', '#009E73', '#E69F00', '#CC79A7']
SEEDS = range(101, 111); t = np.arange(101)

summary = np.load(OLD / 'summary_seeds_uslw.npz')
ref_ev = np.load(ROOT / 'hard_s101/data/evaluation.npz')
old_ev = np.load(OLD / 'eta_s101/data/evaluation.npz')
for key in ['x_mc', 'y_mc', 'w']:
    assert np.array_equal(ref_ev[key], old_ev[key]), key
truth, w = ref_ev['y_mc'], ref_ev['w']
order = np.argsort(truth)
cw = np.cumsum(w[order]) / w.sum()
lo, hi = np.interp([.001, .999], cw, truth[order])
mask = (truth <= lo) | (truth >= hi)

eta = {}
for lam in LAMS:
    eta[lam] = {m: np.array([summary[f's{s}_eta_{lam}_{m}'] for s in SEEDS]) for m in ['pdf_e', 'ext_mse']}
    for a in eta[lam].values():
        assert a.shape == (10, 101) and np.isfinite(a).all() and (a > 0).all()
    folder = 'uslw_seed101' if lam == '0' else f'uslw_lam{lam}_seed101'
    hist = json.loads((OLD / 'eta_s101/results' / folder / 'history.json').read_text())
    assert np.allclose(eta[lam]['pdf_e'][0], [r['pdf_e_mean'] for r in hist], rtol=1e-13, atol=0)

new = {'pdf_ens': [], 'pdf_mem': [], 'pdf_cont': [], 'ext_mem': [], 'ext_ens': []}
for s in SEEDS:
    run = ROOT / f'hard_s{s}/results/uslw_seed{s}_physical'
    rows = json.loads((run / 'history.json').read_text()); assert len(rows) == 101
    ev_s = np.load(ROOT / f'hard_s{s}/data/evaluation.npz')
    for key in ['x_mc', 'y_mc', 'w']:
        assert np.array_equal(ev_s[key], ref_ev[key]), (s, key)
    new['pdf_ens'].append([r['pdf_e_mean'] for r in rows])
    new['pdf_mem'].append([r['pdf_e_members_grid_kde'] for r in rows])
    new['pdf_cont'].append([r['pdf_e_members_continuous'] for r in rows])
    ext_mem_rows = np.array([r['extreme_mse_members'] for r in rows])
    ext_ens = []
    for k in range(101):
        pred = np.load(run / f'round_{k:03d}_predictions.npz')['mc']
        err = (pred.mean(0)[mask] - truth[mask]) ** 2
        ext_ens.append(float(np.average(err, weights=w[mask])))
        check = np.array([np.average((p[mask] - truth[mask]) ** 2, weights=w[mask]) for p in pred])
        assert np.allclose(check, ext_mem_rows[k], rtol=1e-12, atol=1e-18), (s, k)
    new['ext_mem'].append(ext_mem_rows)
    new['ext_ens'].append(ext_ens)
    print(f'seed {s} curves loaded & cross-checked', flush=True)
new = {k: np.array(v) for k, v in new.items()}
# member-level entries are (seeds, rounds, 2 members); precompute per-round member averages
new['pdf_mem_avg'] = new['pdf_mem'].mean(2)
new['pdf_cont_avg'] = new['pdf_cont'].mean(2)
new['ext_mem_avg'] = new['ext_mem'].mean(2)

arrays = {'round': t}
for lam in LAMS:
    for m, a in eta[lam].items():
        arrays[f'eta_{lam}_{m}_all_seeds'] = a
arrays.update(new_hard_pdf_ensemble_grid_kde_all_seeds=new['pdf_ens'],
              new_hard_pdf_members_grid_kde_all_seeds=new['pdf_mem'],
              new_hard_pdf_members_continuous_all_seeds=new['pdf_cont'],
              new_hard_ext_mse_ensemble_all_seeds=new['ext_ens'],
              new_hard_ext_mse_members_all_seeds=new['ext_mem'])
np.savez_compressed(OUT / 'tenseed_curves.npz', **arrays)
columns = {'round': t, 'training_labels': t + 3}
for lam in LAMS:
    for m, a in eta[lam].items():
        for stat, f in [('mean', lambda a: a.mean(0)), ('min', lambda a: a.min(0)), ('max', lambda a: a.max(0))]:
            columns[f'eta_lambda_{lam}_{m}_10seed_{stat}'] = f(a)
for k in ['pdf_ens', 'pdf_mem_avg', 'pdf_cont_avg', 'ext_ens', 'ext_mem_avg']:
    a = new[k]
    columns[f'new_hard_{k}_10seed_mean'] = a.mean(0)
    columns[f'new_hard_{k}_10seed_min'] = a.min(0)
    columns[f'new_hard_{k}_10seed_max'] = a.max(0)
with (OUT / 'tenseed_curves.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w_ = csv.writer(f); w_.writerow(columns); w_.writerows(zip(*columns.values()))

def make_fig(lang):
    if lang == 'zh':
        plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
        L = dict(ens='新 hard-PDF（物理网格）：集成预测，10种子',
                 mem='新 hard-PDF（物理网格）：成员指标平均，10种子',
                 eta_lab='η-learning {lab}：10种子平均',
                 xlabel='主动学习迭代轮数（初始3个标签）',
                 ypdf='PDF 误差 e：相同网格 KDE 口径', yext='极端区域加权 MSE：相同阈值与权重',
                 tpdf='PDF 误差：沿用原评估方法', text='极端区域预测误差',
                 sup='MMT / USLW：物理坐标 hard-PDF 与原 η-learning 十随机种子平均曲线',
                 note1='两侧均为种子101–110；彩色线为各 λ 的集成预测指标十种子平均（浅带为最小–最大）。',
                 note2='黑实线/灰虚线为新 hard-PDF 的集成预测/两成员指标平均（浅带为十种子范围）。PDF 连续积分结果另图展示。')
    else:
        L = dict(ens='New hard-PDF (physical mesh): ensemble, 10 seeds',
                 mem='New hard-PDF (physical mesh): member-average, 10 seeds',
                 eta_lab=r'$\eta$-learning {lab}: 10-seed mean',
                 xlabel='Active-learning round (starting from 3 labels)',
                 ypdf='PDF error e (same weighted grid-KDE estimator)',
                 yext='Extreme-region weighted MSE (same thresholds & weights)',
                 tpdf='PDF error: original estimator', text='Extreme-region prediction error',
                 sup='MMT / USLW: physical-mesh hard-PDF vs prior eta-learning, ten-seed mean curves',
                 note1='Seeds 101-110 on both sides; colored lines: 10-seed mean of the per-lambda ensemble metric (band: min-max).',
                 note2='Black/grey: new hard-PDF ensemble / member-average metric (band: 10-seed range). Continuous-law PDF metric shown separately.')
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6)); fig.subplots_adjust(top=.84, bottom=.27, left=.075, right=.98, wspace=.23)
    for j, metric in enumerate(['pdf_e', 'ext_mse']):
        ax = axs[j]
        for lam, label, color in zip(LAMS, LABELS, COLORS):
            a = eta[lam][metric]
            ax.plot(t, a.mean(0), color=color, lw=1.9, label=L['eta_lab'].format(lab=label), zorder=2)
            ax.fill_between(t, a.min(0), a.max(0), color=color, alpha=.09, lw=0, zorder=1)
        ens = new['pdf_ens'] if j == 0 else new['ext_ens']
        mem = new['pdf_mem_avg'] if j == 0 else new['ext_mem_avg']
        ax.plot(t, ens.mean(0), color='black', lw=2.8, label=L['ens'], zorder=5)
        ax.fill_between(t, ens.min(0), ens.max(0), color='black', alpha=.10, lw=0, zorder=4)
        ax.plot(t, mem.mean(0), color='#707070', lw=1.7, ls='--', label=L['mem'], zorder=4)
        ax.fill_between(t, mem.min(0), mem.max(0), color='#707070', alpha=.10, lw=0, zorder=3)
        ax.set(yscale='log', xlim=(0, 100), xlabel=L['xlabel'])
        ax.grid(alpha=.22, which='both')
        ax.set_ylabel(L['ypdf'] if j == 0 else L['yext'])
        ax.set_title(L['tpdf'] if j == 0 else L['text'])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, fontsize=9.5, frameon=False, columnspacing=1.7)
    fig.suptitle(L['sup'], fontsize=16, y=.97)
    fig.text(.5, .91, L['note1'], ha='center', fontsize=10)
    fig.text(.5, .035, L['note2'], ha='center', fontsize=9.5, color='#444444')
    fig.savefig(OUT / f'comparison_10seed_{lang}.png', dpi=210)
    fig.savefig(OUT / f'comparison_10seed_{lang}.pdf')
    plt.close(fig)

make_fig('zh'); make_fig('en')

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(10.5, 4.8), layout='constrained')
a = new['pdf_cont_avg']
ax.plot(t, a.mean(0), color='#0072B2', lw=2.4, label='连续模型积分：成员指标平均（10种子）')
ax.fill_between(t, a.min(0), a.max(0), color='#0072B2', alpha=.12)
ax.plot(t, new['pdf_mem_avg'].mean(0), color='#E69F00', lw=1.8, label='原网格 KDE：成员指标平均（10种子）')
ax.plot(t, new['pdf_ens'].mean(0), color='black', lw=1.5, ls='--', label='原网格 KDE：集成预测（10种子）')
ax.set(yscale='log', xlim=(0, 100), xlabel='迭代轮数', ylabel='PDF 误差 e')
ax.set_title('新物理网格 hard-PDF（种子101–110）：评估方式诊断（不与η曲线混用）')
ax.legend(fontsize=10); ax.grid(alpha=.2, which='both')
fig.savefig(OUT / 'new_hard_pdf_metric_diagnostic_10seed.png', dpi=200)
plt.close(fig)

report = {'acquisition': 'USLW', 'seeds': list(SEEDS),
          'new_model': 'physical x/6 features, 32 physical-uniform node intervals, 9506 Q_ref knots',
          'extreme_thresholds': {'lo': float(lo), 'hi': float(hi)},
          'round100': {}}
for lam in LAMS:
    report['round100'][f'eta_lambda_{lam}'] = {
        'pdf_grid_KDE_10seed_mean': float(eta[lam]['pdf_e'][:, -1].mean()),
        'extreme_mse_10seed_mean': float(eta[lam]['ext_mse'][:, -1].mean())}
report['round100']['new_hard_physical'] = {
    'pdf_grid_KDE_ensemble': float(new['pdf_ens'][:, -1].mean()),
    'pdf_grid_KDE_member_average': float(new['pdf_mem_avg'][:, -1].mean()),
    'pdf_continuous_member_average': float(new['pdf_cont_avg'][:, -1].mean()),
    'extreme_mse_ensemble': float(new['ext_ens'][:, -1].mean()),
    'extreme_mse_member_average': float(new['ext_mem_avg'][:, -1].mean())}
(OUT / 'tenseed_summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report['round100'], indent=2)); print('Saved', OUT)
