"""Visual explanations using cached real MMT data; no training or PDE solves."""
from pathlib import Path
import json,numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as MplPath
R=Path(__file__).resolve().parent;D=R/'diagnostics/truth_pdf_integration_vs_kde';OUT=D/'visual_explanation';OUT.mkdir(exist_ok=True)
raw=np.load(D/'physical_continuous_unsmoothed.npz');sm=np.load(D/'truth_pdf_curves.npz');meta=json.loads((D/'analysis.json').read_text());peak=json.loads((D/'left_density_peak_attribution.json').read_text());h=meta['kde_bandwidth']
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':12,'axes.titlesize':14,'pdf.fonttype':42})
BLUE='#0072B2';ORANGE='#E69F00';GREEN='#009E73';RED='#D55E00';GRAY='#6b7280'
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=185);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
# 1. Map an actual triangle to a narrow interval, then locate the PDF peak.
fig,axs=plt.subplots(1,3,figsize=(15.5,5.6));fig.subplots_adjust(top=.80,bottom=.20,left=.065,right=.98,wspace=.34)
cell=peak['top_triangles'][0];verts=np.array(cell['physical_vertices']);ys=np.array(cell['output_vertices']);rng=np.random.RandomState(38);bary=rng.dirichlet([1,1,1],80);pts=bary@verts;vals=bary@ys
ax=axs[0];ax.add_patch(Polygon(verts,closed=True,facecolor='#e6f2fa',edgecolor=BLUE,lw=2));ax.scatter(pts[:,0],pts[:,1],c=vals,cmap='viridis',s=14);ax.scatter(verts[:,0],verts[:,1],color=BLUE,s=35)
for j,(xy,yy) in enumerate(zip(verts,ys)):
 offsets=[(8,10),(-9,-24),(8,9)];ax.annotate(f'y={yy:.6f}',xy,xytext=offsets[j],textcoords='offset points',ha='right' if j==1 else 'left',fontsize=9.5)
ax.set(xlabel='输入 x1',ylabel='输入 x2',title='① 一个真实网格三角形',xlim=(-1.32,-1.09),ylim=(-.225,-.015));ax.grid(alpha=.2)
ax=axs[1];jitter=rng.uniform(-.18,.18,len(vals));ax.scatter(vals,jitter,c=vals,cmap='viridis',s=20);ax.axvspan(ys.min(),ys.max(),color=BLUE,alpha=.08);ax.set(xlim=(.0009,.0017),ylim=(-.5,.5),yticks=[],xlabel='这些输入对应的输出 y',title='② 一片输入挤进很窄的输出区间');ax.annotate('',xy=(ys.min(),-.30),xytext=(ys.max(),-.30),arrowprops={'arrowstyle':'<->','color':BLUE,'lw':1.5});ax.text(np.mean([ys.min(),ys.max()]),-.39,f'区间宽度仅 {np.ptp(ys):.6f}',ha='center',fontsize=10);ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(axis='x',alpha=.2)
ax=axs[2];ax.plot(raw['y'],raw['pdf'],color=BLUE,lw=2);ax.scatter([peak['peak_output']],[peak['peak_pdf']],color=RED,s=45,zorder=4);ax.annotate('很多三角形的贡献相加\n形成密度峰，而非响应尖峰',xy=(peak['peak_output'],peak['peak_pdf']),xytext=(.003,78),arrowprops={'arrowstyle':'->','color':RED},fontsize=10);ax.set(xlim=(0,.012),ylim=(0,105),xlabel='输出 y',ylabel='概率密度',title='③ 最终 PDF 在小输出处抬高');ax.grid(alpha=.2)
fig.suptitle('左侧 PDF 尖峰：许多输入映射到相近的小输出',fontsize=17,y=.96)
fig.text(.5,.885,'左图三角形来自真实100×100网格；内部散点按线性插值示意着色，并非新增MATLAB求解。',ha='center',fontsize=10)
fig.text(.5,.06,'这是“输出出现得有多集中”的密度峰。其尖锐细节受三角插值影响：反转对角线后峰位由0.001295变为0.001422。',ha='center',fontsize=10)
save(fig,'01_why_density_peak')
# 2. Show one kernel, then the actual PDF under increasingly wide convolution.
step=sm['y'][1]-sm['y'][0];raw_uniform=np.interp(sm['y'],raw['y'],raw['pdf'],left=0,right=0);total=float(raw_uniform.sum()*step);raw_uniform/=total
fig,axs=plt.subplots(1,2,figsize=(13.5,5.6));fig.subplots_adjust(top=.82,bottom=.19,left=.075,right=.98,wspace=.25)
ax=axs[0];center=peak['peak_output'];x=np.linspace(center-3.5*h,center+3.5*h,1500);kernel=np.exp(-.5*((x-center)/h)**2)/(np.sqrt(2*np.pi)*h);ax.plot(x,kernel,color=ORANGE,lw=2.5);ax.fill_between(x,kernel,color=ORANGE,alpha=.16);ax.annotate('',xy=(center,162),xytext=(center,0),arrowprops={'arrowstyle':'->','color':BLUE,'lw':2.5});ax.text(center+.00025,155,'原来集中在一个输出值\n（示意，不是有限高度的PDF）',fontsize=10,color=BLUE);mask=(x>=center-h)&(x<=center+h);ax.fill_between(x[mask],kernel[mask],color=ORANGE,alpha=.32);ax.annotate('',xy=(center-h,15),xytext=(center+h,15),arrowprops={'arrowstyle':'<->','lw':1.4});ax.text(center,24,f'左右各一个 h={h:.6f}',ha='center',fontsize=10);ax.axvline(0,color=GRAY,ls=':');ax.set(xlabel='输出 y',ylabel='高斯核密度（面积=1）',title='① 平滑：把集中的概率摊成钟形',ylim=(0,180));ax.grid(alpha=.2)
ax=axs[1];ax.plot(raw['y'],raw['pdf'],color=BLUE,lw=1.4,label='原始连续积分 PDF（未平滑）')
for ratio,color,label in [(.25,'#56B4E9','小窗口：h/4'),(.5,GREEN,'中窗口：h/2'),(1,ORANGE,'当前窗口：h')]:
 density=gaussian_filter1d(raw_uniform,ratio*h/step,mode='constant',truncate=12.)
 ax.plot(sm['y'],density,color=color,lw=2,label=label)
ax.axvline(0,color=GRAY,ls=':');ax.set(xlim=(-.005,.025),ylim=(0,105),xlabel='输出 y',ylabel='概率密度',title='② 真实数据：窗口越宽，峰越矮越宽');ax.legend(fontsize=9.5);ax.grid(alpha=.2)
fig.suptitle('“平滑”发生在输出 y 轴上：不改变原始求解值，只改变分布表示',fontsize=16,y=.96)
fig.text(.5,.07,'高斯平滑会把部分概率延伸到负输出。此图右侧使用实际连续积分密度卷积；概率面积仅作微小数值归一化。',ha='center',fontsize=10)
save(fig,'02_how_gaussian_smoothing_works')
# 3. Exact data flow: Q_ref has no Gaussian smoothing; PDF evaluation does.
fig,ax=plt.subplots(figsize=(16,8.6));ax.set(xlim=(0,16),ylim=(0,8.6));ax.axis('off')
def box(x,y,w,height,text,face,edge):
 ax.add_patch(FancyBboxPatch((x,y),w,height,boxstyle='round,pad=0.10,rounding_size=0.10',facecolor=face,edgecolor=edge,lw=1.6));ax.text(x+w/2,y+height/2,text,ha='center',va='center',fontsize=11.5)
def arrow(a,b,color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=color,lw=1.7))
ax.text(8,8.2,'Hard-PDF 到底在哪里用了平滑？',ha='center',fontsize=20,weight='bold')
ax.text(.35,7.55,'A．参考分布构造：不做高斯平滑',fontsize=13,color=BLUE,weight='bold')
box(.55,6.35,3.4,.8,'10000个真实输出\n＋高斯输入概率权重','#e6f2fa',BLUE);box(5.0,6.35,3.6,.8,'计算加权分位数\n节点之间线性插值','#e6f2fa',BLUE);box(10.0,6.35,4.6,.8,'参考分位数 Q_ref\n9506节点；没有高斯平滑','#e6f2fa',BLUE);arrow((4.05,6.75),(4.88,6.75));arrow((8.7,6.75),(9.88,6.75))
ax.text(.35,5.65,'B．网络预测与训练：不做高斯平滑',fontsize=13,color=GREEN,weight='bold')
for x0,w0,text in [(.55,1.5,'输入 x'),(2.75,2.0,'MLP 评分 s'),(5.55,2.25,'评分 CDF：F_s'),(8.6,2.25,'映射 Q_ref'),(11.75,3.1,'该成员最终预测 y')]:box(x0,4.35,w0,.8,text,'#e7f4ed',GREEN)
for a0,b0 in [((2.15,4.75),(2.63,4.75)),((4.85,4.75),(5.43,4.75)),((7.90,4.75),(8.48,4.75)),((10.95,4.75),(11.63,4.75))]:arrow(a0,b0,GREEN)
arrow((11.5,6.25),(9.73,5.25),BLUE)
ax.text(8,3.90,'训练损失：在已采样输入上计算预测值与真值的 MSE；平滑不参与这一步。',ha='center',fontsize=10.5,color=GREEN)
ax.plot([.3,15.65],[3.42,3.42],ls='--',color=GRAY,lw=1)
ax.text(.35,2.98,'C．PDF误差评估：这里才做高斯平滑（两个成员分别执行）',fontsize=13,color=RED,weight='bold')
box(.55,1.65,3.2,.9,'对该成员连续模型积分\n得到输出概率分布','#fff2df',ORANGE);box(4.6,1.65,3.1,.9,'高斯平滑\n带宽来自评估规则','#fff2df',ORANGE);box(8.55,1.65,2.45,.9,'平滑后的\n模型 PDF','#fff2df',ORANGE);box(11.9,1.65,3.0,.9,'与真实值 KDE 比较\n得到日志中的 e','#fff2df',ORANGE)
for a0,b0 in [((3.85,2.1),(4.48,2.1)),((7.8,2.1),(8.43,2.1)),((11.1,2.1),(11.78,2.1))]:arrow(a0,b0,ORANGE)
box(11.9,.40,3.0,.65,'真实输出的加权 KDE','#f2f2f2',GRAY);arrow((13.4,1.15),(13.4,1.53),GRAY)
ax.text(.55,.85,'因此：此前约 0.00045 的 e，是“连续积分＋高斯平滑”后的误差。',fontsize=12,color=RED,weight='bold')
ax.text(.55,.35,'图中的原始真实连续积分 PDF，是另外的诊断对象，没有直接用来构造 Q_ref。',fontsize=10.5,color=GRAY)
save(fig,'03_hardpdf_training_vs_evaluation')
(OUT/'plot_metadata.json').write_text(json.dumps({'source':'cached truth-grid PDF and attribution, no retraining','h':h,'uniform_pdf_mass_before_visual_normalization':total,'smoothing_ratios':[.25,.5,1.],'hard_reference':'weighted-quantile interpolation without Gaussian smoothing','reported_e':'each-member continuous law integration then Gaussian smoothing, compared with truth KDE'},indent=2),encoding='utf8')
print('Saved three explanatory figures to',OUT)
