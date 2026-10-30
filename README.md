# η-learning:分位数正则与 PDF 硬约束的主动学习集成神经网络

本仓库研究**代理模型输出分布的精确性**:在 2D→1D 的主动学习框架中,对比三类方法——

1. **软约束(η-learning)**:损失 = MSE + λ·W1(模型输出分位数 vs 参考律分位数),λ ∈ {0, 1e-9, 1e-6, 1e-3, 1};
2. **PDF 硬约束(hard-PDF E-NN)**:输出经概率网格上的精确单调变换 ŷ = Q_ref(F(score(x))),输出分布**被构造性钉死**在参考律上;
3. **改进版 hard-PDF(物理网格)**:网格取物理坐标均匀剖分 + 精确高斯概率三角形权重,同时保住"约束精确性"与"极端区域分辨率"。

在**三个系统**上各做 **10 随机种子 × {USLW, US} 两种采集函数**的完整实验:

| 系统 | 真值映射 | 真值来源 | 外部依赖 |
|---|---|---|---|
| TOY_2D(五高斯混合) | 解析式 | — | 无 |
| 泊肃叶转捩(pois_transion_2D) | 9×64 网格双三次样条(θ 周期闭合) | `t_RetaoMean_576.csv` | 无 |
| MMT(Majda–McLaughlin–Tabak 波湍流, rank-1) | **原版 MATLAB ETDRK4 求解器直连**(512 点,dt=1e-3,积分至 t=20,y=maxₓ\|Re u\|) | 求解器逐点计算 | **本地 MATLAB** |

---

## 1. 仓库结构

```
eta-active-learning/
├── README.md                         # 本文件
├── requirements.txt
├── eta/                              # 上游框架代码(src / docs / notebooks;示例不依赖它,可独立阅读)
└── myproject/                        # 五个自包含示例工程(下载即可运行,互不依赖)
    ├── 10random_seeds_TOY_2D/            # TOY 系统,USLW 采集,10 种子
    ├── 10random_seeds_TOY_2D_US/         # TOY 系统,US 采集,10 种子
    ├── 10random_seeds_pois_transion_2D/  # 泊肃叶系统,两种采集(结果前缀 uslw_/us_)
    ├── 10random_seeds_MMT_2D/            # MMT 系统(η 阶梯 + 概率网格 hard-PDF),USLW 采集
    └── 10random_seeds_MMT_2D_PHYSICAL_MESH/  # MMT 改进版:物理网格 hard-PDF,两种采集
```

**每个示例工程内部(以 `10random_seeds_MMT_2D` 为例)**:

```
10random_seeds_MMT_2D/
├── _templates/{eta,hard}/   # 方法模板(problem.py 系统定义 / run.py 主循环 / acquisition.py
│                            #   采集函数 / model.py 硬约束网络 / mlp_repro.py 网络与训练复现)
│                            #   —— case 文件夹由 setup_cases.py 从模板整体复制生成
├── eta_s101 … eta_s110/     # 10 个种子组:η 阶梯的运行目录(每组内含 5 个 λ 的结果)
├── hard_s101 … hard_s110/   # 10 个种子组:hard-PDF 的运行目录
├── setup_cases.py           # 由模板生成 20 个 case 文件夹
├── plot_per_seed.py         # 每种子对比图(20 张)
├── plot_seed_summary.py     # 十种子汇总图(中/英)
├── summary_seeds_*.npz      # 历史运行的十种子汇总数据(保留作预期结果参考)
├── figures/                 # 历史运行的出图(保留作预期结果参考)
└── SYSTEM_SWAP_GUIDE.md / README_TODO.md   # 换系统指南与冒烟清单
```

> **发布说明**:为控制仓库体积,各 `results/` 内的逐轮原始数据已清除,超算批量脚本(`hpc/`)已删除——**全部代码均可本地复现**;`figures/` 与 `summary_seeds_*.npz` 是历史运行的交付产物,保留下来作为"复现后应对齐的结果"。`.gitignore` 已屏蔽未来的运行产物。

---

## 2. 环境要求

```bash
# Python ≥ 3.10(验证环境 3.12);CPU 即可,CUDA 可选(仅加速 λ>0 的 W1 项)
pip install -r requirements.txt
```

- **MMT 两个工程需要本地 MATLAB**(≥R2020b;已验证 2020b 与 R2026a 的 ETDRK4 求解器结果逐位一致,差 ≤ 2.8e-17)。求解器已随仓内置(`solver/` 目录),MATLAB 仅作为计算引擎被 `problem.py` 以 `matlab -batch` 方式调用:
  - Windows:自动探测 `C:\Program Files\MATLAB\R2026a\bin\matlab.exe`,或设环境变量 `MMATLAB_EXE` 指向你的 matlab 可执行文件;
  - Linux:`export MMATLAB_EXE=matlab`(取 PATH 中的 MATLAB)。
- TOY 与泊肃叶工程**不需要 MATLAB**。
- 绘图中文用 Microsoft YaHei(Windows 自带);Linux 下请将该字体装入系统或改用其他中文字体。

---

## 3. 快速开始(每个工程 3-5 分钟冒烟)

### 3.1 TOY(零外部依赖,最快上手)

```bash
cd myproject/10random_seeds_TOY_2D/_templates/eta
python problem.py                                  # 生成 data/evaluation.npz + reference_law.npz
python -B run.py --lambd 0 --iterations 1 --name s0            # 基线
python -B run.py --lambd 0.001 --iterations 1 --x-aux 20000 --threads 2 --name s1   # W1 软约束
python -B run.py --lambd 0 --iterations 1 --acq us --name su   # US 采集
cd ../hard && python -B run.py --iterations 0 --adam 50 --lbfgs 0 --threads 2 --name s   # hard 快验
```

核对:各 e 有限(1e-3~10 量级);`s1`/hard 的**成员级** e 显著更低(W1/硬约束生效)。

### 3.2 泊肃叶转捩

同 3.1 的五条命令(在 `10random_seeds_pois_transion_2D/_templates/` 下);真值来自仓内 CSV,无需任何外部数据。

### 3.3 MMT(需要 MATLAB)

```bash
cd myproject/10random_seeds_MMT_2D/_templates/eta
python problem.py        # 仅装配评估集/参考律(数据文件已随仓携带,无需求解)
python -B run.py --lambd 0 --iterations 1 --name s0    # 首轮会调用 MATLAB 算 3+1 个真值
```

改进版物理网格 hard-PDF(`10random_seeds_MMT_2D_PHYSICAL_MESH`)单种子全程运行:

```bash
cd myproject/10random_seeds_MMT_2D_PHYSICAL_MESH
python run_local.py     # 自带监督脚本:种子101 / USLW / 100 轮 / 断点续跑 / 自动溯源快照
# 或直接:
cd hard_s101 && python run.py --iterations 100 --threads 8 --device cpu --acq uslw \
    --lhs-seed 101 --seed-base 1 --mesh 32 --adam 1800 --lbfgs 450 --name uslw_seed101_physical
```

---

## 4. 完整复现十种子实验(全部本地)

种子组 i=0..9:LHS 种子 = 101+i,成员种子基 = 1000·i+1。以下以 MMT 的 USLW 轮为例(其他工程把目录名换掉即可;US 轮把 `--acq` 与结果名前缀 `uslw` 换成 `us`):

```bash
cd myproject/10random_seeds_MMT_2D/_templates/eta && python problem.py   # 确认数据就绪
cd .. && python setup_cases.py                                            # 生成/检查 20 个 case

for i in $(seq 0 9); do
  S=$((101+i)); B=$((1000*i+1))
  ( cd eta_s$S && python -B run.py --lambd 0 --acq uslw --device cpu \
        --lhs-seed $S --seed-base $B --name uslw_seed$S --resume )
  for L in 1e-9 1e-6 0.001 1; do
    ( cd eta_s$S && python -B run.py --lambd $L --acq uslw --device cuda --threads 4 \
          --lhs-seed $S --seed-base $B --name uslw_lam${L}_seed$S --resume )   # 无 GPU 用 --device cpu --threads 2
  done
  ( cd hard_s$S && python -B run.py --acq uslw --threads 8 \
        --lhs-seed $S --seed-base $B --name uslw_seed$S --resume )
done
```

出图与汇总:

```bash
cd myproject/10random_seeds_MMT_2D
python plot_per_seed.py uslw             # 20 张每种子对比图
python plot_seed_summary.py uslw         # 中英十种子汇总图
```

物理网格工程复现(十种子 × 两种采集,参照 `run_local.py` 的参数,把 `--lhs-seed/--seed-base/--acq/--name` 按上表循环),随后:

```bash
cd myproject/10random_seeds_MMT_2D_PHYSICAL_MESH
python summarize_run.py                        # 单种子运行总结
python plot_vs_eta_tenseed_uslw.py             # 与 η 阶梯十种子均值的对比(种子101 版)
python plot_tenseed_physical_vs_eta.py         # 十种子版对比图(读取兄弟工程 summary_seeds_uslw.npz)
```

**参考耗时(单工况 100 轮,CPU)**:TOY 数分钟;泊肃叶 ~20 分钟;MMT η(λ=0,CPU)~35-50 分钟(每轮含一次 MATLAB 求解)、λ>0(GPU)~40 分钟;hard-PDF ~90 分钟;物理网格 hard-PDF ~70 分钟(本地,含 MATLAB)。十种子全矩阵为 60 工况,建议隔夜跑或自行并行。

---

## 5. 发布前冒烟验证(方法与结果)

发布本仓库前,对五个示例工程各做了一次**"原工程 vs 发布副本"同机同命令对拍**:在同一台机器、同一 Python 环境下,分别在原始工程与发布副本中运行同一条冒烟命令,然后逐字段深度比对两侧的 `history.json`(**剔除墙钟时间类字段后要求全等**)。五项全部 PASS(字段差异数=0),证明发布副本与原始工程在数值上完全一致。

**运行方式**(每条命令在原工程与副本各跑一遍,`<ROOT>` 分别为原始工程根与 `eta-active-learning/`):

```bash
# 1) TOY_2D(USLW 采集)
cd <ROOT>/myproject/10random_seeds_TOY_2D/_templates/eta
python -B run.py --lambd 0 --iterations 1 --name smoke
# 2) TOY_2D_US(US 采集)
cd <ROOT>/myproject/10random_seeds_TOY_2D_US/_templates/eta
python -B run.py --lambd 0 --iterations 1 --acq us --name smoke
# 3) 泊肃叶转捩
cd <ROOT>/myproject/10random_seeds_pois_transion_2D/_templates/eta
python -B run.py --lambd 0 --iterations 1 --name smoke
# 4) MMT_2D(η 基线;每轮含一次 MATLAB 求解器调用)
cd <ROOT>/myproject/10random_seeds_MMT_2D/_templates/eta
python -B run.py --lambd 0 --iterations 1 --name smoke
# 5) 物理网格 hard-PDF(种子 101,round 0)
cd <ROOT>/myproject/10random_seeds_MMT_2D_PHYSICAL_MESH/hard_s101
python -B run.py --iterations 0 --threads 8 --device cpu --acq uslw \
    --lhs-seed 101 --seed-base 1 --mesh 32 --adam 1800 --lbfgs 450 --name smoke
```

**运行结果**(两树数值逐位一致;下表为末轮关键指标,单侧读数):

| 工程 | 冒烟命令要点 | 末轮关键指标(两树全等) | history 深度比对 |
|---|---|---|---|
| TOY_2D | λ=0,USLW,1 次迭代 | RMSE=0.197558,e=0.317270,top1=0.252386 | **PASS(0 差异)** |
| TOY_2D_US | λ=0,US,1 次迭代 | RMSE=0.195117,e=0.344974,top1=0.256394 | **PASS(0 差异)** |
| 泊肃叶 | λ=0,USLW,1 次迭代 | RMSE=0.059618,e=5.796243(4 点冷启动的正常量级),top1=0.148310 | **PASS(0 差异)** |
| MMT_2D | λ=0,USLW,1 次迭代(含 MATLAB) | RMSE=0.006531,e=0.065277,top1=0.017082 | **PASS(0 差异)** |
| 物理网格 hard | 种子 101,round 0(含 MATLAB) | RMSE=0.007224,e_cont 成员=4.2e-4/4.4e-4,极端 MSE=5.4e-5/9.8e-5 | **PASS(0 差异)** |

> 比对方法:加载两侧 `history.json`,递归比较全部字段(浮点按精确相等),仅跳过 `seconds`/`elapsed_this_session`/`training` 等墙钟与日志字段。冒烟产生的 `results/smoke`、`oracle_tmp` 已在两侧清理。任何人 clone 本仓库后执行上述命令,应得到与表中一致的数值(同机同版本 Python/PyTorch/MATLAB 前提下;跨环境允许浮点级漂移)。

---

## 6. 预期结果速览(对齐 `figures/` 与 `summary_*.npz`)

| 系统 | 关键结论 |
|---|---|
| TOY_2D | λ=1 两种采集下 10/10 种子失败;hard-PDF 成员级 PDF 误差全场最优(~0.004)且对采集函数不敏感;软约束系方法 USLW 优于 US |
| 泊肃叶 | λ=1 同样 10/10 失败;hard 成员级 e 全场最优(贴理论地板带)且采集免疫;λ=1e-3 仅在 US 轮有条件性补偿 |
| MMT(本仓 `10random_seeds_MMT_2D` 的 `figures/diag_*.png` 有完整机理图) | 映射光滑易拟合:λ=1 十种子**温和退化不崩溃**(e≈0.07,仅比基线差一个量级);λ=1e-3 拿极端 MSE 单项冠军(2.6e-7);**概率网格 hard-PDF 在极端区失效(300 倍差距)**——机理:输入律 PIT 网格把分辨率全花在中心概率质量(最外格宽 4.14,占域宽 34%),且采集在角区自激(75% 查询烧在 0.1% 概率质量的角区) |
| MMT 物理网格改进版 | 两个病灶同时根治:极端 MSE 2.6-7.7e-7(与最优 λ 同级)、连续律误差 5e-4 十种子恒稳;**对采集函数完全免疫**(US/USLW 十种子逐指标同量级,极端 MSE 比值 0.79) |

十种子对比图(物理网格 vs η 阶梯,中英双版)见 `10random_seeds_MMT_2D_PHYSICAL_MESH/figures/tenseed_uslw_comparison/`。

---

## 7. 常见问题

- **MATLAB 相关**:报 `MATLAB executable not found` → 设置 `MMATLAB_EXE`;求解器与网格真值在 MATLAB 2020b/Linux 与 R2026a/Windows 间已验证逐位一致;每次查询约 +15-25 秒(MATLAB 启动占大头)。
- **确定性**:代码启用 `torch.use_deterministic_algorithms(True)` 与显式种子(成员种子 = 基+2k/基+2k+1,逐轮冷重启);同机同版本结果可复现,跨机/跨 torch 版本允许数值级漂移。
- **λ>0 必须用 GPU?** 不必,CPU 可跑(慢 ~5-8 倍);`--device cuda --threads 4` 为推荐配置。
- **`setup_cases.py` 报 refusing to overwrite**:case 文件夹已存在(仓内自带);如需从模板重建,先自行移除旧 case 目录。
- **想换一个新系统?** 按 `10random_seeds_MMT_2D/SYSTEM_SWAP_GUIDE.md` 的 §0 契约表:所有约定(输入律、候选分布、网络输入坐标、评估权重、hard 网格坐标)都定义在 `problem.py`,`run.py/acquisition.py` 零改动。

## 8. 引用

如使用本代码,请引用本仓库。方法细节与三系统完整结论见 `figures/` 各图及各工程 `summary_*.npz`。
