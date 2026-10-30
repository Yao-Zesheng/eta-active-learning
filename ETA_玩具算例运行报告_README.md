# eta（Extreme Event Aware η-Learning）仓库下载与玩具算例运行报告

> 报告日期：2026-10-22
> 本地路径：`<仓库根目录>/eta`
> 仓库版本：v1.1.0（git tag），GitHub: <https://github.com/kai-ovo/eta>

---

## 目录

1. [结论摘要（TL;DR）](#1-结论摘要tldr)
2. [项目背景与核心方法](#2-项目背景与核心方法)
3. [仓库结构](#3-仓库结构)
4. [下载过程与 Windows 特殊问题处理](#4-下载过程与-windows-特殊问题处理)
5. [本地环境配置记录](#5-本地环境配置记录)
6. [玩具算例详解](#6-玩具算例详解)
7. [实测运行结果](#7-实测运行结果)
8. [如何自己运行（手把手指南）](#8-如何自己运行手把手指南)
9. [常见问题排查（FAQ）](#9-常见问题排查faq)
10. [进阶：通往完整 ERA5-Land 实验](#10-进阶通往完整-era5-land-实验)

---

## 1. 结论摘要（TL;DR）

- **已成功**将 `kai-ovo/eta` 仓库克隆到本地 `<仓库根目录>/eta`（v1.1.0，37 MB）。
- **已成功**从 Zenodo 下载玩具数据集 `data/toy.pth`（26.1 MB），并通过官方 SHA256 校验。
- **已跑通全部 3 个玩具入口**：
  1. `examples/smoke_test_toy2d.py` 合成数据冒烟测试（秒级，验证安装）；
  2. `experiments.toy_2d_to_1d`（2D→1D 标量映射玩具实验，完整默认轮数）；
  3. `experiments.toy_2d_to_2d`（2D→2D 状态映射玩具实验，完整默认轮数）。
- **核心结果**：在极端事件尾部指标 tail-W1（尾部分位数 Wasserstein-1 距离）上，η-学习模型远优于纯 MSE 基线——2D→1D 算例改进约 **15.5 倍**（0.0268→0.0017），2D→2D 算例改进约 **35 倍**（0.0808→0.0023），复现了论文的核心主张——**在稀缺数据下，η 正则化能让模型学到正确的极端事件分布尾部**。每次完整实验约 7 分钟（RTX 5060 Laptop）。
- **Windows 三个坑及解法**（详见第 4、9 节）：notebook 文件名含非法字符 `>`；KDEpy 必须用 1.1.8（1.1.12 有断言 bug）；系统 `python` 是 Microsoft Store 占位符。

---

## 2. 项目背景与核心方法

### 2.1 论文与作者

本仓库是论文 ***Extreme Event Aware (η-) Learning***（Chang & Sapsis，MIT）的配套代码。论文研究的是**极端事件数据稀缺条件下的监督学习**问题。

### 2.2 核心思想（用人话解释）

许多物理系统（如降水、海洋波浪）的输出分布是重尾的：极端值极少出现，因此训练集里几乎没有极端样本。普通监督学习（ERM / MSE 损失）在这种"稀缺配对数据"下会系统性地**低估尾部**——模型学会了平均值，但学不会"极端事件有多极端、多频繁"。

η-学习的做法分两步（两阶段课程）：

1. **ERM/MSE 预训练**：先用常规 MSE 损失在稀缺配对数据上训练一个映射模型（输入状态 → 输出标量可观测量/状态场），得到一个"会预测均值"的基线；
2. **η 续训（continuation）**：从预训练权重出发，在损失函数中加入一个 **Wasserstein 正则项**——把模型输出分布的分位数（尤其是尾部加密到 1e-7 量级的极端分位数）对齐到参考分布（由更充足的参考样本或先验如 GEVD 给出）的对应分位数。总损失形如：

   `L = MSE(配对数据) + λ · W1_quantile(模型输出分位数, 参考分位数)`

   其中 W1 分位数损失按论文正文 Eq. (13) 实现（`src/metric.py::wloss`，p=1），每个 `omega` 步刷新一次实现目标分位数的样本索引（SI Algorithms 1-2，`src/train_utils.py::train_eta_xy / train_eta_xuy`）。

这样，即使没有极端事件的配对标签，只要有一个"极端事件分布长什么样"的参考（分位数/先验），模型就能把尾部学对。玩具算例就是用最小成本验证这条逻辑的实验。

### 2.3 玩具问题的设置

玩具数据（`data/toy.pth`，从 Zenodo 下载）：

- 输入 `X ∈ R²`：二维各向同性高斯（均值 0，方差 10）；
- 通过一个映射得到输出 `Y`（2D→1D 是标量；2D→2D 是状态 `u=(u₁,u₂)`，`u₂` 由输入的 Fourier 模式构造，可观测量 `g(u)=2|u₁|+0.5|u₂|`）；
- 输出分布是重尾的；
- **训练集只有 100 个配对样本**（`x_train`/`y_train`）——极端样本基本不在里面，这就是"稀缺"。

评价指标 `tail_w1`：取一组极度偏向尾部的分位数网格（从 0 到 0.9999999，越接近 1 越密；见 `experiments/common.py::toy_scalar_quantiles`），计算模型输出样本与真值样本在这些分位数上的 W1 距离（`src/revision_utils.py::tail_w1_quantile`）。**该值越小，极端事件分布学得越准。**

---

## 3. 仓库结构

```text
eta/
├── README.md                  # 官方英文说明（环境、运行顺序、硬件耗时表）
├── requirements.txt           # pip 依赖锁定（Python 3.11.8 + torch 2.2.0+cu118）
├── environment.yml            # conda 环境完整导出（更权威的环境记录）
├── src/                       # 核心源码
│   ├── models.py              # FCNN、SRCNN、扩散组件、U-Net、Flow Matching 包装
│   ├── train_utils.py         # MSE 预训练 + η-学习训练例程（玩具实验）
│   ├── metric.py              # 分位数/Wasserstein 损失（wloss）
│   ├── test_utils.py          # 标量与状态场输出的评估包装
│   ├── utils.py               # 设备、随机种子、激活函数、Result 容器、分布工具
│   ├── kde.py                 # 核密度估计（KDEpy FFTKDE 封装）
│   ├── plot_utils.py          # 绘图（依赖 seaborn/cartopy）
│   └── revision_utils.py      # 空间度量、先验敏感性、耗时测量等辅助
├── experiments/               # 命令行入口（与 notebook 一一对应）
│   ├── common.py              # CLI 公共参数、数据加载、预训练包装
│   ├── toy_2d_to_1d.py        # ★ 玩具算例 1：2D→1D 标量映射
│   ├── toy_2d_to_2d.py        # ★ 玩具算例 2：2D→2D 状态映射
│   ├── era5land.py            # ERA5-Land 降水降尺度（需外部数据/检查点）
│   ├── era5land_evd.py        # GEVD 先验/错误指定尾部敏感性（需外部数据）
│   ├── era5land_dgm.py        # Flow Matching 训练与采样（需外部数据）
│   ├── era5land_dgm_plot.py   # FM PDF 图 + η 校正样本（需外部数据）
│   └── era5land_computational_overhead.py  # 运行时/显存测量（需外部数据）
├── notebooks/                 # 论文交互式实验记录（Jupyter）
│   ├── toy--2D-to-1D.ipynb    # ★ 玩具 2D→1D（本地重命名，原名 toy--2D->1D.ipynb）
│   ├── toy--2D-to-2D.ipynb    # ★ 玩具 2D→2D（本地重命名，原名 toy--2D->2D.ipynb）
│   └── ERA5Land*.ipynb        # 5 个 ERA5-Land 相关 notebook（需外部数据）
├── examples/
│   └── smoke_test_toy2d.py    # ★ CPU 合成数据冒烟测试（秒级，无需任何下载）
├── scripts/
│   ├── check_external_assets.py  # 检查本地数据/检查点布局（不下载）
│   └── make_smoke_dataset.py     # 从完整日最大值文件生成冒烟测试子集
├── docs/
│   ├── command_line_experiments.md  # 命令行实验完整手册
│   ├── data_and_models.md           # 外部数据与预训练模型布局说明
│   └── equation_map.md              # 论文公式 ↔ 代码位置对照表
├── data/                      # 数据目录（toy.pth 需自行从 Zenodo 下载）
├── checkpoints/               # 检查点目录（需自行从 Zenodo 下载）
└── result_log/                # 结果日志（本次运行的输出在这里）
```

带 ★ 的就是**无需任何大型外部数据即可运行**的玩具入口。

---

## 4. 下载过程与 Windows 特殊问题处理

### 4.1 克隆仓库

```bash
cd "<仓库根目录>"
git clone https://github.com/kai-ovo/eta
```

克隆本身成功，但 **checkout 失败**，报错：

```text
error: invalid path 'notebooks/toy--2D->1D.ipynb'
error: invalid path 'notebooks/toy--2D->2D.ipynb'
```

原因：两个 notebook 原文件名含 `>` 字符（`toy--2D->1D.ipynb`），`>` 是 Windows 文件系统的保留字符，无法落盘。这是 Windows 上克隆本仓库的**已知必然问题**。

### 4.2 解决办法（已执行）

用 git sparse-checkout 排除这两个文件完成检出，再从 git 对象库把内容导出为合法文件名（`>` → `to`）：

```bash
cd eta
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!notebooks/toy--2D->1D.ipynb' '!notebooks/toy--2D->2D.ipynb'
git checkout HEAD -- .
git show 'HEAD:notebooks/toy--2D->1D.ipynb' > notebooks/toy--2D-to-1D.ipynb
git show 'HEAD:notebooks/toy--2D->2D.ipynb' > notebooks/toy--2D-to-2D.ipynb
```

处理后本地得到完整仓库，两个 notebook 内容与原版逐字节一致，仅文件名可被 Windows 接受。**在 Jupyter 里使用时打开重命名后的文件即可，内容完全相同。**

### 4.3 下载玩具数据集（Zenodo）

GitHub 仓库不含大数据文件。数据在 Zenodo：[10.5281/zenodo.21635446](https://doi.org/10.5281/zenodo.21635446)（CC-BY-4.0）。

玩具算例只需要其中的 `toy.pth`（26.1 MB）：

```bash
curl -L -o data/toy.pth "https://zenodo.org/api/records/21635446/files/toy.pth/content"
curl -sL -o SHA256SUMS-data.txt "https://zenodo.org/api/records/21635446/files/SHA256SUMS-data.txt/content"
```

已做 SHA256 校验，与官方发布一致：

```text
a843c5a5a4a86adbcecb61f5ca23eca595047d910db7b10a3b2bb78a3bb59977  toy.pth  ✓ 匹配
```

`toy.pth` 内容为张量字典：`X_all`（全量输入样本）、`Y_all`（全量输出样本）、`in_grid`（输入评估网格）、`out_grid`（输出网格）、`x_train`/`y_train`（100 个稀缺训练样本）。

---

## 5. 本地环境配置记录

### 5.1 使用的环境

本机没有按仓库锁定的 Python 3.11 + torch 2.2.0 新建环境，而是**复用了现成的 conda 环境 `pytorch`**（Python 3.12.14 + torch 2.14.0+cu130，CUDA 可用，GPU 为 RTX 5060 Laptop 8GB）。实测玩具算例在新版栈上完全兼容（仓库代码自带 torch 2.6+ `weights_only` 兼容处理）。

### 5.2 需要补装的包及原因

| 包 | 用途 | 备注 |
|---|---|---|
| `KDEpy==1.1.8` | `src/kde.py` 的 FFT 核密度估计 | ⚠️ **必须 1.1.8**。装 1.1.12 会在 KDE 网格等距断言处崩溃（`AssertionError` in `linbin_cython`），降回 1.1.8 后正常。降级会连带把 numpy 降到 1.26.4，属正常 |
| `flow_matching==1.0.10` | `src/models.py` 顶部导入 `ModelWrapper`（即使玩具算例不用 FM 也必须装） | |
| `torchdiffeq` | FM 采样 ODE 求解 | 顺手装上，ERA5 实验需要 |
| `seaborn` | `src/plot_utils.py`（被 `test_utils.py` 星号导入连带入链） | |
| `cartopy==0.26.0` | 同上（plot_utils 顶部导入；新版有 Windows wheel，pip 直装即可） | |

一条命令（在 `pytorch` 环境中）：

```bash
python -m pip install "KDEpy==1.1.8" flow_matching torchdiffeq seaborn cartopy
```

若想完全按官方锁定复现（Python 3.11.8 + torch 2.2.0+cu118），可参考仓库 `README.md` 与 `environment.yml` 新建 conda 环境；玩具算例对版本并不敏感。

---

## 6. 玩具算例详解

### 6.1 三个可运行入口的关系

| 入口 | 耗时 | 需要数据 | 作用 |
|---|---|---|---|
| `examples/smoke_test_toy2d.py` | 数秒 | 无（纯合成数组） | 验证导入、MSE 与分位数-W1 损失接线正确 |
| `python -m experiments.toy_2d_to_1d` | 约 10–15 min（本机 GPU） | `data/toy.pth` | 论文玩具实验 1：标量映射 |
| `python -m experiments.toy_2d_to_2d` | 约 10 min（本机 GPU） | `data/toy.pth` | 论文玩具实验 2：状态映射 |

### 6.2 2D→1D 标量算例（`experiments/toy_2d_to_1d`）

对应 notebook `toy--2D-to-1D.ipynb`。流程：

1. **MSE 基线**：FCNN（隐藏层 [256,256,256]，pSILU 激活，输入 2 维，输出 1 维，输出非负约束，Xavier 初始化），在 100 个训练样本上以 MSE 训练 `--mse-epochs`（默认 3000）轮；
2. **η 续训**：先用二维高斯随机场做 GRF 预训练（`--pretrain-epochs`，默认 1000）获得初始权重，再以 `MSE + λ·W1_quantile` 续训 `--eta-epochs`（默认 3000）轮，默认 `λ=1`，`omega=100`（每 100 步刷新尾部索引）；分位数网格在尾部加密至 1e-7 分位；
3. **评估**：在输入网格上评估两个模型，输出各自 `final_train_mse` 与 `tail_w1`，可选 `--save-npz` 保存网格/样本/PDF 数组，`--lambda-sweep` 做 λ 敏感性扫描。

### 6.3 2D→2D 状态算例（`experiments/toy_2d_to_2d`）

对应 notebook `toy--2D-to-2D.ipynb`。把输出扩为状态 `u=(u₁,u₂)`（`u₂` 为输入的 Fourier 模式，`g(u)=2|u₁|+0.5|u₂|` 为标量可观测量），网络输出 2 维，η 训练用 `train_eta_xuy`（对可观测量 g 做分位数对齐）。默认 η 种子 28。流程与指标同上。

### 6.4 notebook 与命令行的关系

`experiments/` 下的命令行脚本是 notebook 的忠实复刻（同样的默认超参数），适合批处理与服务器运行；notebook 额外包含交互式绘图与 revision 诊断（如 η 估计实现间的空间不确定性表、超越概率图等），这些诊断默认只在 notebook 内展示、不落盘。

---

## 7. 实测运行结果

运行环境：RTX 5060 Laptop (8 GB)，conda `pytorch` 环境（Python 3.12.14，torch 2.14.0+cu130）。

### 7.1 冒烟测试 `examples/smoke_test_toy2d.py`

```text
smoke_supervised_mse=0.433210
smoke_quantile_w1=0.712410
smoke_test_ok
```

数秒完成，安装验证通过。

### 7.2 玩具算例 1：2D→1D（完整默认配置）

命令：`python -m experiments.toy_2d_to_1d --save-npz result_log/toy2d1d_full.npz`

| 模型 | seed | λ | final_train_mse | **tail_w1（越小越好）** |
|---|---|---|---|---|
| MSE 基线 | 1 | — | 2.599e-05 | 0.02677 |
| **η-学习** | 25 | 1.0 | 1.626e-04 | **0.00173** |

**解读**：η 模型的尾部 Wasserstein 距离比 MSE 基线**小约 15.5 倍**（0.0268 → 0.0017）。这正是论文的核心主张：在只有 100 个（几乎不含极端样本的）配对样本的条件下，MSE 模型输出分布的尾部严重偏离真值，而加入分位数-Wasserstein 正则的 η 模型几乎完美复原了尾部。代价是训练 MSE 从 2.6e-05 升到 1.6e-04（均值精度略降）——这是有意的权衡。

各阶段实测耗时（本机）：

| 阶段 | 轮数 | 耗时 | 速度 |
|---|---|---|---|
| MSE 基线训练 | 3000 | 约 6 秒 | ~482 it/s |
| GRF 预训练 | 1000 | 约 2 秒 | ~520 it/s |
| η 续训 | 3000 | **约 6.4 分钟** | ~7.5 it/s（每步要在百万级辅助样本上做分位数匹配） |

评估在 100×100 输入网格、10⁶ 样本上进行。输出 PDF 对比图见下。

![2D→1D 输出 PDF 对比：真值 vs MSE 基线 vs η-学习](eta/result_log/toy2d1d_pdf_compare.png)

**图 1（2D→1D）读图说明**：横轴为标量输出 `y`，纵轴为概率密度；左图为线性坐标，右图为对数坐标（尾部视图，纵轴跨 10 个数量级）。三条曲线分别是真值分布（蓝）、MSE 基线（橙）、η-学习（绿），均由 10⁶ 评估样本的直方图（400 箱）估计。

- **左图（线性）**：三条曲线都是典型的重尾右偏分布。MSE 基线在 `y≈0` 处的峰值高达约 395，远超真值峰值（约 15–20，峰形最宽），即"概率过度堆积在众数附近"；η-学习峰值约 100，比真值仍偏高但已明显缓解。
- **右图（对数，看尾部）**：这是本图的关键。在尾部 `y≈0.42–0.55` 区域，**MSE 基线（橙）最早陡降、降到约 10⁻² 量级，比真值低约一个数量级——严重低估极端事件概率**；η-学习（绿）下降最晚、最平缓，在 `y≈0.5` 附近仍保持约 10⁻¹ 量级，与真值（蓝）的阶梯形尾部贴合。中部 `y≈0.05–0.4` 三条曲线基本重合。
- η 曲线在 `y≲0` 处的个别向下尖刺是直方图在分布边界处的空箱伪影（η 预训练模型不加非负约束，输出可略微小于 0），不是模型行为异常。

一句话总结：**MSE 模型"峰过高、尾过轻"（向条件均值塌缩），η 正则把尾部概率质量补了回来**——这正是 tail_w1 从 0.0268 降到 0.0017 的图面解释。

### 7.3 玩具算例 2：2D→2D（完整默认配置）

命令：`python -m experiments.toy_2d_to_2d --save-npz result_log/toy2d2d_full.npz`

对状态映射 `u=(u₁,u₂)` 的标量可观测量 `g(u)=2|u₁|+0.5|u₂|` 评估：

| 模型 | seed | final_train_mse | **tail_w1（越小越好）** |
|---|---|---|---|
| MSE 基线 | 1 | 2.607e-08 | 0.08083 |
| **η-学习** | 28 | 1.299e-03 | **0.00229** |

**解读**：状态映射情形结论更强——η 模型的尾部 W1 距离比 MSE 基线**小约 35 倍**（0.0808 → 0.0023）。MSE 模型尽管把训练 MSE 压到了 2.6e-08（几乎逐点拟合训练集），其输出可观测量分布的尾部却偏差巨大，再次印证"逐点精度高 ≠ 分布尾部正确"。

各阶段实测耗时（本机）：MSE 3000 轮约 8 秒；GRF 预训练 1000 轮约 2.5 秒；η 续训 3000 轮约 6.8 分钟。可观测量 PDF 对比图见下。

![2D→2D 可观测量 PDF 对比：真值 vs MSE 基线 vs η-学习](eta/result_log/toy2d2d_pdf_compare.png)

**图 2（2D→2D）读图说明**：横轴为标量可观测量 `g(u)=2|u₁|+0.5|u₂|`，纵轴为概率密度；左图线性坐标，右图对数坐标（尾部视图）。三条曲线同样由 10⁶ 评估样本的直方图（400 箱）估计。

- **左图（线性）**：三条曲线均呈右偏长尾形，峰值在 `g(u)≈0.02–0.03`。真值（蓝）峰最锐利（高度约 18.5）；**η-学习（绿）的峰位与峰高几乎与真值重合**；MSE 基线（橙）峰高只有真值的约三分之一，且形态异常——呈"双峰"结构（`g(u)≈0.02–0.03` 与 `≈0.05–0.07` 各有一峰），峰被压低并向右摊开，即状态场整体被"抹平"。
- **右图（对数，看尾部）**：在 `g(u)≈0.8–1.05` 这一极端事件区域，**MSE 基线（橙）偏离最远——密度在 `g(u)≈0.82` 处提前崩落到 10⁻² 量级以下，而真值在该区间仍维持 10⁻¹~10⁻¹ 的平台**；η-学习（绿）与真值（蓝）全程贴合；`g(u)≳1.05` 后三条曲线才汇合到约 1.5×10⁻²。也就是说，MSE 模型把本应发生的极端事件"截断"了。

一句话总结：**MSE 模型把状态映射过度平滑（双峰伪影 + 尾部截断），η 正则在峰形与尾部两端都还原了真值分布**——对应 tail_w1 从 0.0808 降到 0.0023（约 35 倍改进）。

### 7.4 两次完整实验小结

| 算例 | MSE 基线 tail_w1 | η tail_w1 | η 相对改进 |
|---|---|---|---|
| 2D→1D（标量） | 0.02677 | 0.00173 | **≈15.5×** |
| 2D→2D（状态） | 0.08083 | 0.00229 | **≈35.3×** |

两个玩具算例均一次跑通（退出码 0），定性复现论文核心结论：η-学习以微小的均值精度损失换取极端事件分布尾部的量级级改善（直观图证见上方**图 1**与**图 2**：对数坐标下 MSE 曲线尾部崩落、η 曲线与真值贴合）。每次完整实验总耗时约 7 分钟（本机 RTX 5060 Laptop）；结果数组与日志保存在 `eta/result_log/` 下。

---

## 8. 如何自己运行（手把手指南）

以下命令假设：

- 仓库在 `<仓库根目录>/eta`；
- 使用 conda 环境 `pytorch`，完整 Python 路径为 `C:\Users\dn\miniconda3\envs\pytorch\python.exe`（如已 `conda activate pytorch` 则直接用 `python`）。

### 第 0 步：进入仓库目录

```bash
cd "<仓库根目录>/eta"
```

所有 `python -m experiments.xxx` 命令**必须在仓库根目录下执行**（脚本依赖相对导入与默认数据路径 `data/toy.pth`）。

### 第 1 步：快速验证安装（无需数据，数秒）

```bash
python examples/smoke_test_toy2d.py
```

预期输出末尾出现 `smoke_test_ok`，即安装无误。

### 第 2 步：确认玩具数据在位

```bash
ls -la data/toy.pth   # 应约 26 MB
```

若缺失，按[第 4.3 节](#43-下载玩具数据集zenodo)从 Zenodo 下载并校验。

### 第 3 步：运行玩具算例 1（2D→1D）

```bash
# 完整默认配置（MSE 3000 轮 + GRF 预训练 1000 轮 + η 续训 3000 轮），本机约 15 分钟
python -m experiments.toy_2d_to_1d --save-npz result_log/toy2d1d_full.npz
```

想先快速试跑（约 20 秒）确认流程：

```bash
python -m experiments.toy_2d_to_1d --mse-epochs 100 --pretrain-epochs 50 --eta-epochs 100 --save-npz result_log/toy2d1d_probe.npz
```

加 λ 敏感性扫描（额外按 `[1e-4,1e-3,1e-2,0.1,1,10]` 各训一轮 η，时间约 ×6）：

```bash
python -m experiments.toy_2d_to_1d --lambda-sweep
```

### 第 4 步：运行玩具算例 2（2D→2D）

```bash
python -m experiments.toy_2d_to_2d --save-npz result_log/toy2d2d_full.npz
```

同样支持 `--mse-epochs/--pretrain-epochs/--eta-epochs` 缩短运行。

### 常用参数速查（两个玩具脚本通用）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--mse-epochs` | 3000 | MSE 基线训练轮数 |
| `--pretrain-epochs` | 1000 | η 模型 GRF 预训练轮数 |
| `--eta-epochs` | 3000 | η 续训轮数 |
| `--eta-seeds` | 1D: 25；2D: 28 | η 训练随机种子（可给多个，如 `--eta-seeds 25 26 27`） |
| `--lambda` | 1.0 | η 正则权重 λ |
| `--omega` | 100 | 尾部索引刷新间隔（步） |
| `--lr` | 1e-4 | 学习率（仅 2D→1D 有） |
| `--device` | 自动 | `cuda` / `cuda:0` / `cpu` |
| `--toy-data` | `data/toy.pth` | 数据文件路径 |
| `--work-dir` / `--code-dir` | 仓库根 | 数据/代码根目录（亦可用环境变量 `ETA_WORK_DIR`/`ETA_CODE_DIR`） |
| `--skip-mse` / `--skip-eta` | 关 | 跳过某一阶段 |
| `--save-npz` | 无 | 结果数组保存路径 |
| `--lambda-sweep` / `--lambda-list` | 关 / 6 档 | λ 扫描（仅 2D→1D） |

输出解读：每行形如

```text
mse: seed=1 lambda=nan final_train_mse=… tail_w1=…
eta: seed=25 lambda=1.0 final_train_mse=… tail_w1=…
```

`tail_w1` 越小代表极端事件分布学得越好；η 行的 `final_train_mse` 略高于 mse 行是正常现象（λ 权衡了分布对齐，牺牲少量均值精度换尾部正确性）。

### 第 5 步（可选）：用 Jupyter notebook 交互式运行

```bash
conda activate pytorch
python -m pip install jupyterlab   # 若未装
python -m ipykernel install --user --name pytorch --display-name "pytorch (eta)"
jupyter lab
```

在 JupyterLab 中打开 `notebooks/toy--2D-to-1D.ipynb`（注意用**重命名后的本地文件**，原版文件名 Windows 无法保存），选择内核 `pytorch (eta)`，自上而下运行。notebook 会额外绘制真值/模型输出 PDF（线性+对数坐标）等图，并包含 revision 诊断单元格。

> notebook 单元格通过向上搜索含 `src/` 与 `experiments/` 的目录自动定位仓库根，无需手动设置 `ETA_CODE_DIR`；如 notebook 放在仓库外，需设环境变量 `ETA_CODE_DIR` 指向仓库根。

---

## 9. 常见问题排查（FAQ）

**Q1：`git clone` 报 `error: invalid path 'notebooks/toy--2D->1D.ipynb'`**
Windows 保留字符问题，按[第 4.2 节](#42-解决办法已执行)用 sparse-checkout + `git show` 导出解决。

**Q2：`ModuleNotFoundError: No module named 'xxx'`**
按报错补装：`torch`（PyTorch 官网装）、`flow_matching`、`torchdiffeq`、`KDEpy`、`seaborn`、`cartopy`。导入链：`experiments.common → src/models.py（flow_matching）→ src/test_utils.py → src/plot_utils.py（seaborn+cartopy）`，所以玩具算例也会要求这些包。

**Q3：KDEpy 报 `AssertionError`（`linbin_cython` 中 `np.allclose` 失败）**
KDEpy 版本过新（1.1.12 实测有此问题）。执行 `pip install "KDEpy==1.1.8"` 降级即可。

**Q4：`python` 命令提示"Python was not found"（Microsoft Store 弹窗）**
本机 `python` 是 Store 占位符。使用 `conda activate pytorch` 后的 `python`，或直接用完整路径 `C:\Users\dn\miniconda3\envs\pytorch\python.exe`。

**Q5：CUDA 不可用 / 想 CPU 跑**
玩具模型很小，CPU 也能跑，加 `--device cpu` 即可（速度可能慢数倍）。冒烟测试官方就标明 CPU 可用。

**Q6：结果数字与报告/论文不完全一致**
正常。论文 README 明确说明不保证跨 GPU/CUDA/BLAS 的比特级可复现；随机性来自种子控制下的训练。定性结论（η 的 tail_w1 显著小于 MSE）是稳定的。

**Q7：`FileNotFoundError: Missing toy data: .../data/toy.pth`**
数据未下载或路径不对。下载方法见[第 4.3 节](#43-下载玩具数据集zenodo)，或用 `--toy-data` 显式指定路径。

**Q8：想检查外部数据/检查点布局是否完整（ERA5 实验准备）**
```bash
python scripts/check_external_assets.py
```
该脚本只检查、不下载。

---

## 10. 进阶：通往完整 ERA5-Land 实验

玩具算例之外的全部实验都需要两样外部资产（放仓库根目录）：

1. **数据**：`data/era5land_USA_SouthEast_1999-2023_dailymax.nc`（2.2 GB，美国东南部 1999–2023 日最大降水，80×160 网格）——同一 Zenodo 数据记录 [10.5281/zenodo.21635446](https://doi.org/10.5281/zenodo.21635446)；
2. **预训练模型与样本**：`models/precip-srcnn/`、`models/fm/`、`samples/`——Zenodo 模型记录 [10.5281/zenodo.21635468](https://doi.org/10.5281/zenodo.21635468)，在仓库根解压即得该布局。

准备好后按顺序（对应 notebook 顺序）：

```bash
python -m experiments.era5land                                   # SRCNN 降尺度：评估缓存检查点或 --train-mse/--train-eta 训练
python -m experiments.era5land_dgm --field lores --train --sample  # Flow Matching 训练+采样
python -m experiments.era5land_dgm_plot                           # FM PDF 图 + η 校正
python -m experiments.era5land_evd --misspec-alpha -1 -0.5 0.5 1.5 2  # GEVD 先验敏感性
python -m experiments.era5land_computational_overhead             # 耗时/显存测量
```

官方 V100 参考耗时（摘自仓库 README）：ERA5 MSE 训练 54.8 min、η 续训 10.4 min、FM HR 采样最长约 10.2 h。完整手册见 `docs/command_line_experiments.md`。

---

*报告完。运行日志、结果数组与本文引用的两张对比图（`toy2d1d_pdf_compare.png`、`toy2d2d_pdf_compare.png`）均保存在 `eta/result_log/` 下（另有探针运行输出 `toy2d1d_probe.npz` 与完整运行日志 `toy2d1d_full.log`、`toy2d2d_full.log`）。*
