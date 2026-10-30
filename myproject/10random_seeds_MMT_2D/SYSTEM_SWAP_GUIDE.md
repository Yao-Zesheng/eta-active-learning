# 换新系统的改动清单（SYSTEM SWAP GUIDE）

本项目的 PDF 误差已采用 **nls 式口径**（与 dnosearch `examples/nls` 一致）：真值与模型预测各用
`scipy.stats.gaussian_kde`（Scott 自适应带宽）估计密度，在**固定窗口的 1000 点网格**上求值，
floor 1e-16，左矩形积分 |log₁₀ 差|，再除以 scale 归一。模型预测越出窗口时 KDE 值在窗外自动
衰减到 floor → e 惩罚，**不需要任何沉积网格或模型输出限界**。

换一个新系统研究时，按下表逐项改（改完跑 §4 冒烟，然后按 `../PLAYBOOK_10random_seeds.md` 执行十种子）。

---

## 0. 换系统 0 改动契约（MMT 项目起生效，经验固化）

**所有"约定"都定义在 problem.py 里；run.py 与 acquisition.py（eta 和 hard 两套）对
"网络吃原始变量还是 copula 坐标、候选点是什么分布、评估集怎么加权"完全无感知，换系统永不改动。**

| problem.py 定义 | 含义 | MMT 取值 | 其他系统的合法取值 |
|---|---|---|---|
| `DOMAIN` / `oracle` | 输入域 / 真值 | [-6,6]² + MATLAB 求解器直连 | 任意 |
| `to_unit` / `from_unit` | 候选/LHS 提案坐标 | 线性（域内均匀提案，dnosearch 原版） | 输入律 PIT copula（候选=输入律） |
| `input_pdf` | 输入律密度（USLW 分子、指标权重） | N(0,I) 未截断 | 任意分布 |
| `candidate_pdf` | 提案密度（重要性权重分母） | 1/面积（均匀盒） | PIT 提案时 ≡ `input_pdf` |
| `net_features` | 软约束网络输入坐标 | **恒等（原始变量，dnosearch 惯例）** | `2*to_unit(x)-1`（copula） |
| `law_copula` | hard 概率网格坐标 | 截断 Φ（输入律 PIT） | 恒为输入律 PIT，勿改 |
| `prepare()` | 数据装配 | 100² 均匀网格真值 + 输入律权重 `w` | 输入律样本（`w` 写均匀权重） |

三条规则：

1. **USLW 分母 KDE 权重 = `input_pdf/candidate_pdf`**（acquisition.py 内一个公式）：
   均匀提案 → p_X 加权（dnosearch 原版行为）；PIT 提案 → 权重≡1（自动退化为
   TOY/泊肃叶的无权 KDE）。两种约定共用同一份 acquisition.py。
2. **评估集不是输入律样本时**（如买不起输入律采样的贵系统），`prepare()` 在
   evaluation.npz 里写归一化权重键 `w`；run.py 的全部分布指标（metrics/PDF/W1 分位/
   rank-KS）自动加权。评估集就是输入律样本时 `w` 写均匀值即可，同一套 run.py。
3. **hard 的概率网格坐标必须取输入律 PIT（`law_copula`）**，否则网格均匀测度 ≠ 输入
   律测度，精确约束钉错目标律（MMT 冒烟实测：成员级 e 0.58–0.72 → 修复后 0.126 贴地板）。

---

---

## 1. 必改：系统定义（`_templates/eta/problem.py`，改完复制到 `_templates/hard/`）

| 标记 | 位置 | 内容 |
|---|---|---|
| TODO(1) | `DOMAIN` | 物理输入域 (2,2) |
| TODO(2) | `SCALE_FIXED` | y 归一化上界（推荐填已知安全上界；None=用 10⁶ 样本最大值） |
| TODO(3) | `oracle(x)` | 2D→1D 真值（解析式 / CSV 插值（参考 `../../POIS_2D/problem.py`）/ 求解器） |
| （可选）TODO(5) | `to_unit/from_unit/input_pdf` | 非均匀输入分布三件套（默认均匀；高斯样例见注释） |

改完在 `_templates/eta` 跑一次 `python problem.py` → 生成 `data/evaluation.npz` +
`data/reference_law.npz`（约 110 万次 oracle 调用），再把 `problem.py` 与 `data/` 复制到
`_templates/hard/`，然后 `python setup_cases.py`。

## 2. 必改/必查：**PDF 误差积分的上下限窗口**（容易漏！）

两份 `run.py` 各有一处（搜索 `y_int = np.linspace`）：

```python
# _templates/eta/run.py  （主评估段，循环外）
y_int=np.linspace(0.0, max(scale, float(ev['y_mc'].max())), 1000)
# _templates/hard/run.py （evaluate() 函数内）
y_int = np.linspace(0.0, max(scale, float(evaluation['y_mc'].max())), 1000)
```

- **下限 0 的假设是 y ≥ 0**（代码中已标 TODO(6)）。新系统输出可为负时，把 0 改成
  已知物理下界（推荐）或 `float(ev['y_mc'].min())`。
- **上限**只需**安全盖住真值支撑**即可，宽松无害（nls 的先例：真值实际在 [0.2, 0.42]，窗口
  用了宽松的 [0,1]）。当前取 `max(scale, y_mc.max())`，一般不用改；若 `SCALE_FIXED` 填了
  偏小的值，代码已自动用样本最大值兜底。
- 窗口两端的行为：真值密度在窗外为 floor；模型把概率质量放到窗外时，窗外格点 log₁₀ 差
  拉满（每格最多 16）→ e 增大——这就是越界惩罚，无需额外处理。
- 若想与 nls 完全同款（y 域不归一），去掉两处的 `/scale`；保留 `/scale` 则 e 是 z 域量纲，
  与本项目历史图一致。**二选一后不要再变**（口径变则数值不可比）。

## 3. 必查（按新系统的特点）

1. **输入维度 = 2**：框架写死（LHS 两列、MLP 首层 2、201² 网格、USLW 的 p_X）。其他维度需改
   `problem.py` 全部函数、两份 `run.py` 的 `u_init` 与 `LAYERS`、绘图脚本。
2. **y 为标量**；真值查询成本（prepare ~110 万次 + 每工况 100 次）——贵的系统先离线 CSV。
3. **周期输入**（如 θ）：oracle 内做周期闭合（`np.remainder`，参考 POIS_2D）。
4. **极端事件方向**：metrics 的 threshold 取参考律 `.99` 分位；若极端在低值侧改 `.01`。
   极端区域 MSE 已同时覆盖两端 0.1%，无需改。
5. **参考律节点**：`prepare()` 生成的 (p, q) 含两端几何加密（到 1e-7 分位），供 W1 项与硬约束
   `Q_ref` 使用——自动，无需改。

## 4. 冒烟清单（改完必做，共 5 条）

```bash
PY=/c/Users/dn/miniconda3/envs/pytorch/python.exe
cd _templates/eta
$PY problem.py                                                  # 数据生成，记下打印的 y 范围与 scale
$PY -B run.py --lambd 0 --iterations 1 --name s0                # 基线
$PY -B run.py --lambd 0.001 --iterations 1 --x-aux 20000 --threads 2 --name s1   # W1 路径（CPU 快验）
$PY -B run.py --lambd 0.001 --iterations 0 --device cuda --threads 4 --name sg   # GPU W1
$PY -B run.py --lambd 0 --iterations 1 --acq us --name su       # US 采集
cd ../hard && $PY -B run.py --iterations 0 --adam 50 --lbfgs 0 --threads 2 --name s
```
核对：① e 全部有限且量级合理（0.001–10 级）；② `s1`/hard 的**成员级** e 显著低（W1/硬约束
生效的证据）；③ LHS 102 ≠ 101；④ seeds 记录正确；⑤ `su` 的 config 含 `acq=us`。
（本项目交付前已用一个临时假系统跑通以上全部，历史见会话记录。）

## 5. 之后

`python setup_cases.py` → 按 `../PLAYBOOK_10random_seeds.md` §4 上 HPC 跑十种子
（先 `ACQ=uslw` 一轮 60/60，再 `ACQ=us` 一轮）→ `hpc/compute_summary.py <acq>` 汇总回传 →
`plot_per_seed.py <acq>` / `plot_seed_summary.py <acq>` 出图。

## 附：e 的口径速查（当前实现）

| 项 | 值 |
|---|---|
| 密度估计 | `gaussian_kde`（Scott 带宽，随各自数据自适应；评估集按输入分布采样故无需权重） |
| 窗口 | 1000 点线性网格，下限 0（y≥0 假设）、上限 max(scale, y_max) |
| floor | 1e-16（log₁₀ 保护，单格最大差 16） |
| 积分 | 左矩形（丢最后一格，nls 同款），÷ scale 归一到 z 域 |
| 与旧 TOY 沉积法的关系 | 口径不同、数值不可比；越界惩罚机制 = 窗外 KDE→floor（沉积法是边界堆积） |
