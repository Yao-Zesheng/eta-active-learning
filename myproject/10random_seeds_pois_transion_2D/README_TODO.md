# pois_transion_2D 十随机种子测试项目（待填骨架）

本文件夹是从 TOY_2D 十种子框架抽出的**干净模板**：框架代码全部就绪（作业脚本/watchdog/
汇总/绘图均参数化，USLW 与 US 两轮实验共用同一套案例文件夹），只留 `problem.py` 里的
系统定义待你补充。填完后严格按本清单走一遍即可完整复现十种子测试。

---

## 1. 你必须填的（全部在 `_templates/eta/problem.py`，两份模板共用同一文件）

| 标记 | 内容 | 说明 |
|---|---|---|
| **TODO(1)** | `DOMAIN` | 物理输入域 (2,2)。例：泊肃叶风格 `[[0.001,0.06],[0,π]]` |
| **TODO(2)** | `SCALE_FIXED` | y 归一化上界；有安全上界就填（推荐），否则 None 用 10⁶ 样本最大值 |
| **TODO(3)** | `oracle(x)` | **2D→1D 真值关系**。解析式 / CSV 网格插值（照抄 `../../POIS_2D/problem.py` 的双线性+周期闭合写法）/ 求解器调用 |

填完在 `_templates/eta` 里执行 **一次**：
```bash
python problem.py    # 生成 data/evaluation.npz + data/reference_law.npz（约 110 万次 oracle 调用）
```
成功后把 `problem.py` 和 `data/` 同步到 `_templates/hard/`（或直接 `cp`），然后
`python setup_cases.py` 生成 20 个案例文件夹（`eta_s101..s110`、`hard_s101..s110`）。

## 2. 按你的案例特点检查/可能要改的（重要提醒）

1. **输入分布（默认：域上均匀）**——若非均匀（如高斯），按 `problem.py` 中 TODO(5) 的样例替换
   `to_unit/from_unit/input_pdf` 三个函数（必须互相自洽）。USLW 权重、LHS、候选点全部经
   `to_unit` 生效。
2. **y ≥ 0 假设**——PDF 误差窗口下限是 0（两份 `run.py` 中标了 TODO(6)，`y_int=np.linspace(0.0,...)`）。
   y 可为负的系统把下限改成物理下界。详见 `SYSTEM_SWAP_GUIDE.md` §2。
3. **输入维度 = 2**——整套框架（LHS 两列、MLP 首层 2、201² 绘图网格）按 2 维写死；其他维度需改
   `problem.py` 全部函数、两份 `run.py` 的 `u_init` 列数与 `LAYERS`、绘图脚本。
4. **PDF 误差已换成 nls 式口径**——`gaussian_kde` 双边 + 固定窗口 [0, max(scale,y_max)] 千点网格 +
   floor 1e-16 + 左矩形（与 dnosearch examples/nls 一致；模型越界自动被窗外 floor 惩罚）。窗口
   上下限的调整见 `SYSTEM_SWAP_GUIDE.md` §2。
5. **W1 辅助样本分布无关**——eta `run.py` 的辅助采样是 copula 均匀采样（对任意输入分布恒正确），
   换系统无需改。
6. **oracle 成本**——prepare 约 110 万次调用 + 每工况 100 次。若单次求解昂贵，先离线生成大样本
   CSV 再插值（POIS_2D 的做法），不要直接调求解器。
7. **真值的周期性/边界**——若某输入是周期的（如 θ），在 oracle 插值里做周期闭合
   （`np.remainder`），参考 POIS_2D。
8. **p99 阈值语义**——metrics 里 threshold 取参考律 99% 分位；若你的"极端事件"是低值侧，把
   `.99` 改 `.01` 并相应调整 top1 指标方向（极端区域 MSE 已同时覆盖两端 0.1%，不用改）。

## 3. 填完后必做的本地冒烟（缺一不可）

```bash
PY=/c/Users/dn/miniconda3/envs/pytorch/python.exe
cd _templates/eta
$PY problem.py                                   # 数据生成 + 打印 y 范围/scale
$PY -B run.py --lambd 0 --iterations 1 --name smoke_default                    # 基线（101/1）
$PY -B run.py --lambd 0.001 --iterations 1 --lhs-seed 102 --seed-base 1001 --name smoke_new  # W1 路径
$PY -B run.py --lambd 0.001 --iterations 0 --device cuda --threads 4 --name smoke_gpu          # GPU
$PY -B run.py --lambd 0 --iterations 1 --acq us --name smoke_us                # US 采集
cd ../hard && $PY -B run.py --iterations 0 --adam 100 --lbfgs 0 --threads 4 --acq us --name smoke
```
核对：① 指标全部有限；② LHS 102 ≠ LHS 101 的 initial_lhs.npz；③ round_000.pt 的 seeds 正确；
④ GPU 可跑；⑤ **smoke_us 的 config 里 selection 含 `acq=us`**（上一项目曾因作业脚本漏传
`--acq` 报废 60 个工况——本模板的作业脚本已带，但冒烟仍要验）。
冒烟产物在 `results/smoke*`，可留可删（回收站）。

## 4. 十种子完整流程（详见 `../PLAYBOOK_10random_seeds.md` §4）

```bash
python setup_cases.py                       # 20 个案例文件夹
tar czf pkg.tgz hpc setup_cases.py eta_s1?? eta_s110 hard_s1?? hard_s110 --exclude='*/results/smoke*'
# 上传解包到 ~/scratch/10random_seeds_pois_transion_2D，软链 pylib/pylib_cuda，sed 去 CRLF（见手册）
# 第一轮（USLW）：
sbatch -p tiny -c 1 -t 12:00:00 -J wd_uslw -o logs/watchdog_%j.out \
    --export=ALL,ACQ=uslw --wrap='bash hpc/watchdog.sh'
# 60/60 后第二轮（US，共用文件夹、结果目录自动分开）：
sbatch -p tiny -c 1 -t 12:00:00 -J wd_us -o logs/watchdog_%j.out \
    --export=ALL,ACQ=us --wrap='bash hpc/watchdog.sh'
# 每轮 60/60 后：远程 hpc/compute_summary.py <acq> → 拉 summary_seeds_<acq>.npz + small tar
# 本地出图：
python plot_per_seed.py uslw      # 20 张每种子图
python plot_seed_summary.py uslw  # 十种子汇总（中英）
python plot_per_seed.py us && python plot_seed_summary.py us
```

## 5. 文件索引

| 文件 | 状态 |
|---|---|
| `_templates/eta|hard/problem.py` | **待填**（TODO 1–3；4–5 可选） |
| `_templates/*/run.py` | 就绪（--lambd/--acq/--lhs-seed/--seed-base/--device/--threads；PDF 网格自适应；TODO(6) 标记） |
| `_templates/*/mlp_repro.py acquisition.py model.py training.py` | 就绪（与系统无关） |
| `setup_cases.py` | 就绪（生成 20 个案例；会先检查 data 已生成） |
| `hpc/eta_job.sh hard_job.sh watchdog.sh compute_summary.py` | 就绪（ACQ 参数化） |
| `plot_per_seed.py plot_seed_summary.py` | 就绪（ACQ 参数化；中英两版汇总） |
