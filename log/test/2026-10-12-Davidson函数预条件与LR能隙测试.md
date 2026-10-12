# Davidson 函数预条件与 LR 能隙测试

## 改动与范围

`Diago_DavSubspace::diag` 增加显式函数参数：

```cpp
using Preconditioner = std::function<void(
    T* residuals, const Real* ritz_values, int dim, int nvec)>;
```

回调原位作用于紧凑排列的未收敛根，`residuals` 为 `[nvec][dim]`，
`ritz_values` 与这些根逐一对应。两个缓冲区都在模板指定的 Device 上。
它可以执行非对角逆算符；所需临时空间由回调管理。原有 `diag` 重载继续调用
原 CPU/GPU `precondition_op`，PW 的平滑分母、初猜、停止准则均保持原样。
构造接口兼容旧调用，仍接收对角数组；带回调的 `diag` 不使用该数组执行预条件。

LR 的 `dav_subspace` 使用轨道能隙（不是含 Hxc/EXX 的完整 A 矩阵对角元）：

\[
D_{iak\sigma}=\epsilon_{ak\sigma}-\epsilon_{ik\sigma},\qquad
z_{jn}=\frac{r_{jn}}
{\operatorname{copysign}(\max(|D_j-\omega_n|,10^{-4}\,\mathrm{Ry}),D_j-\omega_n)}.
\]

保留分母符号；只有共振附近作下限保护，`1e-4 Ry` 约为 `1.36 meV`。
该策略和下限放在 LR 层，通用 Davidson 不再决定调用者必须用哪一种预条件。
LR 目前即使在 CUDA 构建中也走 CPU Davidson，故此处实现 CPU 的实数/复数版本。
`dav`、CG 和 LAPACK 的求解算法没有改动。

开壳层预条件数组原来把下自旋偏移写成 `paraX[0].local_size`，遗漏 `nk`，
会覆盖上自旋后续 k 点并留下未赋值尾段。现在复用 `OperatorLRDiag` 的能隙，
按完整 `[spin][k][occupied][virtual]` 块追加，消除手算输出偏移及原位乘法。

新增 LR 的求解后真实残差及算符向量计数。真实残差检查额外计算一次含全部根的
`A X`；打印的迭代调用数/向量数不含这次检查。计时器单列
`HSolverLR::report_davidson_residuals`。不能把额外检查开销忽略后声称总耗时加速。

## 多 k 点初猜修复

只换能隙预条件、沿用旧初猜时，开壳层双 k 点第二根从正确的 `0.22201293 Ry`
变成 `0.51440087 Ry`，尽管残差只有约 `5e-6 Ry`。全矩阵解确认这是漏根，
不能算作有效加速。旧初猜将两自旋跃迁绑在同一个向量里，且优先填满第一个 k 点；
精确能隙逆会使这种初始子空间的限制更突出。

对多 k 点 `dav_subspace`，现在从全部自旋、k 点选取最低能隙的独立跃迁。
每个初猜叠加 `1e-3/sqrt(N)` 幅度的可复现均匀扰动，按全局跃迁次序产生，
不依赖 MPI 分布。原因是现有 Davidson 无条件扩展初始 Ritz 根；双 k 点纯单位初猜
恰好包含精确本征态时，会在零残差归一化处退出。微扰也给其他对称子空间非零投影。
这并未修复通用 Davidson 的零残差处理；后续可独立改进其残差收敛和秩处理。

Gamma 保留原初猜。试验曾把新初猜用于全部 LR，但 HSE 六态从 52 个算符向量恶化到
175 个，因此最终将初猜修复限定为实际暴露漏根问题的多 k 点路径。

## 对照方法与结果

环境：ABACUS `v3.11.0-beta10`；4 MPI ranks；`OMP_NUM_THREADS=1`、
`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`I_MPI_FABRICS=shm`。
运行时固定 Libxc **5.1.7**（含 KXC）。使用 H2O+，23 个 KS 轨道，
`nocc=4/3`、`nvirt=19/20`，`pw_diag_ndim=4`。Gamma 实数、Gamma 复数、
双 k 点复数均覆盖；HSE 六态由 Gamma PBE 算例改为 HSE 基态与核。
这是小体系 CPU 验证，不能外推为 63 原子 NV- 的 GPU 加速比。

表中数字是迭代中 `A` 处理的向量总数；多个运行与编译/其他检查并行，
不使用总墙钟时间推断加速比。`lr_thr` 仍是能量变化阈值，另列最大真实残差。

| 算例，`lr_thr=1e-9` | 旧实现向量数 | 最终实现向量数 | 旧最大残差 / Ry | 最终最大残差 / Ry |
|---|---:|---:|---:|---:|
| PBE Gamma 实数，2 态 | 23 | 26 | 5.64e-6 | 4.54e-6 |
| PBE Gamma 复数，2 态 | 23 | 26 | 5.64e-6 | 4.54e-6 |
| PBE 双 k 点，2 态 | 203 | 10 | 2.25e-5 | 5.36e-6 |
| PBE Gamma，6 态 | 81 | 57 | 1.47e-5 | 3.29e-6 |
| HSE Gamma，6 态 | 81 | 52 | 1.60e-5 | 7.14e-6 |

双 k 点另有“只修复 `nk`，仍用旧平滑预条件与旧初猜”的控制：
99 次迭代/203 个向量变为 15 次/31 个向量，根仍正确。
最终为 5 次/10 个向量。因此其大幅改善首先来自索引修复，不能全归于新分母。
Gamma 两态没有加速；六态 PBE/HSE 向量数分别减少约 30%/36%。

相同初猜、相同更紧的 `lr_thr=1e-12` 下，单独更换预条件仍有改善：

| Gamma 六态 | 旧向量数 | 能隙预条件向量数 | 旧最大残差 / Ry | 能隙最大残差 / Ry |
|---|---:|---:|---:|---:|
| PBE | 103 | 68 | 3.57e-7 | 1.55e-7 |
| HSE | 105 | 66 | 4.32e-7 | 2.24e-7 |

这两组新残差更小，向量数减少约 34%/37%，不是降低解精度得到的结果。

## 力和既有回归的精度限制

在 PBE Gamma 两态上，`lr_thr=1e-9` 的旧力参考也尚未充分收敛。
全矩阵解的 18 个力分量绝对值之和为 `103.0514108594 eV/Angstrom`；
旧迭代为 `103.0514491282`，能隙预条件为 `103.0514645337`。
相对全矩阵的最大逐分量误差分别为 `1.87e-5` 和 `2.20e-5 eV/Angstrom`。
收紧至 `1e-12` 后，最大逐分量误差分别降到 `4.03e-7` 和 `1.46e-6`。
最终能隙实现再收紧至 `1e-14` 时，18 次迭代/34 个向量，最大真实残差
`3.43e-8 Ry`；力总和为 `103.0514105960`，最大逐分量误差 `1.27e-7 eV/Angstrom`。
因此原测试 `1e-6` 的力总和容差比其旧振幅精度还严格；预条件改变迭代轨迹就会
触发参考差异。保留测试 INPUT、容差与参考文件，未为通过测试而替换结果。

旧的闭壳层多 k 点测试采用 `lr_thr=1e-2`，也会随迭代轨迹改变能量；
全矩阵最低两根为 singlet `0.19183641, 0.19213714 Ry`，
triplet `0.18083682, 0.18126541 Ry`，与旧松阈值参考本来就有差异。
收紧至 `1e-14` 后，新实现四个能量均与上述全矩阵结果在打印精度上一致；
但 singlet 仍达到 180 次迭代上限，最大残差 `5.13e-6 Ry`，不能称为满足严格阈值。
同条件旧实现两个通道都达到 180 次上限，最大残差分别 `1.72e-3`、`6.06e-4 Ry`。
这说明预条件改善了此例，但默认子空间及迭代预算还不足以可靠满足很紧的精度。

## 验证命令与记录

原始日志、控制二进制与输入在 `.diagnostics/dav-preconditioner/`。
`run_cases.py` 接受二进制变体名和算例名，并通过 `DAV_BENCH_THR`、
`DAV_BENCH_SOLVER`、`DAV_BENCH_NPROC` 控制诊断运行。

```bash
cmake --build build --target abacus_std_para MODULE_LR_preconditioner -j 6
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 I_MPI_FABRICS=shm \
  ctest --test-dir build -V -R '^(MODULE_HSOLVER_dav_subspace|MODULE_LR_preconditioner)$'
cmake --build build-gpu --target hsolver lr esolver -j 4
python3 .diagnostics/dav-preconditioner/run_cases.py ready
DAV_BENCH_THR=1e-14 python3 .diagnostics/dav-preconditioner/run_cases.py ready-tight pbe-gamma
bash .diagnostics/dav-preconditioner/run-regression-ready.sh
```

- CPU 构建通过；9 项新增单元检查通过：实/复旧接口与等价回调结果逐元素相同、
  非对角回调、未收敛根/能量同步压缩、空回调拒绝、双自旋不同维度与多 k 点布局、
  初猜独立性与可复现性、分母符号/共振保护/无效输入。
- 五个相关 CTest 目标的 26 项单元测试全部通过；额外 4-rank DM 测试每个 rank
  的 4 项检查通过。`tests/08_RI` 的 `^lr_` 共 11 个算例：**4 通过、7 失败**。
  其中 5 项与修改前相同：LDA Gamma、LDA S1 Gamma、PBE Gamma、HF Gamma
  的力参考偏差，以及 HF 开壳层 Z-PCG 的第 0 步负曲率失败（残差 `0.0685146`）。
  新增的两项参考差异为上文分析的开壳层 PBE Gamma 力与闭壳层松阈值多 k 点能量。
  因此不能把本次验证描述为“全部回归通过”。
- PW `tests/01_PW/024_PW_DS`：4 MPI ranks 的 5 项检查通过；
  修改前后筛出的 **45 条能量及电子迭代记录逐行相同**。
- 开壳层双 k 点的 1/4 MPI ranks 都得到 `0.19361548, 0.22201293 Ry`，
  均为 5 次迭代、10 个算符向量；最大残差分别为 `3.64e-6`、`5.36e-6 Ry`。
- CUDA `hsolver`、`lr`、`esolver` 模块编译通过；无可用 GPU，未运行 GPU 算例。
- 没有新增 INPUT 参数，也未改参数解析、默认值或可选值；不需要重新生成 INPUT 文档。
  LR 内部算法变化由本报告及接口注释说明。
- 治理检查无错误。新增头依赖分别用于声明 `std::vector`、调用 LR 自由计算函数、
  使用显式 communicator 的内部归约包装。未增加全局控制依赖。
- 全文件质量评分 9 个文件中 8 个达到 60 分；既有 `hsolver_lrtd.hpp` 为 8 分。
  主要扣分为既有 `.hpp` 实现头（50 分）与 9 个全局引用（27 分），均非本次新增。
  不为消除历史评分而扩大本次重构范围；本次新增 `auto` 的扣分不代表数值或依赖问题。
