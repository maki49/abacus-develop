# LCAO–local RI 的 LR-TDDFT 与 Z-vector：效率诊断和算法路线

日期：2026-10-12。对象：63 原子 NV⁻、819 个 NAO、开壳层、6 个 TDA 根；杂化作业 3635517，以及 PBE 对照作业 3593949。

本文区分三种证据：实际任务与源码中已确认的事实、由公式得到的复杂度分析、尚需测量或实现的改进。原始调研未修改计算代码、调整正在运行的任务或提交新计算。后续实施优先级已更新；公式估算与尚未测量的加速建议均不是实测加速比。

## 1. 结论

**LCAO/NAO + local RI 没有原则上的效率障碍；目前的八小时以上耗时不能解释为占据–虚轨道空间太大导致的必然结果。** 现有证据指向 CPU 绑定异常、算符作用的数据组织、网格积分和迭代收敛策略。是否能在同等资源和力精度下达到 WEST 的绝对墙钟时间，仍需匹配基准。

尤其要区分 local RI 与稠密全局 RI。你们使用的 LibRI 已利用局域原子轨道和原子块稀疏性；不能把稠密 RI 交换核的四次标度直接套到它上面，也不应以“引入局域 RI”作为下一步建议。

按收益、改动范围和验证成本，采用以下实施优先级。保留完整响应空间；运算量减少不等同于整程序实测加速。

| 优先级 | 项目 | 改动成本 | 收益与边界 |
|---|---|---|---|
| 前置 | 验证线程绑定，记录作用向量数、分项时间和残差 | 小 | 建立可信基线；区分单次作用慢与作用次数多 |
| 1 | DM 变换改变收缩顺序 | 小 | 按较小 MO 维度选择中间矩阵；本例该变换 FLOP 减少约 71% |
| 2 | Hxc 合并自旋势后积分、投影 | 中 | 开壳层每向量四次势积分合为两次；先覆盖激发态与 Z 左端 CC_vo |
| 3 | Davidson 子空间实验、LR 专用预条件 | 配置实验小，预条件中等 | 同时减少 Hxc 和 EXX 作用；以相同残差、能量及力精度比较 |
| 4 | 跨离子步复用激发态初猜 | 中 | 复用已有跨几何 AO/MO 重叠及振幅投影；加速后续离子步 |
| 5 | 平均多重态只解一个 Z | 中 | 合并各态右端；核对全部 Z、二次 X、能量加权项及筛选线性关系 |
| 6 | LibRI 固定数据缓存、真正的多右端批处理 | 中到大 | 先按计时选择固定元数据与有界 workspace，再联合收缩多个右端 |
| 后续研究 | ISDF/THC、ADMM、响应轨道架构、受控 inexact 作用 | 大 | 用实测瓶颈和力误差预算决定；硬截断虚空间优先级最低 |

第一批实现选择 **1（DM 收缩换序）和 2（Hxc 自旋融合）**，分别构建、验证并提交。Hxc 第一阶段保留现有四个自旋块的势公式，在输出自旋上累加势后各做一次积分与投影；共享 Hartree/GGA 中间量、同时计算两自旋网格密度及通用多右端批处理另行验证。

Davidson 可先穿插比较 `pw_diag_ndim=4/8/12`（6 根对应 24/48/72 维子空间）及已有 `dav`/`dav_subspace`。三个主要实数数组合计约 97/193/290 MiB。缓存的 AV 可低成本组装子空间残差，但输入相关 EXX 筛选可能破坏线性，需抽查重新作用得到的残差及严格参考作用。

跨离子步初猜已有 `state_ovlp.h` 的 `cross_ao_overlap`、`mo_overlap_dist`、`project_reference_dist` 可复用；当前 `setup_eigenvectors_X` 仍清零并重新生成初猜。需补正交化、缺失方向和失效回退，不能直接复制旧几何的 MO 振幅。

## 2. 两个实际任务给出的信息

### 2.1 杂化作业 3635517：先排除一个明确的运行问题

远程计算目录：

/data/home/df_iopcas_cy49/work/lr-grad/18_NV_diamond/222/ddh/02_3E_avg

作业使用 1 个 MPI 进程、申请 6 个 CPU、1 张 V100-SXM2-32GB；启动脚本将 OpenMP 和 MKL 线程数设为 6。实际读取结果如下：

| 项目 | 观测 |
|---|---|
| 查询时累计墙钟时间 | 10:43:05，尚在第一次 LR 激发态求解 |
| 累计 CPU 时间 | 约 10:40:05 |
| 进程 CPU 利用率 | 约 99.5%，接近一个逻辑 CPU |
| Slurm 分配的 CPU 集合 | 0–2、24–26 |
| 进程全部 8 个线程的允许 CPU 集合 | **每个线程都是 CPU 0** |
| 主机最大常驻内存 | 约 69.4 GiB |
| GPU 短时采样 | 多个空闲采样夹有 82%/100% 活跃采样；显存随批次变化 |

这表明申请到的 6 个 CPU 没有被 OpenMP 线程有效使用。MPI 启动绑定、OpenMP 绑定及继承的 affinity 需要一起检查；现有观测还不能唯一确定是哪一层配置造成。不能由此承诺修复后恰好快 6 倍，但它足以使当前作业失去作为算法效率基线的资格。

GPU 采样仅持续数秒，证明“间歇工作”，不能用于推算整个十小时的平均 GPU 利用率。也不能把某个 491 MiB 的采样理解为整个程序始终只使用这点显存。

修正应在短测试中完成：确认每个线程的实际 affinity 落在分配的 CPU 内，再测固定数量的 LR 算符作用。Intel MPI 的 MPI/OpenMP 配置说明推荐按 OpenMP 线程组建立进程绑定域；Slurm 还要求留意线程继承父进程绑定的行为。具体设置须结合本集群启动方式验证，不能只凭一个环境变量认定成功。[Intel MPI](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-6/running-an-mpi-openmp-program.html)、[Slurm srun](https://slurm.schedmd.com/srun.html)

本次任务还有这些重要设置：

- DDH 通过 PBE0 形式设置，精确交换比例 0.18；完整 819 个 KS 轨道。
- 上下自旋占据数 128/126、虚轨道数 691/693；6 个激发态。
- lr_solver = dav_subspace，lr_thr = 1e-12，pw_diag_ndim = 4；最大 Davidson 子空间为 24。
- lr_degen_mode = average，目标为最低多重态的平均面；力阈值 0.01 eV/Å。
- exx_dm_threshold = 1e-4；该数值是密度块筛选阈值，不能直接换算成激发能或力误差。

当前输出缺少逐次 Davidson 残差和算符计时。因此还不能定量回答“究竟是一次作用慢，还是作用次数过多”，更不能把两者混为一谈。

证据：[运行快照](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-job3635517/runtime-snapshot.txt)、[INPUT](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-job3635517/INPUT)、[启动脚本](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-job3635517/launch.sbatch)。

### 2.2 PBE 对照：昂贵的不只有交换核

远程目录：

/data/home/df_iopcas_cy49/work/lr-grad/18_NV_diamond/222/pbe/02_3E_avg

提交记录对应作业 3593949；脚本配置 4 MPI × 10 OpenMP，CPU 为双路 Xeon Gold 6138。日志确认 4 个 MPI 进程；任务已结束，未对它的历史线程 affinity 做实测。启动记录与程序内嵌提交号不同，报告源码行为时以当前核对的版本为准，不将两个二进制视为完全相同。

PBE 同样使用完整虚空间、6 个根、24 维 Davidson 子空间，但 lr_thr = 1e-5。14 次离子步能量/力评估耗时 34,826 秒，约 9 小时 40 分；输出中的步号为 0–13。

| 计时项 | 14 次累计 | 每次平均 | 占总时间 |
|---|---:|---:|---:|
| 基态 KS 求解 | 647 s | 46 s | 1.86% |
| LR 激发态求解 | 16,052 s | 1,147 s，19.1 min | 46.09% |
| 激发态求力 | 17,983 s | 1,284 s，21.4 min | 51.64% |
| 其中 Z 方程整个流程 | 16,727 s | 1,195 s | 48.03% |
| 其中 Z-CG 迭代 | 16,184 s | 1,156 s | 46.47% |
| Hxc 网格计算，跨能量与力阶段 | 32,975 s | — | **94.69%** |

计时存在嵌套关系，最后三行不能再与前面的求解/求力时间相加。网格密度与势积分分别累计约 18,433 s、15,218 s。这个对照非常明确地提示：优化共同的响应算符可以同时加速激发态和力。

日志打印的 “Average iterative diagonalization steps” 从 28 增长至 388，是累计量；与最终 cal_grad 的 388 次一致。相邻差分显示每步约 26–38 次，不能把最后一行误读为最后一步独立做了 388 次迭代。

限制：当前 GPU 分支已经加入共享过渡密度缓存等变化，旧 PBE 的 94.69% 不能直接当作当前杂化二进制的性能占比。PBE 与杂化还存在资源、泛函、收敛阈值、版本差异；不能从二者总时长直接提取“交换积分额外开销”。

证据：[PBE 计时与迭代](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-pbe-3593949/timing-and-iterations.txt)、[INPUT](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-pbe-3593949/INPUT)、[提交记录](D:/typora-note/beyond-DFT/LR-TDDFT/解析梯度/implement/research-evidence/2026-10-12-pbe-3593949/submission.json)。

## 3. nocc × nvirt 到底有多大？

本例完整 TDA 空间维数为

\[
d=128\times691+126\times693=175\,766.
\]

一个实双精度向量只占 1.341 MiB，6 个向量约 8.05 MiB，24 个子空间向量约 32.18 MiB。日志中 psi、hpsi、spsi 各约 32.18 MiB，吻合这个计数。显式构造整个 d×d 矩阵则需约 247 GB，避免它当然必要。

**17.6 万维并不意味着 17.6 万次昂贵交换积分。** 一次作用通常先将一个完整 X 变成过渡密度，再计算线性响应势，最后投影回 MO 空间。真正需要优化的是这条算符作用路径及其重复次数。

该规模的向量代数和小规模 Rayleigh–Ritz 本身不足以解释十小时耗时。当前约 69 GiB 主机内存也不是由几个 32 MiB 的 Davidson 数组占去的，应独立核查积分数据、原子块映射、临时张量和各处拷贝。

### 3.1 WEST 没有删掉同一批物理自由度

采用 X 为 v×o 矩阵的记号，完整虚轨道表示与响应轨道表示的关系为

\[
U=C_v X,\qquad C_o^\dagger S U=0.
\]

在固定有限 AO 基组、完整虚空间且数值满秩的条件下，二者是同一响应空间的不同坐标。占据空间投影可写为

\[
Q=I-C_oC_o^\dagger S.
\]

U 有 M×o 个系数，但受到 o×o 个占据空间正交约束，独立自由度仍为 o(M−o)。本例两自旋 U 的原始系数合计 208,026，反而略多于 X 的 175,766。

WEST 用平面波表示响应轨道，并用投影排除占据分量，因此不必显式计算/存储全部 KS 虚轨道；但仍需存储和更新占据轨道数 × 网格规模的响应数据。它没有通过换个写法把响应空间维数消除。

对 NAO 改成 AO 响应轨道形式的潜在收益是省去全虚轨道对角化、方便局域性和批量作用，而不是维数突然下降。本例 M=819，已有完整 KS 轨道，而且 MO 表示下对角项很便宜，未必值得首先重写整个架构。

## 4. local RI 的标度应怎样分析？

### 4.1 基态低标度基础已经存在

local RI 将 AO 对的乘积展开在相关原子中心的辅助函数上。交换作用可示意为

\[
K[D]_{\mu\nu}
=\sum_{\lambda\kappa}(\mu\lambda|\nu\kappa)D_{\lambda\kappa}
\;\sim\;C\,V\,C\,D.
\]

这里 D 可以是非对称过渡密度，而不一定是基态密度。轨道有限支撑使每个 AO 对的局域辅助展开规模受限；固定密度、固定轨道截断半径下，每个原子的重叠邻居数在大体系极限不随总原子数增长。

Lin、Ren、He 的 ABACUS 工作已展示 local RI、原子块筛选与并行结合的低标度 HFX 构建，线性标度依赖短程相互作用或绝缘体密度矩阵衰减等条件。它是本实现具有算法潜力的直接证据；这不是关于当前 LR 激发态总成本的线性标度证明。[原始论文，JCTC DOI 10.1021/acs.jctc.0c00960](https://arxiv.org/abs/2009.12505)

### 4.2 为什么基态快，LR 仍可能慢？

这是需要重点研究的差别：

- 基态占据投影密度 D₀ 在有隙体系中具有近视性。
- 任意 Davidson 向量产生的 Dₓ=CᵥXCₒ† 未必有同样衰减；单个离域占据–虚轨道对也不具备整个占据投影的抵消结构。
- 最终缺陷态 X 可能较局域，但初始向量、校正方向、较高根和 Z 方向不一定如此。
- DDH 含长程 Coulomb 交换，不能直接继承短程 HSE 对相互作用范围的假设。
- 63 原子的小超胞与较长轨道截断半径还可能处于“邻居很多、渐近局域性尚未显现”的区间。

由此可做的复杂度判断是：若局域 AO 对与辅助块大小有界，而远距原子对仍普遍相连，理想组织的交换构建有机会呈近似二次的块数增长；若响应密度或有效相互作用也可在受控误差下局域，可能进一步降低。若代码遍历大量不存在的块，或者周期镜像/辅助块数量很大，实际耗时还会更差。**这是条件分析，不是对当前 LibRI 已测出的指数。**

即使交换构建是低阶的，完整、离域 MO 与 AO 之间的稠密变换仍可能是三次标度。因此“local RI 的 HFX 接近线性”和“整个 LR 求解接近三次或更差”并不矛盾。

作为对照，Jin 文中给出的未加速平面波杂化 TDDFT 工作量为 O(o²N_g log N_g)。在固定密度和基组截断下，o 与 N_g 都随体系增大，形式上约为 O(N³ log N)。这个表达式也不是 WEST 全部优化后的唯一标度：局域化可减少其中的轨道对，其他投影、正交化和通信仍各有成本。比较标度时还必须固定目标根数与误差，并单独记录迭代次数。NAO 的稠密变换三次标度与 local RI 的低阶块收缩，因而没有天然比平面波更差的指数障碍；小体系的巨大前因子仍可能决定实际胜负。[WEST 方法与复杂度说明](https://arxiv.org/html/2309.03513v2)

应记录每次作用的 C/V/D 有效块数、D 块随距离分布、通过筛选的连接数、GEMM 尺寸分布和搬运字节数。至少比较初猜、已收敛 X、残差方向、Z 方向四类输入，不能只用基态密度作性能代理。

### 4.3 稠密 RI 的四次标度只作为反例

若完全放弃局域结构，使用稠密三指标张量，则交换作用可写成

\[
(KX)_{ia}=\sum_P\sum_{jb}B^P_{ij}X_{jb}B^P_{ab}.
\]

它的常规收缩成本为

\[
O\!\left(R(o^2v+ov^2)\right),
\]

当辅助维数 R、o、v 均随体系线性增长时仍是四次标度。相比之下，Coulomb 型的可分离作用通常为 O(Rov)。这仅说明“采用 RI + Davidson”本身不保证低标度；**不能据此宣称 LibRI 的当前 local RI 路径也是四次标度。**

## 5. WEST 的哪些优势能迁移？

Jin 等人的加速是在完整平面波响应空间上，通过交换算符压缩、占据轨道局域化和 Z 方程的 inexact Krylov 实现的。215 原子基准中三种近似合用使总时间减少约 87%，同时最大力误差约 10 meV/Å；这是带精度预算的加速。[WEST 原文第 3 节](https://arxiv.org/html/2309.03513v2#S3)

| WEST 策略 | 对当前 NAO-MO 实现的意义 |
|---|---|
| ACE 加速基态 KS/Fock 对响应轨道的作用 | 当前对角项已经是 (εₐ−εᵢ)Xₐᵢ；该处没有同样的昂贵 Fock 作用，不能照搬 ACE 的整项收益 |
| 占据轨道局域化、筛选重叠对 | 局域性原则可借鉴，但本项目已有 local RI；应寻找尚未利用的响应/轨道局域性，而非重复引入现有技术 |
| Z 方程后期使用便宜算符 | 可研究，但它解决的是后续求力，不会直接缩短当前尚未结束的激发态求解 |
| 多层并行与 GPU 数据布局 | 多右端、矩阵批处理、减少反复搬运即使在单卡也有价值 |

占据轨道的幺正局域化本身不截断空间；但局域化后占据 Fock 块通常不再对角，不能继续不加修改地使用原来的 εᵢ 对角表达式。局域化之后的筛选才引入额外近似。

ACE 构造中使用有限数量空轨道，也不是对响应虚空间的硬截断。它控制交换算符近似精度。

绝对计时也需要对齐：论文 Figure 4 的 511 原子、单 GPU 节点粗读约 2.5 小时，那个节点有 **4 张 A100**；约 2 分钟的点使用 128 个节点、512 张 A100。基准计算第一激发态的激发能与力，基态 SCF 预先完成。它与本项目单 V100、6 个根、包含基态准备的耗时不是直接的一比一测试。[WEST 第 4 节与 Figure 4](https://arxiv.org/html/2309.03513v2#S4)

## 6. 保持完整响应空间的优先改进

### 6.1 Hxc：先合并自旋贡献，再做网格积分

当前 HamiltULR 按向量、输入自旋、输出自旋依次调用；已有共享缓存，同一输入自旋的过渡密度不会在两个输出自旋间重复构造。但每个自旋块仍分别完成势到 AO 的积分与投影。

在固定基态下，Hxc 对过渡密度是线性的。可先形成两个输出自旋的完整势：

\[
\delta v_\sigma
=v_H[\delta\rho_\uparrow+\delta\rho_\downarrow]
+\sum_{\sigma'} f^{xc}_{\sigma\sigma'}\,\delta\rho_{\sigma'}.
\]

然后每个输出自旋只做一次 AO 积分和 MO 投影。对 GGA，密度梯度响应也应按同一线性算符合并。这有望把该路径每向量的四次势积分合并成两次，Hartree 响应也可共享；它不是“全部 Hxc 快两倍”的保证。

进一步给网格计算增加多个右端，复用 AO 网格值、局域邻接表、静态基态核和批量积分。先用实际计时核实哪些量已经缓存，再决定缓存大小。若网格反复计算仍占主导，可研究过渡密度的辅助空间表示，但那会新增拟合误差及其力导数，优先级低于等价的自旋合并与批处理。

源码：[HamiltULR::hPsi](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/hamilt_ulr.hpp:94)、[Hxc 网格路径](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/operator_casida/operator_lr_hxc.cpp:106)。

### 6.2 过渡密度变换：改变矩阵乘法括号

当前实数 VO 路径先生成 M×v 的中间矩阵，再经 v 维收缩得到 M×M 过渡密度。以 X 为 v×o：

\[
D=C_v(C_oX^\mathsf T)^\mathsf T
=(C_vX)C_o^\mathsf T.
\]

右边先生成 M×o 的中间矩阵，完全等价。计算量从

\[
2Mov+2M^2v
\quad\longrightarrow\quad
2Mov+2M^2o.
\]

代入两自旋实际维数，合计由约 2.145 GFLOP 降到 0.629 GFLOP，约少 3.41 倍。这里只是该收缩的浮点操作数比，不是整程序加速比；也不改变三次渐近标度。复数、多 k、分布矩阵的共轭与转置约定须分别验证。

源码：[cal_dm_trans_pblas](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/dm_trans/dm_trans_parallel.cpp:50)。

### 6.3 local RI：从逐向量调用改成可复用的块作用

实际版本核对：

- 用户指出的原库为 /home/fortneu/LibRI，提交 8a2e936daa3c12860805600ea43b4aeb9cedd894。
- 本地 build-gpu 使用 /tmp/abacus-libri-gpu；普通 build 使用原库且未启用 LibRI GPU。
- 超算运行构建使用工程中的 libri-fixed，并启用 ENABLE_LIBRI_GPU。
- 超算与 /tmp/abacus-libri-gpu 的 LRI-cal_loop3_GPU.hpp SHA256 一致；它与原库文件有输出累积等修订，不能把三个目录当作逐字相同。

当前已经做了两项正确的事：C/V 积分按几何复用；交换矩阵回投影已经走组装后的矩阵乘法及 GPU workspace。不能再把“每次重新生成所有积分”或“逐 ia 小投影”当成当前瓶颈。

仍可见的重复工作包括：过渡密度拷贝与原子块转换；每次调用构建 GPU context、转置视图、映射和打包数据；按收缩标签上传、计算、下载；多个 Davidson 向量在外层依次执行。GPU 路径有约 5 GiB 的分批阈值和 CPU 打包/筛选环节，线程绑定异常尤其可能放大后者。

建议按几何建立可复用的收缩计划，缓存固定 C/V 的已打包 tile 和索引，将数个 Dₓ 作为多个右端处理。只有随输入变化的块数据需要更新；若筛选随 D 变化，则静态与动态连接必须分开管理。可预收缩的固定中间量也应先测内存收益，不能为了省一次乘法无界缓存所有中间张量。

还可研究把“生成整个 AO 交换矩阵再投影”融合成直接计算 K[Dₓ]Cₒ，最后左乘 Cᵥ†。固定的占据轨道因子可尝试提前与局域 C 块收缩，并复用 U=CᵥX。这保留所有虚轨道，可能减少完整 AO 矩阵的组装和搬运；但预收缩也可能破坏原子块稀疏性或增加存储，不能未经比较就声称降低了标度。这是比单纯改 GPU kernel 更深入、且能沿现有 local RI 接口探索的算法方向。

主机内存约 69 GiB，而 GPU 只有 32 GiB，合理目标是有界 tile 缓存、批量摊销和流水，不是要求整个 local RI 数据集常驻显存。批量大小应以实测峰值内存和有效 GEMM 大小决定。

源码：[EXX::cal_Hs](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/operator_casida/operator_lr_exx.cpp:166)、[实际 GPU 收缩库](//wsl.localhost/Ubuntu-24.04/tmp/abacus-libri-gpu/include/RI/ri/LRI-cal_loop3_GPU.hpp:32)、[原始 LibRI](//wsl.localhost/Ubuntu-24.04/home/fortneu/LibRI/include/RI/ri/LRI-cal_loop3_CPU.hpp:22)。

### 6.4 Davidson：已有能隙预条件，但还可改进

源码先把 precondition 初始化为 1，随后用 OperatorLRDiag 作用，实际得到 εₐ−εᵢ；不能停留在初始化那一行就认为没有预条件。当前 dav_subspace 还使用本征值位移，其分母为

\[
x=|\Delta\epsilon-\omega|,\qquad
p(x)=\tfrac12\!\left(1+x+\sqrt{1+(x-1)^2}\right).
\]

这是平滑、有下界的正分母；它与化学响应问题常用的带正则化的近似 (Δε−ω)⁻¹ 不完全相同。对于近共振的缺陷跃迁，它可能没有很好近似耦合核的逆，但这需要残差曲线来证明。

可比较：

- 保留现有方案，作为稳定基线。
- LR 专用、处理小分母的能隙预条件器。
- 低能跃迁块的近似核逆，外部空间仍使用对角预条件。
- 廉价辅助基组构造的近似 TDDFT 算符作为预条件器；最终残差仍针对原算符。
- 24、48、72 等重启空间；扩大空间会增加正交化成本，但本例向量存储尚小。
- 跨离子步投影/运输已收敛多重态子空间，减少重新建立初猜的代价；不能直接忽略 AO 基组随几何变化。

Zhou 与 Parker 的小辅助空间预条件研究，在其分子测试中将平均迭代数相对对角预条件减少约 2–3 倍，且不改变最终 TDDFT 问题；该结果支持路线可行，并非对 NV⁻ 的收益预测。[JCTC 2024，DOI 10.1021/acs.jctc.4c00577](https://arxiv.org/abs/2404.17133)

当前收敛标志采用相邻本征值变化 |Δω|，而不是直接以每个根的 ||AX−ωX|| 为终止条件。lr_thr=1e-12 Ry 对应约 1.36e-11 eV 的本征值变化，比 meV 能量尺度严格得多，但它不自动等同于同等级的本征向量或力精度。应同时输出真实残差、重启次数、未收敛根数及每次算符处理的向量数。

建议在固定几何上对求解容差做阶梯测试，再在一个位移几何上检查力；不能只凭能量小数位决定把阈值放宽多少。

源码：[能隙初始化](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_esolver/esolver_lr_lcao_tddft.cpp:80)、[预条件核](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_hsolver/kernels/bpcg_kernel_op.cpp:154)、[收敛判据](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_hsolver/diago_dav_subspace.cpp:194)。

### 6.5 输入密度筛选与“线性算符”需一致

当前 EXX 代码明确将 Z Hessian 的输入密度筛选关闭，理由是小 Krylov 方向被按绝对幅度丢弃会破坏线性递推；LR 能量路径仍可使用输入阈值。

若筛选掩码由 Dₓ 的幅度决定，一般会出现

\[
\widetilde A(\alpha x)\ne\alpha\widetilde A(x),\qquad
\widetilde A(x+y)\ne\widetilde A(x)+\widetilde A(y).
\]

这不证明当前激发态卡住就是筛选造成，但在追求 1e-12 的停止阈值时值得优先排查。

在小体系或固定几个测试向量上检查齐次性、可加性、自伴性及严格参考作用的残差。若采用自适应精度，需使用受控的 inexact eigensolver/Krylov 思路，并周期性用参考算符检查真实残差。不能把“每次随意改变阈值”视为普通 Davidson/CG 的无害替换。固定几何、输入无关且保持对称性的筛选掩码是一种候选，但其几何变化与力的一致性仍要验证。

## 7. 求力阶段的独立算法机会

### 7.1 平均多重态力可能只需要一次 Z 求解

当前 average 路径已经只处理目标多重态，未对全部 6 个根求力，也跳过了 Jahn–Teller 模式需要的非对角梯度。仍然可以更进一步。

在同一基态、同一几何和固定权重下，各态 Z 方程共享同一个线性算符：

\[
LZ_s=R_s.
\]

因此

\[
\bar R=\sum_s w_sR_s,\qquad
L\bar Z=\bar R,\qquad
\bar Z=\sum_s w_sZ_s.
\]

如果最终只需要平均能量的梯度，可以先平均各态二次密度、右端及相应显式梯度中间量，再解平均 Z，避免求出每个成员的完整 Z。对于两重态，它有机会将 Z 右端数由 2 降至 1；这与当前一次函数调用接收多个右端是不同的优化。总力计算还含其他工作，不能承诺总耗时减半。

不能先平均 X 再代入二次表达式：一般 R[平均 X] 不等于平均 R[X]。还必须保留各态 ω 加权项、Pulay/能量加权密度等全部贡献。

适用范围是固定群组和固定权重的平均面；若还要分别输出每态力、计算 JT 方向或完整简并梯度矩阵，则仍需相应独立响应。群组随几何改变的跟踪问题也需另行处理。

源码：[当前平均力路径](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_esolver/esolver_lr_rlx.cpp:216)、[多态 Z 方程构建](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/zeq_solver.hpp:314)。

### 7.2 Z 的预条件、多个右端与 inexact 作用

PBE 对照中 Z-CG 约占全部时间的 46.5%，值得与能量求解同等重视。当前 GPU 分支的 Z 求解已经有 CPU/GPU PCG、轨道能隙预条件、多个右端接口及真实残差输出；默认容差为 1e-6、最大 100 次。不能把 Davidson 的仅用本征值变化判据这一问题照搬到 Z 求解器上，也不应建议从零增加已经存在的预条件或多个右端接口。下一步应改善预条件质量、让底层 Hxc/local RI 真正摊销多个右端的成本，必要时研究子空间复用。如果只需平均力，先研究上述线性合并。[当前 Z-PCG](//wsl.localhost/Ubuntu-24.04/tmp/abacus-lr-grad-gpu/source/source_lcao/module_lr/zeq_cg_solver.cpp:159)

WEST 的 inexact Z 策略说明后期昂贵交换作用可以有受控近似。但具体删减哪一个交换项、如何决定切换阈值、如何保持最终力精度，需要按本项目的 Hessian 结构重新设计。NAO 还需一致处理基组 Pulay、重叠和 RI 导数。不能直接把论文中的某个阈值抄到 LibRI 的输入筛选上。

## 8. 更长期的算符压缩路线

### 8.1 优先沿 local RI 继续，而非先推倒重来

在数据证明 exchange contraction 占主导之前，不建议先替换整个积分后端。先利用已有 C/V、固定几何连接、局域块和多右端；随后根据实际块密度选择：

- 稀疏性良好：加强可证明误差界的原子块筛选、局域块调度和缓存。
- 原子块过密：比较低秩压缩是否能降低局域块或中间量的维数。
- AO/MO 变换主导：考虑响应轨道形式或保留局域占据轨道表示。
- 网格主导：优先改善 Hxc 自旋融合、批量积分和核表示。

### 8.2 ISDF/THC：压缩积分，不删除高能虚轨道

这是一条可系统研究的替代路线。以 X 为 o×v，若交换积分有可分离近似

\[
(ij|ab)\approx
\sum_{PQ}O_{iP}O_{jP}M_{PQ}V_{aQ}V_{bQ},
\]

则

\[
KX\approx O\left[M\odot(O^\mathsf T X V)\right]V^\mathsf T.
\]

当分解秩 r 随体系线性增长时，这种作用可组织为三次标度，并保留完整的 o×v 振幅。分解秩控制积分近似，与减少 nvirt 是不同的误差来源。

NAO-HFX 已有 ISDF 实现文献；THC/ISDF 的系统可改进低秩形式也有研究。[Qin 等，JPCA 2020](https://arxiv.org/abs/2003.01654)、[Lee、Lin、Head-Gordon，JCTC 2020](https://arxiv.org/abs/1911.00470)

这些研究不是“现成可用于本程序的周期 TDDFT 解析力”。还需验证非对称过渡密度、交换对称性、拟合/插值点随几何的导数以及全部 Pulay 项。对只有 819 个 AO 的体系，秩常数和建立分解的代价可能抵消优势；而对已低标度的 local RI，“三次”也不自动更优。可先作为预条件器试验，再考虑替换精确作用。

### 8.3 ADMM：存在局域基组的成功先例，但不是无成本近似

Hehn 等在 CP2K 的局域 Gaussian 轨道框架中实现了杂化 TDDFT 激发态性质与解析力，用较小辅助轨道基组近似交换，再用泛函修正补偿。它说明局域基组能实现实用的周期杂化激发态优化。[JCTC 2022，DOI 10.1021/acs.jctc.2c00144](https://pubmed.ncbi.nlm.nih.gov/35759470/)

其 ADMM 辅助轨道基组与 RI 的乘积辅助基组不是同一概念；你们已经用 DZP，能进一步缩小的空间可能少于使用大轨道基组的 CP2K 基准。论文大周期体系计时还使用大量 CPU 核，不能据此承诺单卡性能。复用公式时需同时核对作者的 [2024 勘误](https://doi.org/10.1021/acs.jctc.4c00604)。

### 8.4 小活性空间更适合初猜和预条件

将完整矩阵按低能空间 P 与其余空间 Q 分块，形式上有

\[
X_Q=-(A_{QQ}-\omega I)^{-1}A_{QP}X_P,
\]

\[
A_\mathrm{eff}(\omega)=
A_{PP}-A_{PQ}(A_{QQ}-\omega I)^{-1}A_{QP}.
\]

硬截断相当于丢掉第二项及 Q 响应。更稳妥的短期用途是让小空间给初猜或预条件，再用完整空间残差校正。若真正构建降维有效模型，其能量依赖、自能导数和响应恢复都需要推导；它不是只改 nvirt 参数。

## 9. 窗口误差记录应怎样解释？

现有验证文档 §3.8 对 N₂/HF 给出的例子包括：

| 设置 | 记录中的解析/差分相对偏差 |
|---|---:|
| nvirt=8，nbands=26，X/Z 窗口不一致 | 7.18% |
| nvirt=15，nbands=20，X/Z 一致但轨道总数不完整 | 0.648% |
| nvirt=21，nbands=26，完整空间 | 约 0% |

hBN/HF 也记录了由 2.74% 降到近零。这里的“近零”是文档给定测试和数值精度内的一致性，不是数学上无任何误差。

这个结果支持：**当前解析梯度推导和实现应保持其所需的完整 X/Z/KS 空间。** 但它不能推出“任何经过正确推导的截断模型都无法有一致解析力”。

需要区分两种误差：

1. 截断模型相对于完整基组响应模型的物理/模型误差。
2. 解析力相对于同一个截断能量的有限差分不一致。

你们记录中的窗口问题至少涉及第二种。只把 Z 扩满并将 X 零填充，并不自动补齐能量选择虚空间随几何变化的投影导数、活性/非活性轨道旋转等响应项。若真要用截断模型，需要为该模型重建一致的 Lagrangian 和导数。

此外，激发能对已收敛向量具有驻定性，小振幅分量对能量可能较弱，但力还依赖轨道响应和参数导数。NTO 中几个大奇异值、或者激发能已经收敛，不能单独保证 Z 和力也能以相同低秩截断。

因此本次建议首先压缩和复用算符，而保留完整响应空间。

来源：[窗口测试原文](//wsl.localhost/Ubuntu-24.04/home/fortneu/lr-grad/abacus-develop/LR-Grad-formulas/log/2026-08-最新解析&差分结果.md:909)。

## 10. 可执行的验证顺序

以下给出实施和验收顺序；实际执行结果单独记录，尚未执行的项目不作为性能证据。

| 顺序 | 实验/改动 | 要回答的问题 | 精度验收 |
|---|---|---|---|
| 前置 | 验证绑定，记录作用向量总数、分项时间、重启和残差 | 单次作用代价与收敛代价各占多少 | 抽查重新作用及严格参考算符的残差 |
| 1 | DM 按较小收缩维度选顺序 | 等价变换减少多少运算与存储 | 实数/复数、串行/分布式、VO/OO/VV；覆盖 nvirt 大于和小于 nocc |
| 2 | 合并 Hxc 自旋势，再积分与投影 | 四次势积分合成两次的收益 | 两自旋占据数不等、非对称 D、CPU/GPU；激发能和 Z/力回归 |
| 3 | Davidson 子空间、现有求解器、LR 预条件和容差实验 | 能否减少昂贵作用总数 | 相同残差、激发能、平均力及位移几何一致 |
| 4 | 跨离子步投影初猜 | 后续几何步可减少多少迭代 | 多重态跟踪、正交性、投影失效回退 |
| 5 | 平均多重态单 Z 求解 | 能否减少求力工作量 | 与逐态求力再平均逐分量比较 |
| 6 | LibRI 固定 tile/计划复用，再做多右端 | 打包和搬运是否主导 | 固定阈值下结果一致，几何/分布变化时缓存失效 |
| 后续 | 可控筛选、低秩、inexact 及不同规模基准 | 近似收益与实际标度 | 相同模型解析/FD、相对完整参考的力误差、相同资源 |

一次算符作用建议拆分为：

MO→AO、过渡密度打包、网格密度、Hxc 势与积分、local RI 筛选/打包、上传、GPU 收缩、下载/累加、AO→MO；再统计正交化与子空间求解。计时应记录向量数，不能把一个处理 6 个向量的调用与一个处理 1 个向量的调用直接比较。

报告指标应包括：固定几何准备时间、单向量与批量吞吐、达到目标残差所需作用次数、激发态总时间、Z 与最终力时间、CPU/GPU 峰值内存、有效积分块数。

对于优化阈值 10 meV/Å，可以先将算法近似的最大力误差预算设在更小的 1–5 meV/Å 范围，再根据用途调整；这是建议的验收目标，并非已证明当前近似能满足。有限差分需使用相同模型与充分收敛的基态，检查步长依赖，并对简并态使用一致的群组跟踪/平均能量。

## 11. 判断边界

已经确认的是：当前任务有 CPU 绑定异常；PBE 对照被 Hxc 网格计算主导；完整振幅空间存储并不大；代码已有能隙与本征值位移预条件、共享密度缓存、local RI 和矩阵回投影。

由公式支持但尚未实施的是：Hxc 自旋融合、DM 收缩次序调整、平均右端单 Z 求解；这些都有保持现有模型的实现路线。

仍需实验决定的是：哪一项支配当前杂化 LR 的单次作用，密度筛选是否影响收敛，修正绑定后能快多少，local RI 的有效标度，以及低秩或响应轨道重写是否值得。

**我会继续沿 NAO–local RI 路线推进，先建立可信的单卡基准并减少重复作用，而不会因为这一次十小时任务就判定需要更换基组或牺牲虚空间完整性。**

## 12. 主要文献与核对范围

1. Jin 等，JCTC 19, 8689–8705 (2023)，[DOI 10.1021/acs.jctc.3c00986](https://doi.org/10.1021/acs.jctc.3c00986)。核对本地正文、性能整理和 [arXiv 全文/SI](https://arxiv.org/html/2309.03513v2)。本地正文：[PDF](//wsl.localhost/Ubuntu-24.04/home/fortneu/work/tex/lr-grad/ref/23JCTC-Jin-West-TD-PW.pdf)。
2. Lin、Ren、He，Efficient Hybrid Density Functional Calculations for Large Periodic Systems Using Numerical Atomic Orbitals，[DOI 10.1021/acs.jctc.0c00960](https://doi.org/10.1021/acs.jctc.0c00960)，[预印本](https://arxiv.org/abs/2009.12505)。用于 local RI 的局域性与筛选条件。
3. Hehn 等，Excited-State Properties for Extended Systems: Efficient Hybrid Density Functional Methods，[DOI 10.1021/acs.jctc.2c00144](https://doi.org/10.1021/acs.jctc.2c00144)。核对本地正文 [PDF](//wsl.localhost/Ubuntu-24.04/home/fortneu/work/tex/lr-grad/ref/22JCTC-Hehn-CP2K-LRTD-gradient.pdf)；另查 [2024 勘误](https://doi.org/10.1021/acs.jctc.4c00604)。
4. Zhou、Parker，Converging TDDFT Calculations in 5 Iterations with Minimal Auxiliary Preconditioning，[DOI 10.1021/acs.jctc.4c00577](https://doi.org/10.1021/acs.jctc.4c00577)，[预印本](https://arxiv.org/abs/2404.17133)。
5. Qin 等，Interpolative separable density fitting decomposition for accelerating Hartree-Fock exchange calculations within numerical atomic orbitals，[DOI 10.1021/acs.jpca.0c02826](https://doi.org/10.1021/acs.jpca.0c02826)，[预印本](https://arxiv.org/abs/2003.01654)。
6. Lee、Lin、Head-Gordon，Systematically Improvable Tensor Hypercontraction，[DOI 10.1021/acs.jctc.9b00820](https://doi.org/10.1021/acs.jctc.9b00820)，[预印本](https://arxiv.org/abs/1911.00470)。

代码核对基于 /tmp/abacus-lr-grad-gpu 的 60a0a57b5f1c87ddc9f4025ec96f7bf9a7b27d4c。激发态入口、Davidson 子空间求解器、LR 交换算符三个关键文件已与超算运行工程作 SHA256 对照，结果一致；没有据此声称整个构建环境或所有源码逐项相同。
