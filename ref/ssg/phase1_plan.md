# Phase 1 可执行实现计划:共线 altermagnet 自旋空间群 (nspin=2 SSG)

> 目标:让 nspin=2 共线磁性体系利用**自旋翻转陪集** `[C2⊥spin ‖ g]`(把 up 子晶格映到 down 子晶格的空间操作 g,配一个自旋 up↔down 翻转)做对称约化,从而缩小 IBZ、加速 SCF/EXX。当前 nspin=2 磁性路径把这些操作当作"跨磁类型"直接丢弃(`symm_magnetic.cpp:10-79` 的 relabel-by-magmom),这是 altermagnet 的能力缺口。

## 0. 关键物理:与 nspin=4 磁群的区别

| | nspin=4 磁群(现有) | nspin=2 SSG 翻转陪集(本期) |
|---|---|---|
| 陪集操作 | `Θ·g`(**反幺正**,含复共轭 K) | `[C2⊥spin ‖ g]`(**幺正**,纯 up↔down 交换) |
| 判据 | `spin_so3(g)·m_i = -m_{g(i)}` | `mag[iat] = -mag[g(iat)]`(标量) |
| 密度作用 | Θ 反号磁化 `rho^y,...`(耦合 3 分量) | 交换 `rho_up ↔ rho_down` + 空间旋转 |
| k 空间 | `k → -k` + 反号 | `(k,up) ↔ (g·k, down)`,**无 −k** |
| 存储 | `gmatrix_anti[]`, `nrotk_anti`, `magnetic_nspin4` | **新** `gmatrix_flip[]`, `nrotk_flip`, `spin_flip_nspin2` |

要点:nspin=2 翻转陪集是**幺正**的(实 Hamiltonian,up/down 两块独立实矩阵,翻转只是交换两块 + 旋转 k),**不含复共轭**,因此与现有 `trs_inv=-1` 的反幺正机制是不同的东西,需独立存储,不能复用 `gmatrix_anti`。

## 1. 门控 (gating)

新增输入旗标 `symmetry_ssg`(int,默认 0)。仅当:
```
symmetry_ssg==1  &&  nspin==2  &&  !pricell_loop(有子晶格磁矩差异)  &&  symmetry on
```
才走 SSG 新路径。**旗标关闭时,所有现有行为逐位不变**(纯新增门控分支)。SOC(lspinorb=1)与 nspin=4 不受影响。

## 2. 分三个增量 (increment) 实现

### Increment 1 — 检测 + 存储 + 输出(基础,零回归)✅ 已完成并验证 (2026-08-14)

**验证结果**(rutile FeO₂ 人造 altermagnet,Fe mag ±5 + O 配体,`build/abacus_std_para`):
- `symmetry_ssg=1`:完整化学群 16 op → **keep=8(幺正磁子群 D_2h)+ flip=8(自旋翻转陪集)= 16 SSG op**。打印 `SPIN-FLIP COSET OPERATIONS = 8` / `SPIN SPACE GROUP OPERATIONS (unitary + spin-flip) = 16`。
- `symmetry_ssg=0`:旧路径 `SPACE GROUP OPERATIONS = 8`(仅磁子群),`analyze_spin_space_group_nspin2` **未进入** → 零回归确认。
- **关键判据认知**:CsCl-AFM(bcc Fe,mag ±5)得到 flip=0——因为其两子晶格由**体心平移 (½,½,½) = 自旋反平移**关联,而非点/空间操作,my 检测正确地不误报(anti-translation 属 Increment 3 之外的"自旋平移群",本期不处理)。**altermagnet 的判据是子晶格由旋转关联**(rutile 的 4₂ 螺旋/对角镜面),正是本检测捕获的 flip 陪集。收益:旧 8 op → SSG 16 op ≈ **2×**。

原始设计如下:

**数据结构** `source/source_cell/module_symmetry/symmetry.h`(纯新增字段,仿 `gmatrix_anti` 块):
```cpp
// (nspin=2, collinear SSG) UNITARY spin-flip coset [C2_perp ‖ g]:
// spatial ops g that map the up-sublattice onto the down-sublattice (mag[iat] = -mag[g(iat)]).
// Distinct from the antiunitary gmatrix_anti[] (no complex conjugation here).
ModuleBase::Matrix3 gmatrix_flip[48];
ModuleBase::Matrix3 kgmatrix_flip[48];
ModuleBase::Vector3<double> gtrans_flip[48];
int nrotk_flip = 0;
bool spin_flip_nspin2 = false;
std::vector<std::vector<int>> isym_rotiat_flip_;  // atom map for flip ops
```

**检测函数** `symm_magnetic.cpp` 新增 `analyze_spin_space_group_nspin2(const Atom*, const Statistics&)`:
- 仿 `analyze_magnetic_group_nspin4`(`:81-189`),但作用在**完整化学空间群**上(标量磁矩 `atoms[it].mag[ia]`)。
- 前置:当 SSG 门控命中时,`symm_analysis.cpp:123-126` 分支改为**先跑 plain `getgroup`**(得到完整化学群,填 `gmatrix`/`isym_rotiat_`),再调此函数;它:
  1. 保留 `mag[iat]==mag[g(iat)]` ∀ 的 op → 幺正子群(compact 到 `gmatrix[0..nrotk)`,与旧 relabel 路径产出同一子群);
  2. 捕获 `mag[iat]==-mag[g(iat)]` ∀ 的 op → 翻转陪集 `gmatrix_flip[]`,`nrotk_flip`,`isym_rotiat_flip_`;设 `spin_flip_nspin2=true`;
  3. 其余 op 拒绝。
- 打印:`SPIN-FLIP COSET OPERATIONS (nspin=2 SSG)` = `nrotk_flip`,以及合并 SSG 阶数 `nrotk + nrotk_flip`。

**调用点** `symm_analysis.cpp:123`:
```cpp
if (PARAM.inp.symmetry_ssg && !pricell_loop && nspin == 2) {
    this->getgroup(...full chemical group...);   // 先拿完整化学群
    this->analyze_spin_space_group_nspin2(atoms, st);
} else if (!pricell_loop && nspin == 2) {
    this->analyze_magnetic_group(atoms, st, nrot_out, nrotk_out);  // 旧路径不动
} else { getgroup(...); }
```

**输入旗标** `input_parameter.h` + `read_input_item_output.cpp`(或 system 段):`int symmetry_ssg = 0;`,跟随现有 `symmetry` 旗标模式。

**验收(Increment 1)**:编译通过;旗标默认 0 时任一现有测例 running.log 逐位不变;altermagnet 测例开旗标后 running.log 打印出非零 `nrotk_flip`(手工核对 RuO₂ 应有把两个 Ru 子晶格互换的 C4/镜面操作)。此增量**不改密度、不改 k**,故结果不变,纯诊断。

### Increment 2 — 密度对称化耦合 up↔down

`symm_rho.cpp` `symmetrize_rho`:nspin==2 且 `spin_flip_nspin2` 时,不再对 `rho[0]`、`rho[1]` 独立 `begin()`,改为耦合:
- 幺正子群 op:各自道空间对称化(现有 `psymmg`);
- 翻转陪集 op:空间旋转 `g` 后 **`rho_up ↔ rho_down` 互换**再累加。
- 实现:新增 `psymmg_flip(rhog_up, rhog_down, ...)`,仿 `psymmg_soc` 的多分量耦合结构(`symm_rhog.cpp`),但混合矩阵是 2×2 置换(子群=I,陪集=σx 交换),无复共轭。

### Increment 3 — k 点约化纳入翻转陪集

`k_vector_utils.cpp:542` 旁新增分支:`spin_flip_nspin2` 时把 `kgmatrix_flip[]` 并入折叠群(index 约定 `j+nrotk ↔ flip[j]`)。IBZ 变小 → k 点减少。密度重构须与 Increment 2 的 up↔down 交换一致(翻转陪集的星成员贡献给相反自旋道)。

## 3. 需要读/改的文件

| 功能 | 文件 | 增量 |
|---|---|---|
| 存储字段 | `source/source_cell/module_symmetry/symmetry.h` | 1 |
| 检测 | `source/source_cell/module_symmetry/symm_magnetic.cpp` | 1 |
| 调用点/门控 | `source/source_cell/module_symmetry/symm_analysis.cpp:123` | 1 |
| 输入旗标 | `input_parameter.h`, `read_input_item_*.cpp` | 1 |
| 密度耦合 | `source/source_estate/module_charge/symm_rho.cpp`, `symm_rhog.cpp` | 2 |
| k 约化 | `source/source_cell/k_vector_utils.cpp:542` | 3 |

## 4. 验证方案(整体)

1. **零回归**:`symmetry_ssg=0` 时,现有 nspin=1/2/4 与 SOC 测例能量、k 点数、running.log 逐位不变(门控保证)。
2. **正确性(主判据)**:altermagnet(RuO₂ / MnTe / CrSb)三档 `nosym`/`symk`(仅 k 约化)/`sym`(全开)总能。`sym` 应与 `nosym` **逐位相同**(PBE 最灵敏)。
3. **收益**:开 SSG 后 IBZ k 点数应比现有 nspin=2 磁性路径明显减少;SCF wall-time(及 HSE 的 `Exx_LRI cal_exx_elec` timer)下降。
4. **物理指纹**:altermagnet 动量依赖自旋劈裂;`sym` 与 `nosym` 能带叠图重合,展示对称强制的零劈裂节面。
5. 若 `sym`≠`nosym`,用 `symk` 隔离到检测/密度/k 哪一层。

## 5. 风险

- Increment 1 零风险(纯诊断 + 门控)。
- Increment 2/3 有物理风险:密度 up↔down 交换与 k 折叠必须严格一致,否则 `sym≠nosym`。以 PBE 逐位比对为闸门,逐增量验证。
- 完整化学群 vs relabel 子群的一致性已论证相等(见 Increment 1 检测第 1 点)。

## 6. 实现顺序

先 Increment 1(本次:存储 + 检测 + 输出 + 门控,编译 + 零回归验证 + altermagnet 打印核对),再 Increment 2(密度),最后 Increment 3(k 约化)。每步以"PBE `sym`==`nosym` 逐位"为闸门。
