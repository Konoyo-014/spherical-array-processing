# CODEX Review B13.1

## Scope

本轮外审覆盖了 `spherical_array_processing/dirac/analysis.py:74`、`spherical_array_processing/room/shoebox.py:18`、`spherical_array_processing/room/fdn.py:66` 这三处 b13 变更核心实现，并回跑了聚焦测试与全量测试。聚焦套件是 `tests/test_dirac.py:28`、`tests/test_room.py:12`、`tests/test_room_fdn.py:72`，全量结果为 **547 passed**。

## DirAC physics

你在 b13 里引入 **normalisation-invariant ψ** 的方向是对的，但原先“先把输入系数统一到 SN3D，再把速度通道除以 `√3`”这一组合，在这个仓库自己的 **SH coefficient** 语义下并不成立。根因不在 IIR 平滑，也不在 STFT，而在于这个包里的 `normalization` 切换是 **系数重标定**，不是把信号重新解释成另一套虚拟麦克风通道。对应修正已经落在 `spherical_array_processing/dirac/analysis.py:159` 到 `spherical_array_processing/dirac/analysis.py:180`。

更具体地说，包内的平面波编码默认从正交归一基出发，再通过 `convert_ambi_normalization` 做系数重标定，见 `spherical_array_processing/ambi/encoder.py:29` 和 `spherical_array_processing/ambi/format.py:94`。在这种语义下，把同一个物理场统一到 **orthonormal FOA coefficients** 再对一阶笛卡尔通道施加 `1/√3`，才会使理想相干平面波满足 **`||I|| = E`**，从而得到 **ψ≈0**。原来的 “SN3D + `1/√3`” 会把理想平面波固定推到 **ψ≈0.1339746**，这不是残余误差，而是标度错位。修正后，`tests/test_dirac.py:44` 不再只检查“足够低”，而是把理想平面波锁到数值零附近；`tests/test_dirac.py:71` 也继续锁住了 ortho / n3d / sn3d 三种声明下的严格一致性。

如果把问题放回 **Merimaa–Pulkki 2005** 和 **Politis 2015** 的语境，结论会更清楚。那些公式假定的是 **B-format / pressure-velocity proxy channels** 的物理配比，而不是任意 SH 系数归一化下都能直接套用的代数形式。也就是说，`1/√3` 本身不是错，它在这个仓库里对应的是 **orthonormal** 或 **n3d** 这一类“偶极向量范数相对 W 恰好多出 `√3`”的内部系数标度；把它直接套到这里的 SN3D 系数上就错了。你观测到的 **ψ≈0.134** 正是这个错位的直接数值签名。

## Shoebox sinc correctness

`_scatter_sinc` 自身的卷积中心和裁剪逻辑是对的。核心索引在 `spherical_array_processing/room/shoebox.py:49` 到 `spherical_array_processing/room/shoebox.py:64`：它先取 `center = round(d)`，再用 `frac = d - center` 构造 `sinc(tap_idx - frac)`，于是核的几何中心确实落在 **`center + frac = d`**。边界上通过 `lo_src`、`hi_src`、`lo_clip`、`hi_clip` 做源区间和目标区间的同步裁剪，没有 wrap-around，数值实验也证实了这一点。

真正的问题出在更上游的预筛选。b13 原实现虽然加了 `_scatter_sinc`，但 `_shoebox_contributions` 仍然按 **rounded sample index** 预先丢弃超窗贡献，于是所有“核中心略微越过末端、但 FIR 支撑仍与缓冲区重叠”的反射，都会在进入 `_scatter_sinc` 之前被错误删除。这个根因已经在 `spherical_array_processing/room/shoebox.py:18` 到 `spherical_array_processing/room/shoebox.py:46` 和 `spherical_array_processing/room/shoebox.py:252` 到 `spherical_array_processing/room/shoebox.py:345` 修掉了。现在 `nearest` 和 `sinc` 走各自正确的 write-mask：前者按四舍五入的落点判定，后者按 **kernel support overlap** 判定。新的回归锁定在 `tests/test_room.py:134`，专门覆盖“中心略超出最后一个采样点，但裁剪后仍应留下尾部能量”的场景。

## Shoebox SH path

`shoebox_sh_rir(interpolation="sinc")` 的方向语义是保住的。关键是 `weights = (amplitudes[:, None] * y).T` 这一层在 `spherical_array_processing/room/shoebox.py:429` 到 `spherical_array_processing/room/shoebox.py:438` 仍然是按“每个 image source 一组 SH 权重”去广播，`_scatter_sinc` 对 `(Q, K)` 权重矩阵的最后一维逐源散射也没有把通道混在一起。因此每个反射依旧共享同一个时间核，只是在所有 SH 通道上乘上该反射自己的 `Y_q(Ω_k)` 权重。

这一点最重要的物理检验是 **W 通道单声道关系**。原有测试 `tests/test_room.py:170` 已经保证最近邻路径下 `W = rir / √(4π)`。我补了一条更关键的 sinc 边界回归 `tests/test_room.py:188`，专门检查当核在缓冲区末端被裁剪时，这个关系仍然成立。当前实现通过这个检验，说明 SH 路径没有在广播或裁剪中破坏通道语义。

## FDN orthogonality tolerance

`1e-6` 的绝对容差没有问题。`spherical_array_processing/room/fdn.py:66` 到 `spherical_array_processing/room/fdn.py:106` 采用的是 `max(abs(MMᵀ - I))`，而标准 `float64` QR 得到的随机正交矩阵通常在 **1e-15** 量级，哪怕输入是 `float32` 再转 `float64`，偏差也大多还在 **1e-7** 左右，所以 `tests/test_room_fdn.py:95` 的“QR 生成正交矩阵应被接受”是稳的，不是侥幸过线。这个 gate 的紧度足以挡掉真实的非正交矩阵，同时不会误伤正常构造出的正交矩阵。

## Regressions and remaining quirk

从测试面看，我没有看到 b12 到 b13 的新增回归还残留在当前树上。聚焦套件通过，全量套件也通过，结果是 **547 passed**。这说明 b13 想解决的三件事里，**FDN orthogonality validation** 已经站稳，**shoebox fractional delay** 在边界处补上了最后一个 correctness 缺口，**DirAC ψ** 也从“跨声明一致但物理零点错位”修到了“跨声明一致且理想平面波零点正确”。

还剩下一点值得你在后续版本里用文档讲清楚的地方。`spherical_array_processing/ambi/encoder.py:53` 到 `spherical_array_processing/ambi/encoder.py:56` 现在仍然容易让读者把 `normalization="sn3d"` 理解成“直接 AmbiX/B-format 数值约定”，但这个包内部实际上采用的是 **同一物理场的 SH 系数归一化切换**。这不再是 b13 的阻塞 bug，因为 DirAC 现在已经按这个内部语义自洽了；不过它确实是这次误导推导的来源，属于 **非阻塞的 ergonomics / documentation quirk**。

## Verdict

**TAG**。

这次外审里真正需要动手修的 correctness 问题有两处，现在都已经落地并通过回归：DirAC 的内部标度修正在 `spherical_array_processing/dirac/analysis.py:159`，shoebox sinc 的尾端 overlap 筛选修正在 `spherical_array_processing/room/shoebox.py:18`。就 b13 关心的这三项二级问题来说，当前树已经达到可通过状态。
