# b14 A1 review

**Verdict: TAG**

我把 `spherical_array_processing/ambi/intensity.py` 和 `spherical_array_processing/dirac/analysis.py` 逐行核对了一遍，也跑了定点测试和全量测试。结论是这次抽取基本是**纯提取**，`dirac_analysis` 的核心物理量定义没有发生行为漂移，`intensity_vector(physical_units=True)` 也确实覆盖了 DirAC 内部正在使用的那条 canonical pressure-velocity 路径。

## Pure extraction

旧版 `dirac_analysis` 的逻辑可以拆成两个动作。前一个动作是把输入归一化先转到 **orthonormal**。后一个动作是按 **ACN → Cartesian** 的顺序取出 `W, X, Y, Z`，同时对一阶速度项乘上 `1/√3`。新实现里，这两个动作被完整封装进 `_canonical_foa_pv`，位置在 `spherical_array_processing/ambi/intensity.py:57`，而 `dirac_analysis` 现在只是在 `spherical_array_processing/dirac/analysis.py:158` 调这个 helper。

更关键的是，helper 内部的通道映射和旧代码完全一致。`_acn_to_cartesian` 在 `spherical_array_processing/ambi/intensity.py:35` 先读取 `q=1` 为 `Y`、`q=2` 为 `Z`、`q=3` 为 `X`，然后返回顺序是 `(w, x, y, z)`，也就是最终给出 `(W, X, Y, Z)` 的 **Cartesian** 排列。紧接着 `_canonical_foa_pv` 在 `spherical_array_processing/ambi/intensity.py:93` 到 `spherical_array_processing/ambi/intensity.py:95` 对 `x, y, z` 统一乘 `1/√3`。这和旧版 `vx = foa[:, 3, :] / √3`、`vy = foa[:, 1, :] / √3`、`vz = foa[:, 2, :] / √3` 是同一件事。

我还额外做了一个**精确数值比对**，直接拿随机复数张量、不同 `coeff_axis`、不同 `normalization` 去比较“旧内联实现”和 `_canonical_foa_pv` 的输出。结果是 **exact match**，不是只到 `allclose`，而是逐元素完全相等。这意味着在当前 `dirac_analysis` 的调用方式下，这次抽取可以视为无行为变化。

## `physical_units=True` 是否覆盖 DirAC 语义

这一点也是成立的。`intensity_vector` 在 `spherical_array_processing/ambi/intensity.py:152` 进入 `physical_units=True` 分支后，直接调用同一个 `_canonical_foa_pv`，然后用 `W* · (v_x, v_y, v_z)` 形成强度。`dirac_analysis` 内部也是同样先拿到 `(w, vx, vy, vz)`，再在 `spherical_array_processing/dirac/analysis.py:164` 到 `spherical_array_processing/dirac/analysis.py:170` 形成 `I = Re{p* v}` 和 `E = 0.5(|p|² + |v|²)`。

这就解释了为什么它确实是 DirAC 入口的**超集**。DirAC 用的是其中的 pressure-velocity canonicalisation，再往上叠加平滑、方向归一化和 diffuseness 计算。`intensity_vector(physical_units=True)` 暴露的是这个 canonicalisation 之后、但尚未做 DirAC 时域平滑的原始物理强度，所以语义上完全对齐，而且覆盖范围更底层。

测试也已经把这个关系钉住了。`tests/test_ambi_intensity.py:98` 的新回归测试验证了 `physical_units=True` 和共享 helper 构出来的内部强度逐点一致，`tests/test_ambi_intensity.py:124` 则验证了平面波满足 **`||I|| = E`** 这个 textbook invariant。与此同时，`tests/test_dirac.py:44` 继续保证 coherent plane wave 的 diffuseness 收敛到零，这和新的物理单位路径是相互咬合的。

## Backward compatibility

`intensity_vector` 默认参数仍然是 `physical_units=False`，定义位置在 `spherical_array_processing/ambi/intensity.py:104`。默认分支没有走 helper，而是继续保留历史上的 **coefficient-space intensity** 路径：先转到 orthonormal，再直接计算 `Re{W* · (X, Y, Z)}`，没有 `1/√3` 这一步。这一点在 `spherical_array_processing/ambi/intensity.py:161` 之后的分支逻辑里很清楚。

我也单独把当前默认分支和“b14 之前的内联公式”做了逐元素精确比对，结果同样是 **exact match**。所以从实现上看，默认调用 `intensity_vector(...)` 保持了向后兼容。

不过这里有一个小的测试覆盖结论需要说清楚。当前测试集里，确实有不少旧测试仍然在走默认分支，比如 `tests/test_ambi_intensity.py:43`、`tests/test_ambi_intensity.py:80` 和 `tests/test_ambi_intensity.py:152`，它们间接说明默认行为没有被破坏。只是目前**还没有一条专门的回归测试**，直接把默认输出和历史 coefficient-space 公式逐点钉死。这个缺口不构成实现错误，但如果后续还会继续整理 intensity API，我建议补一条显式的 backward-compat regression test，让“默认值保持老语义”这件事从推断变成被测试锁死的契约。

## Regression result

我按要求跑了 `tests/test_ambi_intensity.py` 和 `tests/test_dirac.py`，结果是 **25 passed**。随后跑了全量套件，结果是 **549 passed, 1 warning**。这个 warning 来自 `tests/test_rafaely_example_scripts.py` 触发的 Matplotlib `tight_layout` 提示，和这次改动没有关系。

综合来看，b14 A1 的核心改动是干净的。共享 helper 的引入把 pressure-velocity canonicalisation 收敛成了**单一真源**，同时没有改变 `dirac_analysis` 的行为，也没有破坏 `intensity_vector` 的默认历史语义。因此这次审阅给出 **TAG**。
