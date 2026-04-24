# Codex Review for b12

## Verdict

**TAG**。

b12 针对 b11 审计里那四个最高优先级问题的修复，经过代码通读、聚焦数值验证和全量回归之后，我认为已经**实质闭环**。`diffuseness.intensity_vectors_from_foa` 的方向语义现在和包内 ACN 约定一致，`dirac.dirac_analysis` 的默认 `coeff_axis=-2` 确实匹配 `sap.stft.stft` 产出的标准 `(F, Q, T)` 轴布局，`sap.plotting` 的惰性导入在“无 matplotlib”模拟环境里成立，新的 **`jy`** 默认在低阶 `n <= 20` 上和 **`sakurai`** 数值等价到测试容差之内。全量测试在本次复核后为 **537 / 537**。

## What I Checked

我首先直接审阅了这四处改动的实现入口，分别是 `spherical_array_processing/diffuseness/estimators.py:12`、`spherical_array_processing/dirac/analysis.py:70`、`spherical_array_processing/plotting/__init__.py:1` 和 `spherical_array_processing/sh/rotation.py:111`。紧接着我交叉阅读了回归测试，重点看了 `tests/test_independent_audit.py:686` 和 `tests/test_import_contract.py:7`，确认 b12 不是只修了表面形状，而是真的把**语义**锁住了。

对于 **FOA 方向 bug**，我额外跑了独立的平面波 round-trip：用 `spherical_array_processing/ambi/encoder.py:29` 的 `encode_plane_wave` 生成一阶 ACN FOA，再喂给 `spherical_array_processing/diffuseness/estimators.py:12` 的 `intensity_vectors_from_foa`。我扫了多个水平面方位角，包含 `-180°`、`-135°`、`-90°`、`-45°`、`0°`、`30°`、`60°`、`90°`、`135°` 和 `179°`。这些点的平均强度向量方向和编码方向的**角误差都是 0°**，没有再出现旧版那种 90° 旋转。这里的根因也和实现完全对上了，因为 ACN 一阶实基的通道顺序本来就是 **`[W, Y, Z, X]`**，现在 `spherical_array_processing/diffuseness/estimators.py:72` 到 `spherical_array_processing/diffuseness/estimators.py:75` 会先把它重排成笛卡尔顺序 **`(X, Y, Z)`** 再做 `Re{W*·v}` 收缩。

对于 **DirAC 默认轴**，我用 `spherical_array_processing/stft.py:18` 的 `stft` 对 `foa_qt` 做了 STFT，得到标准形状 `(257, 4, 17)`，然后直接调用 `spherical_array_processing/dirac/analysis.py:70` 的 `dirac_analysis(Z, freqs)`，完全不传 `coeff_axis`。结果和显式传 `coeff_axis=1` 的输出**逐元素一致**，方向场最大差值是 `0.0`，diffuseness 最大差值也是 `0.0`。我再把估计的平均 DOA 和编码方向对比，角误差仍然是 **0°**。这说明 `spherical_array_processing/dirac/analysis.py:125` 到 `spherical_array_processing/dirac/analysis.py:129` 的轴规范化逻辑现在和包内 STFT 约定是严格对齐的。平面波的平均 diffuseness 大约是 `0.13397`，这并不表示分析失真，因为模块文档已经在 `spherical_array_processing/dirac/__init__.py:15` 到 `spherical_array_processing/dirac/__init__.py:18` 和 `spherical_array_processing/dirac/analysis.py:85` 到 `spherical_array_processing/dirac/analysis.py:87` 解释了，**正统 DirAC 的 ψ 标定假设一阶通道是 SN3D**；这里用的是包内正交归一化，所以 DOA 正确，但 ψ 会有系统偏差。这属于已文档化的物理量标定差异，不是 b12 回归。

对于 **plotting 惰性导入**，我不只跑了现有的 `tests/test_import_contract.py:7`，还在独立子进程里把 `sys.modules['matplotlib']` 和 `sys.modules['matplotlib.pyplot']` 都置为 `None`，模拟“环境里没有 matplotlib，且任何隐式导入都会立刻炸掉”的场景。在这个条件下，`import spherical_array_processing.plotting` 仍然成功，而真正调用 `sap.plotting.apply_matlab_like_style()` 才抛出 `ImportError`，错误文本正是 `spherical_array_processing/plotting/__init__.py:15` 到 `spherical_array_processing/plotting/__init__.py:24` 里那条安装提示。这说明当前惰性导入栈不只是“顶层 import 不拉起 matplotlib”，而是连直接 `import spherical_array_processing.plotting` 也不会越界触发依赖导入。我把这个缺口补成了回归测试，位置是 `tests/test_import_contract.py:35`。

对于 **`jy` 默认替代 `sakurai`**，我分别比较了 `spherical_array_processing/sh/rotation.py:111` 的 `wigner_small_d`、`spherical_array_processing/sh/rotation.py:161` 的 `wigner_D`，以及旋转矩阵栈 `spherical_array_processing/sh/rotation.py:192`、`spherical_array_processing/sh/rotation.py:263` 和 `spherical_array_processing/sh/rotation.py:344`。在 `n <= 20` 的低阶范围，`wigner_small_d(jy)` 相对 `wigner_small_d(sakurai)` 的最坏绝对差是 **`1.089e-11`**，`wigner_D` 的最坏绝对差是 **`5.734e-12`**，实数和复数 SH 旋转矩阵的最坏绝对差都在 **`1e-15`** 量级。这和 changelog 中“低阶开销可接受、数值更稳”的表述是吻合的，也说明 b12 没有把低阶数值基线偷偷改坏。

## Regressions and Fixes

我没有发现会把 b12 从 **TAG** 打成 **NEEDS-WORK** 的功能性回归。全量测试在当前工作区复跑后从 **536 / 536** 变成 **537 / 537**，新增的 1 个测试正是把“无 matplotlib 环境下 plotting 子模块仍可导入，但调用时才报错”的语义锁住了。对应改动是 `tests/test_import_contract.py:35`。

不过我确实发现了一个**真实但轻量的文档回归**：`README.md:46` 之前还把 `matplotlib` 写成核心依赖，这和 b12 的打包行为已经不一致。我已经把它修正为只列出 `numpy` 和 `scipy` 作为核心依赖，并在 `README.md:47` 明确说明 plotting 需要单独安装 extra。

## Changelog Calibration

`CHANGELOG.md:14` 到 `CHANGELOG.md:22` 对 **FOA channel order bug** 的描述是准确的，而且把 bug 的实际症状写成了“ACN 输入会产生 90° 旋转 DOA”，这一点和我的独立 round-trip 验证完全一致。`CHANGELOG.md:26` 到 `CHANGELOG.md:36` 对 **`coeff_axis`** 和 **`jy`** 默认值变化的破坏性影响也写得比较准，因为这两处确实会改变“未显式传参”的运行语义。

唯一略显不够校准的地方在 `CHANGELOG.md:24` 这个小节标题 **“Changed (breaking defaults)”**。`matplotlib` 移到 extra 这件事写在这里，事实层面没有错，但它更像是**打包/依赖契约变化**，而不是“默认值变化”。如果后续想让 changelog 的信息结构更干净，最好把这一条独立放到 **Packaging**、**Dependencies** 或普通 **Changed** 小节里。这个问题只影响叙事清晰度，不影响发布判断。

## Files Touched During This Review

这次复核里我实际修改了两处。`README.md:46` 到 `README.md:47` 现在和 b12 的可选依赖模型一致，不再误导用户把 `matplotlib` 当成硬依赖。`tests/test_import_contract.py:35` 新增了“屏蔽 `matplotlib` 的子进程导入检查”，把 b12 惰性导入修复的关键契约锁成可执行回归测试。

## Validation Summary

我跑了整套 `pytest -q`，结果是 **537 passed**。我还单独跑了 `tests/test_import_contract.py`、`tests/test_independent_audit.py::TestDiffuseness`、`tests/test_dirac.py` 和 `tests/test_sh.py`，结果是 **27 passed**。除此之外，我补做了四类独立脚本检查：平面波 DOA round-trip、`stft -> dirac_analysis` 默认轴路径、无 matplotlib 子进程导入路径，以及 `jy`/`sakurai` 的低阶数值等价性检查。这些独立检查都支持 **TAG** 结论。
