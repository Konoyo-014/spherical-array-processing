# b15 A4 + A5 review

## Verdict

我这次按你列出的七个问题重新做了实现核查、定向测试、`python -m` 运行、wheel 内容检查，以及独立虚拟环境安装验证。结论是 **A4 和 A5 都达到 TAG**。你这次做的不是表面兼容层，而是把 FOA 强度这条线真正**收敛到单一语义源**，同时把“安装后可运行示例”从仓库侧资产提升成了**wheel 内正式交付物**。这两件事都成立，而且我没有测出回归。

紧接着说最关键的判断。**A4 的 wrapper 语义已经和 `ambi.intensity_vector` 对齐**。我不仅跑了你新增的 `tests/test_diffuseness_intensity_wrapper.py`，也额外做了 `np.array_equal(...)` 级别的对照，覆盖了 `normalization` 取 `orthonormal`、`n3d`、`sn3d`，以及 `physical_units` 取 `False`、`True` 的全部组合。结果都是**字节级一致**。默认路径如此，带归一化和物理单位开关时也是如此；FuMa 输入在最后一轴做 `[0, 2, 3, 1]` 重排后再委托给 `intensity_vector`，输出也和直接对 ACN 场调用 canonical API **完全一致**。与此同时，历史错误消息里 `"4 channels"` 和 `"channel_order"` 这两个匹配点也保住了，所以旧的 `pytest.raises(..., match=...)` 合同没有被破坏。

更重要的是，**旧合同没有被 wrapper 回归破坏**。`tests/test_independent_audit.py::TestDiffuseness` 这一组旧的 diffuseness 审计测试在不改测试代码的前提下全部通过，这说明这次“收敛到 canonical implementation”没有偷偷改掉历史张量布局、实数输入容忍度或者 FuMa/ACN 的外部行为。

## A5 check

A5 这部分也过了，而且过得比较扎实。`spherical_array_processing/examples/__init__.py`、`spherical_array_processing/examples/plane_wave_doa.py`、`spherical_array_processing/examples/binaural_em32_to_ears.py` 现在确实是 **随 wheel 一起安装** 的包内示例，不再依赖仓库态 `examples/` 目录才能跑。仓库根下的 `examples/binaural_em32_to_ears.py` 也确实退化成了兼容 shim，所以旧调用方式还在，安装态新调用方式也成立。

我直接跑了两个 `-m` 入口。`python -m spherical_array_processing.examples.plane_wave_doa` 会打印一个合理的 DOA 恢复摘要，在我这里默认输出里 **PWD** 和 **MUSIC** 都给出大约 **1.53°** 的角误差，同时报告扫描网格步长约 **4.01°**。这个量级符合你对 2562 点 Fibonacci grid 分辨率地板的描述。`python -m spherical_array_processing.examples.binaural_em32_to_ears` 也能正常运行，并输出左右耳能量摘要。换句话说，**安装后直接运行示例** 这件事现在不只是文档承诺，而是实测成立。

然后我做了你最在意的 wheel 边界检查。由于当前环境网络是 restricted，标准的 `python3 -m build --wheel` 会在 **isolated build env** 阶段试图联网拉 `setuptools>=69`，所以它在这个沙箱里失败；这是构建环境限制，不是包本身的问题。为了继续完成审查，我改用 `python3 -m build --wheel --no-isolation` 生成 wheel，然后检查 zip 内容。结果确认 wheel 里包含 `spherical_array_processing/examples/__init__.py`、`spherical_array_processing/examples/binaural_em32_to_ears.py` 和 `spherical_array_processing/examples/plane_wave_doa.py`，同时 **没有** 把 `spherical_array_processing/repro`、`spherical_array_processing/regression`、`spherical_array_processing/experimental` 带进去，所以 **A2 的安装边界仍然成立**。

更进一步，我在工作区里建了一个**独立虚拟环境**，从一个不在仓库根目录下的干净工作目录安装刚构建的 wheel，并确认导入到的是 `site-packages` 里的已安装版本，不是源码树。随后在这个环境里再次运行 `python -m spherical_array_processing.examples.plane_wave_doa` 和 `python -m spherical_array_processing.examples.binaural_em32_to_ears`，两者都成功。这说明 **A5 的“用户无仓库 checkout 也能直接跑示例”合同是成立的**，而且示例模块的导入链条确实是自洽的。

## Bonus symmetry

你顺手补的 `simulate_sh_array_response` 对称校验也已经打通了。现在当 `array_type="rigid"` 却传 `dir_coeff=0.5` 时，函数会在仿真层直接抛出清晰的 `ValueError`，报文是 `dir_coeff is only meaningful for array_type='directional', got array_type='rigid'`。当 `array_type="directional"` 且 `dir_coeff=1.5` 时，也会在仿真层直接抛出 `dir_coeff must be in [0, 1] ...`。这和你想要对齐到 equalizer 层的三分支合同是一致的，不再把错误拖到更底层的声学函数里才暴露。

## Regression picture

回归方面，我先跑了定向子集，`tests/test_diffuseness_intensity_wrapper.py`、`tests/test_independent_audit.py::TestDiffuseness`、`tests/test_installed_examples.py`、`tests/test_directional_array_plumbing.py` 合计 **37 passed**。随后我跑了全量 `PYTHONPATH=. python3 -m pytest -q`。这里要说明一个环境细节：如果工作区里**没有**现成 `dist/*.whl`，那么 `tests/test_import_contract.py` 里有两条 wheel 检查会按设计 skip，所以结果是 **582 passed, 2 skipped**；如果先构建 wheel 并保留 `dist/`，那么这两条测试会被激活，我这边拿到的是 **584 passed**。因此，这个环境里不会稳定出现你写的 **583 passed** 这个精确数字，但这不是 A4/A5 代码回归，而是 **是否存在 wheel 工件** 改变了测试计数。

## Ergonomics note

我只看到一个**非阻塞**但值得你顺手处理的发布卫生问题：`pyproject.toml` 里的版本号现在仍然是 `0.4.0b14`。如果这批改动准备作为 **b15** 对外发布，那么打出来的 wheel 目前还是 `spherical_array_processing-0.4.0b14-py3-none-any.whl`。这不会影响我对 A4/A5 的 **TAG** 结论，因为功能和边界都已经成立；但它会影响发布工件和审查批次标签的一致性，最好在正式发版前补上。

总体上看，**A4 已经把 FOA intensity 的双轨实现真正收束成一条 canonical path，A5 也把安装态示例从“仓库附赠”提升成了“发行工件的一部分”**。这两项都达到了你在 `CODEX_SCORECARD_B14.md` 里追求的 9.5+ 修补方向。
