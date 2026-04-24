# spherical-array-processing b15 A1 + A2 审阅结论

## Verdict

**TAG**。

这次复核的核心结论很直接：你为 **A1** 和 **A2** 做的修正都已经真正落到了发行边界和测试行为上，而不只是“仓库态看起来对”。我重新检查了现有 wheel、又走了一遍 **sdist → wheel** 的终端用户路径、还在隔离虚拟环境里验证了导入边界和无 `matplotlib` 的 collection 行为。结果显示，**wheel 现在确实是干净的**，**sdist 再构建出来的 wheel 也仍然干净**，并且 **`include-package-data = false` 没有把 `py.typed` 这种真实运行时元数据打掉**。

唯一需要顺手指出的是一个很小的测试健壮性问题：`tests/test_import_contract.py` 里的 `test_built_wheel_does_not_ship_developer_only_layers` 在 `dist/` 为空时原本会调用 `pytest.skip(...)`，但文件里没有导入 `pytest`。这个问题不会在你当前的仓库态里触发，因为 `dist/` 已经存在；不过在更干净的开发环境里它会从“优雅跳过”退化成 `NameError`。我已经把这个边界补上了。

## A2：wheel / sdist 边界复核

我先按你要求直接检查现成 wheel：运行 `unzip -l dist/spherical_array_processing-0.4.0b14-py3-none-any.whl | grep -E "(repro|regression|experimental)"`，结果是**零匹配**。这说明开发层目录没有被重新带进 wheel。

紧接着我走了你特别点名的 **sdist 再构建** 路径，也就是 `python3 -m pip wheel dist/spherical_array_processing-0.4.0b14.tar.gz --no-deps --wheel-dir=/tmp/sdist-build`。这个过程成功产出新的 wheel。随后对这个新 wheel 再跑同样的 `grep` 检查，结果仍然是**零匹配**。这一步很关键，因为它验证的不是仓库里的“直出 wheel”，而是终端用户从 sdist 安装时真正会经历的那条路径。当前这条路径是自洽的。

更进一步地看，两个 wheel 在包内的载荷规模也一致，`spherical_array_processing/` 前缀下都是 **64 个文件**。同时我检查了 wheel 里的非 `.py` 文件，结果只有 **`spherical_array_processing/py.typed`**。这就说明 **`include-package-data = false` 的副作用目前没有出现**：它确实拦住了原本可能经由 `MANIFEST.in` 漏进来的开发层文件，但没有把当前唯一真正需要随 wheel 分发的类型标记文件去掉。

## 安装态导入边界

为了避免源码树遮蔽 site-packages，我没有在仓库根目录里直接做安装后导入，而是新建了隔离虚拟环境、安装 wheel，然后切到 `/tmp` 再测。这一点很重要，因为如果人在仓库根目录运行 Python，当前目录会优先进入 `sys.path`，你看到的就会是源码树而不是已安装 wheel，结论会被污染。

在这个真正的安装态下，`spherical_array_processing` 以及 `spherical_array_processing.encoding`、`spherical_array_processing.decoding`、`spherical_array_processing.beamforming`、`spherical_array_processing.plotting` 都能正常导入。与此同时，`spherical_array_processing.repro`、`spherical_array_processing.regression` 和 `spherical_array_processing.experimental` 都按预期抛出 **`ModuleNotFoundError`**，错误文本分别是 `No module named 'spherical_array_processing.repro'` 这一类标准缺失提示。也就是说，**终端用户安装 wheel 后看到的稳定层 / 开发层边界已经是干净的**。

## A1：无 matplotlib 时的 plotting 收集行为

我另外建了一个**不安装 `matplotlib`** 的虚拟环境，只装了 `numpy`、`scipy` 和 `pytest`，然后运行你指定的命令：`python -m pytest tests/test_plotting_helpers.py tests/test_rafaely_plot_wrappers.py --co -q`。结果不再是以前那种 collection-time 的 `ModuleNotFoundError` 崩溃，而是直接输出 **`no tests collected in 0.04s`**。

这说明 **A1 的目标已经实现**：这两份 plotting 测试文件在缺失 `matplotlib` 时现在会被 collection 阶段安全绕开，而不是把整个测试发现流程打爆。这里有一个很细的工程语义要提醒你：`pytest --co` 在这种场景下给出的退出码是 **5**，因为它把结果视为“没有收集到测试”，而不是通常运行态里可见的“若干 skip”。所以从“防止 collection crash”的角度看，A1 是通过的；但如果有人在 CI 里机械地把 `pytest --co` 的退出码 5 当作失败，这会形成一个**工具链层面的 ergonomics 小坑**。这个问题不在你的实现逻辑里，而在 `pytest --co` 的语义本身。

## 回归状态

我重新跑了基线命令 `PYTHONPATH=. python3 -m pytest -q`。结果是 **558 passed, 4 warnings**。这比你消息里写的 **557 passing** 多出一个测试，和你这次新增的 wheel 合同测试是吻合的，所以这里不是回归，而是**测试面又向前推进了一格**。

这些 warning 仍然主要来自你有意保留的 developer-only `FutureWarning` 以及一个脚本绘图的 `tight_layout` 提示，不构成 A1 / A2 的回退证据。

## 总评

如果把这次审阅只聚焦在你承诺完成的两项工作，那么我的判断是：**A1 已完成，A2 已完成，而且 A2 不是“仓库里看上去干净”，而是连 sdist 再构建这条最终用户路径都已经干净**。`include-package-data = false` 在你当前的包结构下是**安全的**，因为 wheel 中的关键类型标记 **`py.typed`** 仍然存在，除此之外也没有发现别的真实运行时数据文件被意外剔除。

更有价值的是，这次复核还暴露了两个非常具体的工程细节。其一是**安装态验证必须在仓库树之外进行**，否则源码树会遮蔽 wheel；其二是 **`pytest --co`** 在“全模块级 `importorskip`”的场景下会给出退出码 5，这不会再导致 collection 爆炸，但如果你以后要把这类检查放进自动化脚本，最好显式接受这个退出码，或者改成普通运行态加 `-rs` 来看 skip 报告。

## 本次顺手修复

我只做了一个极小且和本次审阅直接相关的修复：在 `tests/test_import_contract.py:5` 增加了 `import pytest`，从而让 `dist/` 缺失时的 `pytest.skip(...)` 真的能按设计工作。
