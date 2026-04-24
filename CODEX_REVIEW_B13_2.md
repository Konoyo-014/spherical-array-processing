# CODEX Review B13.2

## Pytest

我按要求运行了 `PYTHONPATH=. python3 -m pytest -q`。结果是 **547 passed, 1 warning**，总耗时 `8.32s`。这说明你提到的两处真实正确性修复没有在当前工作树里回退，整个现有测试面也仍然稳定。唯一的 warning 来自 `tests/test_rafaely_example_scripts.py::test_run_rafaely_ch1_pn_script`，内容是 Matplotlib 的 `tight_layout` 无法为所有坐标轴装饰留出足够边距。这个 warning 与本次 DirAC 和 shoebox 修复无关，也不构成签核阻塞。

## Sdist Verification

我按要求运行了 `tar tzf dist/spherical_array_processing-0.4.0b13.tar.gz | grep -E "(dirac/analysis|room/shoebox|room/fdn)"`。归档中命中了 **3** 个目标路径，分别是：

```text
spherical_array_processing-0.4.0b13/spherical_array_processing/dirac/analysis.py
spherical_array_processing-0.4.0b13/spherical_array_processing/room/fdn.py
spherical_array_processing-0.4.0b13/spherical_array_processing/room/shoebox.py
```

这说明你重新打包的 `sdist` 至少在文件层面已经包含了本轮需要复核的实现文件，没有出现漏包或旧包残留的迹象。

## Re-read Findings

我重新检查了 `spherical_array_processing/dirac/analysis.py`。当前实现先把输入 FOA 系数在内部**规范到 orthonormal**，随后再对笛卡尔一阶通道施加 `1/√3` 的速度缩放，并据此计算强度向量和能量。这和你描述的修复方向一致，也和该包以系数重标定为核心的 normalisation 约定一致。按这套写法，纯相干平面波满足 `||I|| = E`，于是 **ψ 可以回到 0**，此前那种把内部规范化指向 SN3D 的残余比例误差没有再出现。

紧接着我复查了 `spherical_array_processing/room/shoebox.py`。当前文件里存在新的 **`_write_mask`** 辅助函数，并且它明确把 `nearest` 和 `sinc` 两种写入策略区分开来。对 `sinc` 模式，mask 不是再看最近采样点是否落进缓冲区，而是检查**截断后的分数延迟核是否仍与输出缓冲区重叠**。这正是你描述的根因修复，因此那类“核中心略微越界，但支持域仍然覆盖最后几个样本”的合法贡献现在会被保留下来。`tests/test_room.py` 里的回归测试 `test_sinc_keeps_kernel_that_overlaps_buffer_end` 也仍然在位，并且和现有实现严格对应。

最后我复查了 `CHANGELOG.md`。`dirac_analysis` 条目现在明确写的是**内部 canonicalises the input to orthonormal FOA coefficients**，不再把内部规范化写成 SN3D；后面的 `Fixed (codex b13 review)` 小节也准确记录了两处修复的技术实质，包括 DirAC 内部 canonicalization 方向修正，以及 shoebox sinc 预过滤 mask 从“最近采样点判定”改为“支持域重叠判定”。文字描述和当前代码实现是对齐的，没有发现表述漂移。

## New Findings

我这次复核里**没有发现新的正确性问题**。当前能看到的唯一非绿项仍然只是那个和绘图排版相关的测试 warning，它不影响数值语义，也不影响这次 b13 的发布判断。

## Verdict

我的最终结论是 **TAG**。
