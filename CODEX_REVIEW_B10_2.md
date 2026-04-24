# b10 Final Sign-off Review

## Pytest

我按要求运行了 `PYTHONPATH=. python3 -m pytest -q`，结果是 **510 passed, 1 warning**。这和预期的 **510/510 通过**一致；唯一的 warning 仍然来自 `tests/test_rafaely_example_scripts.py` 中 matplotlib 的 `tight_layout()` 提示，没有显示出新的功能性回归。

## Sdist Verification

我按要求检查了 `dist/spherical_array_processing-0.4.0b10.tar.gz` 的内容。`tar tzf dist/spherical_array_processing-0.4.0b10.tar.gz | grep -E "(hrtf/sofa|room/fdn|test_hrtf_sofa_writer|test_room_fdn)"` 返回 **恰好四条匹配**，对应 `spherical_array_processing/hrtf/sofa.py`、`spherical_array_processing/room/fdn.py`、`tests/test_hrtf_sofa_writer.py` 和 `tests/test_room_fdn.py`，说明这次 review 相关的源码和回归测试都进了 sdist。

## Fix Integrity

我重新检查了 `spherical_array_processing/hrtf/sofa.py:268` 之后的 `save_sofa` 实现。之前修掉的 **`SimpleFreeFieldHRIR` mandatory globals** 仍然在，`Version`、`AuthorContact`、`Organization` 和 `ListenerShortName` 还在按字符串属性写入；`Data.Delay` 仍然在 `spherical_array_processing/hrtf/sofa.py:317` 以 **`float64`** 零数组写出；`ListenerUp` 在 `spherical_array_processing/hrtf/sofa.py:339` 仍然只创建数据集，没有重新加回不该有的 `Type` / `Units` 属性。这说明第一个修复没有被回滚。

我也重新检查了 `spherical_array_processing/room/fdn.py:295` 之后的 `fdn_sh_tail` 返回逻辑。当前代码先形成 `sh = (y.T @ out_lines) / np.sqrt(n_lines)`，随后在 `spherical_array_processing/room/fdn.py:297` 通过 `np.iscomplexobj(sh)` 分支保留 **`complex128`** 输出，否则才转成 `float64`。这和上一次 review 修掉的 **complex-basis dtype 截断** 问题一致，第二个修复也还在。

## Changelog Spot-check

我 spot-check 了 `CHANGELOG.md:38` 开始的 `Fixed (codex b10 review)` 小节。这里对 `save_sofa` 的描述和当前实现是对得上的，因为 `spherical_array_processing/hrtf/sofa.py:270` 到 `spherical_array_processing/hrtf/sofa.py:317` 的确写入了这些 mandatory globals 和 **`float64 Data.Delay`**，而且 `spherical_array_processing/hrtf/sofa.py:339` 处的 `ListenerUp` 也确实没有 `Type` / `Units`。对 `fdn_sh_tail` 的描述同样准确，因为 `spherical_array_processing/room/fdn.py:297` 到 `spherical_array_processing/room/fdn.py:299` 明确保留了复数输出，`tests/test_room_fdn.py:114` 也确实有对应的回归测试。

## New Findings

**nothing**。这轮 sign-off 里我没有发现新的 correctness 问题、打包缺项或者修复被覆盖的情况。

## Verdict

最终结论是 **TAG**。
