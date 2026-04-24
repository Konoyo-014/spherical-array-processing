# CODEX review for b7, pass 2

## Pytest summary

我先按你的要求运行了 `PYTHONPATH=. python3 -m pytest -q`。在这次二审开始时，仓库确实是 **426 passed, 381 warnings**。紧接着，我在做边界输入抽查时发现了一个新的真实问题：`room.convolve_sh_to_sh` 对畸形输入的防御还不完整，标量 `sh_signal` 会漏出底层的 `ZeroDivisionError`，一维输入则会给出误导性的轴配置报错，而不是清晰的 API 级 `ValueError`。这个问题已经在 `spherical_array_processing/room/reverb.py` 内联修复，并且我在 `tests/test_room_reverb.py` 增加了回归测试。

修复后我重新跑了定向测试和全量回归。现在 `tests/test_room_reverb.py` 是 **13 passed**，全量套件是 **428 passed, 381 warnings**。warnings 没有新增风险点，仍然是 `matplotlib` / `pyparsing` 的第三方弃用提示，以及一个示例脚本里的 `tight_layout` 提示，不构成这次 b7 的发布阻塞项。

## sdist verification

我执行了你给的检查命令，也在补丁落地后重建了一次 `dist/spherical_array_processing-0.4.0b7.tar.gz`，避免源码和产物脱节。重建后的 sdist 里，`spherical_array_processing/ambi/io.py`、`spherical_array_processing/ambi/nfc.py`、`spherical_array_processing/room/reverb.py`、`tests/test_ambi_io.py`、`tests/test_nfc_hoa.py` 和 `tests/test_room_reverb.py` 这六个目标文件都在，路径匹配数也是六个。与此同时，我还从 tarball 里直接抽查了 `room/reverb.py` 和 `test_room_reverb.py`，确认这次新增的输入验证修复和回归测试都已经进入发行包。

## Re-read of the three b7 source files

我把 `spherical_array_processing/ambi/io.py`、`spherical_array_processing/ambi/nfc.py` 和 `spherical_array_processing/room/reverb.py` 又从头到尾通读了一遍。你上一次让我修的几处关键问题都还在，而且实现没有回退。

`convolve_mono_to_ambi` 仍然使用 `np.moveaxis(out, (-2, -1), (ax_t, ax_t + 1))` 恢复用户的轴语义，所以一维 `axis=0` 和非末尾时间轴这两类先前出错的场景都还被正确处理。`convolve_sh_to_sh` 仍然先把输入标准化到 `batch_axes + (Q, T)` 再沿最后一轴卷积，因此高维 batched 输入的修复也是完整保留的。

`read_ambix_wav` 和 `write_ambix_wav` 仍然共享同一个 `_validate_axis_layout`，所以非法 `axis` 字面量不会再静默落到错误分支。`write_ambix_wav` 也仍然把 `np.asarray(data_tq)` 直接交给 `soundfile.write`，没有再强行转成 `float32`，因此 `subtype="DOUBLE"` 的双精度保真修复仍然成立。

`nfc_hoa_distance_filter` 这边，`c > 0` 的参数校验还在，docstring 里关于高频渐近行为的表述也仍然是正确的 `**(R/d) · exp(-ik(d-R)) + O(1/k)**` 语义，没有回到之前那种错误的“每阶幅度滚降”说法。

## CHANGELOG spot-check

我复核了 `CHANGELOG.md` 里 `Fixed (codex b7 review)` 小节的原有五条。它们和我在第一轮真实做过的修改是一一对应的，没有夸大，也没有把没做的事情写进去。关于 `convolve_mono_to_ambi` 的轴恢复、`convolve_sh_to_sh` 的高维输入处理、`read_ambix_wav` / `write_ambix_wav` 的 `axis` 校验、`write_ambix_wav` 的双精度保真，以及 `ambi.nfc_hoa_distance_filter` 的 `c > 0` 校验与文档修正，这些描述都准确。

因为这次二审我又发现并修掉了一个新的真实输入验证问题，所以我额外给这个小节补了一条，明确说明 `convolve_sh_to_sh` 现在会对标量和一维畸形输入抛出干净的 `ValueError`，不再泄漏底层异常。

## New findings

这次二审里我只发现了一个新的真实问题，而且已经修完：`convolve_sh_to_sh` 对少于两个轴的 `sh_signal` 缺少前置验证，导致标量输入会触发 `ZeroDivisionError`，一维输入会落到误导性的轴错误。这个问题现在已经被修复，并且有新的回归测试锁住。

除此之外，我没有再看到新的发布前阻塞项。剩下能提的只有一个纯人体工学层面的观察：全量测试中的 warnings 主要来自第三方库的弃用提示，和 b7 本身的功能正确性无关。如果后续想把 CI 输出做得更干净，可以单独安排一次依赖升级或 warning 过滤整理，但这不影响当前签发判断。

## Final verdict

在这次新增验证修复已经入树、测试重新全绿、sdist 也重建并复核完成的前提下，我给 b7 的最终结论是 **TAG**。
