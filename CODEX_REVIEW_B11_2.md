# Codex Review B11 #2

## Pytest

我按要求运行了 `PYTHONPATH=. python3 -m pytest -q`。结果是 **533 passed, 381 warnings in 12.29s**，通过数与预期一致。告警全部来自第三方依赖 `matplotlib` / `pyparsing` 的弃用提示，以及一个示例脚本里的 `tight_layout` 用户告警，没有看到新的失败或回归迹象。

## sdist 验证

我按要求运行了 `tar tzf dist/spherical_array_processing-0.4.0b11.tar.gz | grep -E "(ambi/uhj|ambi/translation|hrtf/sofa|hrtf/dataset|test_ambi_translation|test_ambi_uhj|test_hrtf_sofa)"`。归档里确实包含你要核对的路径，也包含新增的 translation 文件与修正后的 UHJ / SOFA 路径：`spherical_array_processing/ambi/translation.py`、`spherical_array_processing/ambi/uhj.py`、`spherical_array_processing/hrtf/dataset.py`、`spherical_array_processing/hrtf/sofa.py`、`tests/test_ambi_translation.py`、`tests/test_ambi_uhj.py`、`tests/test_hrtf_sofa.py`，以及额外命中的 `tests/test_hrtf_sofa_writer.py`。就源码分发包完整性而言，这一项是正常的。

## 复核 #1 修复

我重新读了 `spherical_array_processing/ambi/uhj.py:75`。`_apply_hilbert` 的 FIR 路径仍然是 **full convolution + group-delay trim**，具体实现位于 `spherical_array_processing/ambi/uhj.py:94` 到 `spherical_array_processing/ambi/uhj.py:98`。这说明此前修掉的长度 bug 还在，已经不再依赖 `mode="same"` 的长度行为。

我也复核了 `spherical_array_processing/hrtf/sofa.py:108` 里的 `load_sofa`。`Data.Delay` 的二维情况现在只接受 `(1, 2)` 或 `(M, 2)`，关键校验在 `spherical_array_processing/hrtf/sofa.py:176` 到 `spherical_array_processing/hrtf/sofa.py:181`。如果 `k != M`，这里会直接抛 `ValueError`，所以 review #1 提到的 `(k, 2)` 漏检问题仍然已经被堵住。

紧接着我看了 `spherical_array_processing/ambi/translation.py:1` 和 `spherical_array_processing/ambi/translation.py:150`。模块说明现在把 **小 `kr` 条件** 写成了“reproduced for `|k r| ≪ 1`”并明确强调这是 **leading-order geometric advance**，同时还写明 FFT 零填充“reduces, but does not eliminate” 周期延拓伪影。你在 review #1 里要求软化“exact shift”措辞，这个修正仍然在代码文档里保留着。

## CHANGELOG 对照

这里我发现了一个 **新的文档不一致**。`CHANGELOG.md:20` 到 `CHANGELOG.md:25` 的 Added 段仍然写着 `translate_foa` 在 `|r|·k ≪ 1` 时“**Mathematically exact**”。这和 `spherical_array_processing/ambi/translation.py:23` 到 `spherical_array_processing/ambi/translation.py:28` 里已经被软化的表述不一致，也和 `CHANGELOG.md:56` 到 `CHANGELOG.md:60` 这个 `Fixed (codex b11 review)` 小节里“ earlier \"exact\" wording overclaimed the math ”的结论直接冲突。

这意味着 **代码与修复说明是对的**，但 **同一份 CHANGELOG 的 Added 段仍残留了旧的过强表述**。如果这份 changelog 会随 beta 一起发布，那么读者仍然会从发布说明里读到与实际代码文档相矛盾的数学承诺。

## 结论

我的最终结论是 **NEEDS-WORK**。原因不是实现回退，也不是打包或测试失败，而是 `CHANGELOG.md:20` 到 `CHANGELOG.md:25` 这段发布说明仍然保留了“**Mathematically exact**”这句过强表述，和当前代码文档以及同页的 fixed 小节互相打架。这个问题很小，改掉后我会愿意给 **TAG**。
