# CODEX Review B9.2

## Pytest summary

按要求运行 `PYTHONPATH=. python3 -m pytest -q`，结果是 **487 passed, 1 warning**。这和预期的 **487 / 487 passing** 一致。唯一警告来自 `tests/test_rafaely_example_scripts.py::test_run_rafaely_ch1_pn_script`，内容是 `fig.tight_layout()` 的版面警告，不影响本次 b9 修复核查结论。

## sdist verification

按要求运行 `tar tzf dist/spherical_array_processing-0.4.0b9.tar.gz | grep -E "(ambi/uhj|ambi/intensity|test_ambi_uhj|test_ambi_intensity)"`，得到 **四个匹配项**，分别是 `spherical_array_processing/ambi/uhj.py`、`spherical_array_processing/ambi/intensity.py`、`tests/test_ambi_uhj.py` 和 `tests/test_ambi_intensity.py`。进一步对比后，这四个文件在 `sdist` 中的内容与当前工作树 **一致**，说明重建后的 b9 源码包确实带上了当前修正版本。

## Source re-read

重新检查 `spherical_array_processing/ambi/uhj.py` 后，`uhj_encode` 仍然保留了 **`0.5 * (...)`** 的半增益实现，左右声道分别是 `0.5 * (real_sum + hilbert_part + real_diff)` 和 `0.5 * (real_sum - hilbert_part - real_diff)`。同时，Hilbert 正交项里的 **`0.5098604`** 也仍然是修正后的标准系数，没有回退成错误值。

重新检查 `spherical_array_processing/ambi/intensity.py` 后，模块级说明和 `doa_from_intensity` 的函数说明都一致写明：**active intensity points toward the source**。也就是说，b9 修正后的方向性表述仍然在，模块叙述和函数文档没有再次出现互相矛盾。

## CHANGELOG spot-check

`CHANGELOG.md` 里的 **Fixed (codex b9 review)** 段落和实际文件状态能够对上。关于 UHJ 的条目，对应到了 `uhj.py` 中保留的 **`(S ± D)/2`** 半尺度修复、**`0.5098604`** 系数修复，以及 `tests/test_ambi_uhj.py` 中仍然存在的 `test_y_only_uses_classical_half_gain` 与 `test_matches_classical_reference_formula_on_tone` 两个回归测试。关于 intensity 的条目，对应到了 `intensity.py` 中仍然存在的 **toward** 表述。关于 standing-wave 测试的条目，对应到了 `tests/test_ambi_intensity.py` 中仍然存在的 **bin-centered** 载频设置 `freq_hz = 3 * 16000.0 / T`。

## New findings

**nothing**。

## Final verdict

**TAG**。
