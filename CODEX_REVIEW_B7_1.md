# CODEX review for b7

## Summary

我把 `spherical_array_processing/ambi/io.py`、`spherical_array_processing/room/reverb.py` 和 `spherical_array_processing/ambi/nfc.py` 全部通读了一遍，并且按要求跑了定向测试与全量回归。现在仓库状态是 **426 / 426 通过**。从数学上看，**NFC-HOA 的 DC 极限 `F_n(0) = (R/d)^{n+1}` 是正确的**。我独立复核的依据是小参量渐近式 `j_n(x) ~ x^n / (2n+1)!!` 与 `y_n(x) ~ -(2n-1)!! / x^{n+1}`，因此 `h_n^{(2)}(x) = j_n(x) - i y_n(x) ~ i (2n-1)!! / x^{n+1}`，比值自然收敛到 `(R/d)^{n+1}`。这和 `spherical_array_processing/ambi/nfc.py:117` 的实现一致，也和直接数值计算 `h_n^{(2)}(kd) / h_n^{(2)}(kR)` 的结果一致。

紧接着看卷积语义时，我确认了一个真实的轴恢复 bug：`convolve_mono_to_ambi` 原先在正轴和部分负轴情况下会把卷积后的时间轴挪回错误位置，最明显的复现是单声道一维输入配 `axis=0` 时输出会变成 `(T_out, Q)`，而正确结果应是 `(Q, T_out)`。这个问题已经修正，并且同类的高维轴语义问题也一并在 `convolve_sh_to_sh` 中处理掉了。

## Changes

**卷积轴语义修复** 已落在 `spherical_array_processing/room/reverb.py:57` 到 `spherical_array_processing/room/reverb.py:69`。这里现在先把原始时间轴标准化成最后一轴，再在卷积后用 `np.moveaxis(out, (-2, -1), (ax_t, ax_t + 1))` 把 **SH 轴插到原始时间轴之前**，把 **新时间轴放回原始时间轴位置之后**。这正好满足你要求的输出形状约定，也同时覆盖了正轴和负轴。对应的回归测试在 `tests/test_room_reverb.py:59` 和 `tests/test_room_reverb.py:69`，一个专门锁住 `axis=0` 的一维输入，另一个锁住非末尾时间轴的高维输入。

**SH-to-SH 卷积的高维正确性** 已落在 `spherical_array_processing/room/reverb.py:106` 到 `spherical_array_processing/room/reverb.py:121`。原实现把通道轴和时间轴挪到 `(0, 1)`，这在二维输入时没问题，但在三维及以上输入时会让 `oaconvolve(..., axes=-1)` 沿着错误的轴工作。现在实现改成先把输入重排为 `batch_axes + (Q, T)`，再把 RIR 广播成 `batch_axes + (Q, T_rir)` 的兼容形状，然后只沿最后一轴卷积，最后把 `(Q, T_out)` 两轴一起挪回去。回归测试在 `tests/test_room_reverb.py:106`。

**AmbiX 读路径的 layout 校验** 已补在 `spherical_array_processing/ambi/io.py:31`、`spherical_array_processing/ambi/io.py:101` 和 `spherical_array_processing/ambi/io.py:170`。之前 `read_ambix_wav` 对非法 `axis` 字面量不会报错，而是静默走到 `channels_last` 分支；现在读写两条路径都共享同一个校验器，错误信息也一致。对应测试在 `tests/test_ambi_io.py:139`。

**AmbiX 归一化换算路径** 我复核后认为是正确的。`soundfile.read(..., always_2d=True)` 返回的是 `(T, Q)`，因此 `read_ambix_wav` 在 `spherical_array_processing/ambi/io.py:117` 到 `spherical_array_processing/ambi/io.py:120` 使用 `convert_ambi_normalization(..., axis=1)` 是对的。写路径也是同理，`write_ambix_wav` 在 `spherical_array_processing/ambi/io.py:186` 到 `spherical_array_processing/ambi/io.py:189` 先把输入标准化成 `(T, Q)`，再沿 `axis=1` 做归一化变换，因此 **读一遍、写一遍的 round-trip 语义是自洽的**。相关测试原本已经覆盖 `orthonormal ↔ sn3d` 往返，位置在 `tests/test_ambi_io.py:47`。

**WAV 写出精度语义** 我顺手做了一个小修正，位置在 `spherical_array_processing/ambi/io.py:191`。原实现不管 `subtype` 取什么都先把数据强制转成 `float32`，这会让 `subtype="DOUBLE"` 的文件虽然容器是双精度，但样本精度已经提前损失。现在写出时直接把数组交给 `soundfile`，让它按 `subtype` 决定落盘格式。对应测试在 `tests/test_ambi_io.py:88`。

**NFC 物理文档修正** 落在 `spherical_array_processing/ambi/nfc.py:19` 到 `spherical_array_processing/ambi/nfc.py:35`，以及 `spherical_array_processing/ambi/nfc.py:101`。这里的核心实现本身没有错，错的是文档里对高频渐近行为的表述。更准确的渐近式是 `h_n^{(2)}(x) ~ i^{n+1} e^{-ix} / x [1 + O(1/x)]`，所以 `F_n` 的高频极限应是 **`(R/d) · exp(-ik(d-R))` 加上 `O(1/k)` 修正**，而不是“带每阶幅度滚降”。这也解释了为什么当 `d < R` 时，低频和高频的幅值基线都大于 1，只是低频端的阶数依赖更强。与此同时，我也把 `d → ∞` 时的描述改成了“所有阶都趋于 0，`n=0` 以 `R/d` 收敛，高阶更快”，这和 `tests/test_nfc_hoa.py:102` 的数值验证一致。顺手还在 `spherical_array_processing/ambi/nfc.py:101` 增加了 `c > 0` 的输入校验，对应测试在 `tests/test_nfc_hoa.py:102`。

## Flagged items

这一版在修补之后，**没有剩下阻塞性的正确性问题**。真正需要持续提醒的只有一个格式层面的事实：**AmbiX WAV 本身不携带可靠的阶数与归一化元数据**，所以 `read_ambix_wav` 只能从通道数推断 `max_order`，而 `normalization` 仍然必须由调用方知道或约定。这不是实现缺陷，而是容器格式的限制。也正因为如此，任何“通道数刚好是平方数”的普通多声道 WAV，都可能被这组帮助函数解释成某个 HOA 阶数的文件。

## Verdict

我给这一版的最终判定是 **TAG**。b7 里新增的三组特性在数学主线上是成立的，`max_order` 不匹配路径的异常信息是干净的，WAV 读写的归一化转换路径是对的，而且我发现的真实问题都已经就地修复并由测试锁住。当前回归结果是 **426 / 426 通过**，没有看到相对于 b6 的回归迹象。
