# b10 External Review

## Changes

这次 review 落地了两处实质性修复。第一处在 `spherical_array_processing/hrtf/sofa.py:25` 和 `spherical_array_processing/hrtf/sofa.py:268`。`save_sofa` 之前写出的文件能被本仓库自己的 `load_sofa` 正常回读，但对外部 SOFA 校验器并不完全干净。我补上了 **`Version = "2.1"`**，也补上了 `SimpleFreeFieldHRIR` 约定里被视为 mandatory 的全局属性 **`AuthorContact`**、**`Organization`** 和 **`ListenerShortName`**。紧接着，在 `spherical_array_processing/hrtf/sofa.py:317` 我把 **`Data.Delay`** 改成了 convention 对齐的 `float64`，并且在 `spherical_array_processing/hrtf/sofa.py:339` 去掉了 **`ListenerUp`** 上不属于该 convention 的 `Type` / `Units` 属性。修完以后，导出的文件可以被第三方 `sofar` 库直接 `read_sofa()` 并通过 `verify()`，同时也能被 `netCDF4.Dataset()` 打开。

第二处在 `spherical_array_processing/room/fdn.py:199`。`fdn_sh_tail` 的 `basis` API 明确支持 **`"real"`** 和 **`"complex"`**，但旧实现无论哪种基底都在返回前强制转成 `float64`，这会把 **complex SH basis** 的虚部直接丢掉。现在它会在 `spherical_array_processing/room/fdn.py:297` 根据结果类型返回 `complex128` 或 `float64`，和 API 语义一致。

为了把这些结论钉死到测试里，我在 `tests/test_hrtf_sofa_writer.py:51` 增加了对 SOFA mandatory globals、`Version`、`ListenerUp` 属性约定以及 `Data.Delay` dtype 的断言；在 `tests/test_room_fdn.py:114` 增加了 **complex basis 必须返回 complex 输出** 的回归测试。

## Findings

SOFA writer 的核心数据集名字是对的。当前写法使用 **`Data.IR`**、**`Data.SamplingRate`**、**`Data.Delay`**、**`SourcePosition`**、**`ReceiverPosition`**、**`ListenerPosition`**、**`ListenerView`**、**`ListenerUp`** 和 **`EmitterPosition`**，这与 `SimpleFreeFieldHRIR` 约定一致。`SourcePosition` 使用 **`Type="spherical"`** 和 **`Units="degree, degree, metre"`** 也是对的。`ReceiverPosition` 写成 **`(2, 3, 1)`** 也符合静态双耳接收器在该约定中的常见布局，第三方 `sofar` 默认对象写出的也是这个 shape。

FDN 部分的核心数学关系是对的。`_fdn_run` 在 `spherical_array_processing/room/fdn.py:66` 先读出每条 delay line 的延迟端样本，再把它作为当前 line output，随后计算 **`mixing @ (decay_gains * delayed)`** 并写回当前 ring-buffer 写指针。这个顺序对应的是“先取出 `n - L_i` 时刻的状态，再生成要在未来 `L_i` 样本后被读到的新状态”，没有 off-by-one。用最小 toy case 数值展开以后，延迟一拍和两拍的线都按预期开始出声。

衰减系数公式 **`α_i = 10^(-3·L_i / (RT60·fs))`** 也是对的。因为 `L_i / fs` 是单次回路经过该 delay line 的时间，要求经过 `RT60` 秒衰减到振幅 `10^-3`，直接得到 `α_i = 10^{-3 (L_i/fs) / RT60}`，也就是当前实现。换句话说，这个公式对应的正是 **振幅 -60 dB**，也就是能量 **-120 dB** 的经典 RT60 定义。

混合矩阵和衰减的位置我认为也放对了。当前实现是 **先做每条线自己的损耗，再做正交 mixing**。这对应的是每条反馈支路有各自的衰减器，然后这些衰减后的 line outputs 进入 Jot 式反馈矩阵。若把 `α` 放到 mixing 之后，物理含义会变成“按接收线而不是按发射线施加损耗”，对于不同 delay 的线不再是同一个模型。

`fdn_sh_tail` 的空间构造在统计意义上是合理的。每条 FDN line 被分配到球面上的一个独立随机方向，再按 plane-wave 的 SH 系数编码并叠加，这会自然给出非零的高阶方向通道。更细一点说，在本项目的 **orthonormal SH** 归一化下，W 通道的系数恒为 **`1/sqrt(4π)`**，因此它和 line outputs 的单通道和只差一个固定归一化因子，而不是额外的方向性偏差。这和“扩散尾声场在一阶以上通道存在统计波动、均值接近零”这一预期一致。

## Flagged Items

还有两个点我建议作为后续 ergonomics 事项记录下来。第一，`HRTFDataset` 目前没有承载 **`Data.Delay`** 的字段，所以 `load_sofa` 读到的非零延迟信息不会在对象里保留，随后 `save_sofa` 只能写回全零 `Data.Delay`。这不是 b10 新引入的 regression，因为 b9 的 reader 也没有把它纳入数据模型，但它意味着“外部 SOFA 文件完整元数据往返”还不是百分之百保真。

第二，`fdn_sh_tail` 的 **`seed=0`** 默认值让函数在不显式传参时总是生成同一组散射方向。这样做对可复现性很好，但从 API 直觉上看，有些用户会预期默认行为是“每次都不一样”，也就是 `seed=None`。我没有在这次 review 里改它，因为这更像产品选择而不是 correctness bug。

## Validation

我先跑了聚焦测试，再跑了全量套件。当前结果是 **`tests/test_hrtf_sofa_writer.py` + `tests/test_room_fdn.py` 共 23 通过**，随后 **全量 510 通过**。在仓外兼容性方面，我用第三方 **`sofar`** 实际读取 `save_sofa` 产物，`read_sofa()` 和 `verify()` 均通过；同时也用 **`netCDF4.Dataset()`** 成功打开了导出的 `.sofa` 文件。Matlab SOFA API 我无法在这个环境里直接运行，所以这一项我只能给出“按 convention 字段和第三方 Python 工具链看是兼容的”的结论。

## Verdict

在这两个修复合入之后，我给 **TAG**。我没有看到相对 b9 的回归。SOFA writer 现在对外部工具更稳，FDN 主体数学顺序是对的，而 `fdn_sh_tail` 的 complex-basis 截断 bug 也已经补掉了。
