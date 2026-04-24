# spherical-array-processing 0.4.0b11 独立审阅报告

## 审阅说明

这份报告完全基于我对当前工作树的**独立阅读、独立判断和独立验证**。审阅过程中我重新跑了整套测试，结果是 **533/533 通过**；同时我也重新执行了打包构建，`sdist` 和 `wheel` 都能成功生成。就工程信号来说，这和 `pyproject.toml:5` 的正式包定义、`.github/workflows/ci.yml:9` 的多 Python 版本 CI，以及 `.github/workflows/ci.yml:47` 的打包校验形成了比较扎实的闭环。按我对当前工作树的本地统计，源码大约是 **14029 行 Python**，测试大约是 **7476 行 Python**，这个比例对科学计算类库来说相当健康。

我下面所有评价，都尽量把判断落到具体实现位置，而不是重复 README 口号。凡是我认为重要的正面结论、风险判断和短板，都会直接指向具体代码位置，例如 `spherical_array_processing/sh/rotation.py:77` 这种格式。

## 整体定位和成熟度

我的总体判断是：这不是学生作业，也还不是工业级成品；它属于**高质量研究级工具包，带明显的预发布工程化气质**。之所以这么定级，先看正面证据。这个包不是“论文代码打个包”那种状态，它有清晰的包元数据和可发布配置，见 `pyproject.toml:5`；它声明了类型支持并带有 `py.typed`，见 `pyproject.toml:34` 和 `spherical_array_processing/py.typed:1`；顶层包做了懒加载，避免一上来就把绘图依赖拉进来，见 `spherical_array_processing/__init__.py:91` 和 `tests/test_import_contract.py:7`；CI 不只是跑单一环境，还覆盖了 `3.9` 到 `3.12` 与 macOS，见 `.github/workflows/ci.yml:15`；README 不是空壳，模块功能、测试方式和发布检查都写得比较完整，见 `README.md:12`、`README.md:111`、`README.md:146` 和 `README.md:159`。这些都说明作者已经在按“可长期维护的科学库”来组织它。

但它又没有到工业级，原因也很具体。其一，工程面虽然扎实，**接口语义还没有完全打磨到“普通用户很难踩坑”的程度**。最典型的例子是 ACN 通道顺序在绝大部分模块中都被当成包级默认约定，`ambi.intensity` 也是按 ACN 解释 FOA，见 `spherical_array_processing/ambi/intensity.py:35`，但 `diffuseness.intensity_vectors_from_foa` 却要求输入顺序是 `[W, X, Y, Z]`，见 `spherical_array_processing/diffuseness/estimators.py:8` 和 `spherical_array_processing/diffuseness/estimators.py:13`。其二，若明天发正式版，文档和示例的**发行工件体验**还不够圆满，因为 README 明写 `examples/` 和 `docs/` 属于仓库开发资产，见 `README.md:5`，而 `MANIFEST.in:58` 与 `MANIFEST.in:59` 又把它们从 `sdist` 中裁掉了。其三，房间、FDN、实验模块里仍保留了一些“研究原型合理、工业产品不够”的取舍，比如 `shoebox_rir` 直接把到达时刻四舍五入到采样点，见 `spherical_array_processing/room/shoebox.py:166`，这对研究基线足够，但对严肃 RIR 仿真仍然偏粗。

如果和公开库横向比较，我会把它放在一个很有意思的位置。和 **MATLAB Politis 原版**相比，这个包在**打包、类型、自动测试、CI、统一 API** 上明显更现代，尤其是把 SRP、ESPRIT、MagLS、AllRAD、SOFA、房间和 DirAC 这些原本分散的研究链路揉成了一个连贯 Python 包，见 `README.md:20` 到 `README.md:31`。但就“作为原始参考实现的权威性”而言，它还没有完全取代 MATLAB 源头的地位，因为这里面还有不少通过 `repro` 层保存的迁移和对照语义，见 `spherical_array_processing/repro/__init__.py:1` 和 `tests/test_repro_layers.py:7`。

和 **spaudiopy** 相比，这个包在**球阵列处理、SH 数学、阵列建模、回归和物理公式显式性**上并不弱，某些地方甚至更“研究工具化”，比如 `Wigner-D` 的高阶稳定后端、SRP/ESPRIT/AllRAD/DirAC/rigid-sphere HRTF/频段 shoebox/测量驱动编码都在一个包里，见 `spherical_array_processing/sh/rotation.py:77`、`spherical_array_processing/doa/esprit.py:138`、`spherical_array_processing/decoding/decoders.py:512`、`spherical_array_processing/dirac/analysis.py:70` 和 `spherical_array_processing/hrtf/rigid_sphere.py:50`。但 spaudiopy 这种成熟公开库的优势通常在于**用户文档、例子、社区记忆和接口惯性**，这部分当前包还没到那个层次。

和 **sound_field_analysis** 相比，我会把当前包放在**更高一档的工程成熟度**。原因不只是功能多少，而是这里的测试面、CI、打包和错误处理明显更系统，见 `.github/workflows/ci.yml:9`、`tests/test_cross_module_consistency.py:1`、`tests/test_extended_audit.py:1` 和 `MANIFEST.in:8`。和 **spharpy** 相比，当前包的特点是**范围更宽**，尤其是在编码、解码、双耳、房间、DirAC 和 HRTF 上明显更完整；但 spharpy 那种偏数学工具箱式的 API 收敛度和概念纯度，这里还略逊一筹，尤其体现在通道顺序、轴约定和一些历史兼容接口的混合上。

## 模块覆盖盘点

### 基础层：`types`、`coords`、`stft`

`types` 是整个包的骨架，`SHBasisSpec`、`SphericalGrid`、`ArrayGeometry` 和 `SpatialSpectrumResult` 这些类型把后续模块串起来了，见 `spherical_array_processing/types.py:22`、`spherical_array_processing/types.py:51`、`spherical_array_processing/types.py:97` 和 `spherical_array_processing/types.py:125`。这一层我给 **核心可用**。它的优点是把角度约定、通道数和元信息都封装得比较清楚；它的问题是 `SHBasisSpec` 默认 `az_colat`，而 `SphericalGrid` 默认 `az_el`，虽然文档已经提示，见 `spherical_array_processing/types.py:26` 和 `spherical_array_processing/types.py:55`，但这仍然是一个长期的易错点。

`coords` 负责球坐标和笛卡尔坐标互转，以及 `az_el` 与 `az_colat` 两套角度系统的转换，见 `spherical_array_processing/coords.py:1`。这一层也是 **核心可用**。它的职责边界很干净，测试也有大量一阶几何恒等式和 round-trip 检查，见 `tests/test_independent_audit.py:74`。

`stft` 是一个薄包装，但意义不小，因为它统一了全包默认的 `F, M, T` 布局，见 `spherical_array_processing/stft.py:1` 和 `spherical_array_processing/stft.py:56`。我同样把它评为 **核心可用**。它没有花哨功能，没有流式状态，也没有多窗口策略；但就离线科学计算工具而言，这个模块已经完整地完成了自己的任务。

### SH 与阵列数学层：`sh`、`acoustics`、`array`

`sh` 是这个包最硬的一块。`basis` 给出复实球谐矩阵、ACN 索引、复实系数互转，见 `spherical_array_processing/sh/basis.py:21`、`spherical_array_processing/sh/basis.py:115` 和 `spherical_array_processing/sh/basis.py:161`；`transforms` 负责直接/逆 SHT，见 `spherical_array_processing/sh/transforms.py:1`；`rotation` 则实现了 `Wigner-d`、`Wigner-D`、复基/实基旋转矩阵和时变旋转，见 `spherical_array_processing/sh/rotation.py:31`、`spherical_array_processing/sh/rotation.py:159`、`spherical_array_processing/sh/rotation.py:261` 和 `spherical_array_processing/sh/rotation.py:388`。这一层我给 **核心可用，而且是全包最成熟的核心之一**。从 `tests/test_cross_module_consistency.py:195` 到 `tests/test_cross_module_consistency.py:260`，再到 `tests/test_extended_audit.py:207`，它不只是“能跑”，而是做了比较扎实的数学合同验证。

`acoustics` 目前几乎全部集中在 `radial.py`，负责球 Bessel/Hankel、刚性球/开放球/心形/一阶指向性阵列的径向模态系数，见 `spherical_array_processing/acoustics/radial.py:14`、`spherical_array_processing/acoustics/radial.py:172` 和 `spherical_array_processing/acoustics/radial.py:286`。我把它评为 **核心可用**。它的功能不是“多”，而是“公式清楚而且打在主干位置上”。测试里既有数值稳定性也有和模拟链的交叉校验，见 `tests/test_radial.py:1` 和 `tests/test_cross_module_consistency.py:48`。

`array` 由三部分组成。`sampling` 提供 Fibonacci、Gauss-Legendre 和 t-design fallback 网格，见 `spherical_array_processing/array/sampling.py:8`、`spherical_array_processing/array/sampling.py:46` 和 `spherical_array_processing/array/sampling.py:74`；`presets` 提供 Eigenmike、四面体、立方体和圆环预设，见 `spherical_array_processing/array/presets.py:65`、`spherical_array_processing/array/presets.py:92`、`spherical_array_processing/array/presets.py:184` 和 `spherical_array_processing/array/presets.py:223`；`simulation` 提供自由场和 SH 模态两条仿真链，见 `spherical_array_processing/array/simulation.py:14` 和 `spherical_array_processing/array/simulation.py:97`。这一层整体也是 **核心可用**。它不算“阵列数据库”式的完整，但作为球阵列研究工具已经足够实战。

### 波束形成、DOA 与统计层：`beamforming`、`doa`、`covariance`

`beamforming` 覆盖了固定波束和自适应波束。固定波束一侧有 cardioid、hypercardioid、supercardioid、MaxEV、Butterworth 和 Dolph-Chebyshev，见 `spherical_array_processing/beamforming/fixed.py:11`、`spherical_array_processing/beamforming/fixed.py:35`、`spherical_array_processing/beamforming/fixed.py:72`、`spherical_array_processing/beamforming/fixed.py:104`、`spherical_array_processing/beamforming/fixed.py:158` 和 `spherical_array_processing/beamforming/fixed.py:181`；自适应一侧有 MVDR 和 LCMV，见 `spherical_array_processing/beamforming/adaptive.py:7` 和 `spherical_array_processing/beamforming/adaptive.py:60`。这一层是 **核心可用**。它的一个加分点是外部数值参照不只是“看起来合理”，而是直接在测试里对齐了 `spaudiopy` 和 `spharpy` 风格的参考数值，见 `tests/test_beamforming_doa.py:65` 和 `tests/test_beamforming_doa.py:72`。

`doa` 的覆盖面其实比很多研究包都宽。它有基于谱图扫描的 `PWD` 与 `MUSIC`，见 `spherical_array_processing/doa/spectra.py:160` 和 `spherical_array_processing/doa/spectra.py:236`；有宽带 `SRP` / `SRP-PHAT`，见 `spherical_array_processing/doa/srp.py:46`；还有闭式的球谐 `ESPRIT` 和源数估计，见 `spherical_array_processing/doa/esprit.py:138` 和 `spherical_array_processing/doa/source_count.py:48`。这一层我同样给 **核心可用**。它的不足不在数学对错，而在“产品层能力”上还没有做成完整的宽带跟踪框架、时序平滑器或阵列校准工作流。

`covariance` 很小，但做得很稳。`Ledoit-Wolf`、`OAS`、前后向平均和对角加载都在一个文件里，见 `spherical_array_processing/covariance/shrinkage.py:10`、`spherical_array_processing/covariance/shrinkage.py:99`、`spherical_array_processing/covariance/shrinkage.py:156` 和 `spherical_array_processing/covariance/shrinkage.py:199`。我把它评为 **核心可用**，而且是那种“模块很小，但没有明显短板”的类型。`tests/test_covariance.py:18` 直接和 `sklearn` 做了 Ledoit-Wolf 数值对照，这是很强的信号。

### 编码、解码与 Ambisonics 层：`ambi`、`encoding`、`decoding`、`dirac`、`diffuseness`、`coherence`

`ambi` 的广度很不错。`format` 做归一化和 ACN/FuMa 转换，见 `spherical_array_processing/ambi/format.py:1`；`io` 处理 AmbiX WAV，见 `spherical_array_processing/ambi/io.py:1`；`nfc` 给出 NFC-HOA 距离补偿，见 `spherical_array_processing/ambi/nfc.py:1`；`uhj` 做两声道 UHJ 编解码，见 `spherical_array_processing/ambi/uhj.py:1`；`encoder` 则给平面波编码，见 `spherical_array_processing/ambi/encoder.py:1`。这一层的大部分我给 **核心可用**。唯一需要降一档的是 `translation`，因为它明确只实现了 **FOA 级别** 的虚拟听点平移近似，见 `spherical_array_processing/ambi/translation.py:30` 和 `spherical_array_processing/ambi/translation.py:51`。所以 `ambi` 这个命名空间整体很强，但内部有一块是 **只有基本功能**。

`encoding` 分成理论径向均衡和实测阵列均衡两块。理论部分包括 Tikhonov、WNG 限幅和统一接口，见 `spherical_array_processing/encoding/radial_filters.py:16`、`spherical_array_processing/encoding/radial_filters.py:69`、`spherical_array_processing/encoding/radial_filters.py:135` 和 `spherical_array_processing/encoding/radial_filters.py:187`；实测部分给出 `regLS` 与 `regLSHD`，见 `spherical_array_processing/encoding/measured_filters.py:43` 和 `spherical_array_processing/encoding/measured_filters.py:137`。这一层我给 **核心可用**。需要说明的一点是，实测均衡这块虽然对用户是统一 API，但内部仍然直接调用了 `repro.politis` 里的函数，见 `spherical_array_processing/encoding/measured_filters.py:34`，这意味着它在产品表面上已经成熟，在实现血缘上仍然带有“研究代码迁移层”的痕迹。

`decoding` 的完成度比我预想的高。`SAD`、`MMD`、`EPAD`、`AllRAD`、`VBAP`、双频带解码和 imaginary loudspeakers 都已经进来了，见 `spherical_array_processing/decoding/__init__.py:1`，以及 `spherical_array_processing/decoding/decoders.py:70`、`spherical_array_processing/decoding/decoders.py:100`、`spherical_array_processing/decoding/decoders.py:139`、`spherical_array_processing/decoding/decoders.py:371`、`spherical_array_processing/decoding/decoders.py:512` 和 `spherical_array_processing/decoding/decoders.py:721`。这一层我给 **核心可用**。如果从公开 Python 空间音频库横向看，它已经不是“只有基础解码矩阵”的水位了，而是进入了相当实战的层级。

`dirac` 是一个 **总体可用，但还没有完全抛光** 的模块。分析、合成和时域管线都在，见 `spherical_array_processing/dirac/analysis.py:70`、`spherical_array_processing/dirac/synthesis.py:36` 和 `spherical_array_processing/dirac/pipeline.py:30`；测试也覆盖了平面波、扩散场和时域渲染，见 `tests/test_dirac.py:28`。我会把它定为 **核心可用**，但在正式版标准下仍然带着一个明显风险点：它没有像 `ambi.intensity_vector` 那样做归一化统一，而是直接在输入系数上计算强度和能量，见 `spherical_array_processing/dirac/analysis.py:126` 到 `spherical_array_processing/dirac/analysis.py:162`。这意味着它的物理解释和归一化鲁棒性还不够统一。

`diffuseness` 是我在 API 层最不满意的模块之一。它包含 `IE`、`TV`、`SV` 和 `CMD` 四个常见估计器，见 `spherical_array_processing/diffuseness/estimators.py:42`、`spherical_array_processing/diffuseness/estimators.py:81`、`spherical_array_processing/diffuseness/estimators.py:130` 和 `spherical_array_processing/diffuseness/estimators.py:174`，从算法广度看并不弱。但它的 FOA 输入顺序居然写成 `[W, X, Y, Z]`，见 `spherical_array_processing/diffuseness/estimators.py:8` 和 `spherical_array_processing/diffuseness/estimators.py:13`，这和包里其他地方坚持的 ACN `[W, Y, Z, X]` 不一致，见 `spherical_array_processing/ambi/intensity.py:41` 和 `spherical_array_processing/dirac/analysis.py:11`。所以我只能把它评为 **只有基本功能，而且还欠一个关键的一致性修正**。

`coherence` 目前只做了两个扩散场相干性工具，见 `spherical_array_processing/coherence/diffuse.py:7` 和 `spherical_array_processing/coherence/diffuse.py:55`。这个模块没有错误，也有用途，但功能面确实窄，所以我把它评为 **只有基本功能**。

### 双耳、HRTF 与房间层：`binaural`、`hrtf`、`room`

`binaural` 是另一个明显强项。`MagLS` 在 `spherical_array_processing/binaural/magls.py:26`，`BiMagLS` 在 `spherical_array_processing/binaural/bimagls.py:59`，一键时域渲染管线在 `spherical_array_processing/binaural/pipeline.py:62`。我把整个 `binaural` 命名空间评为 **核心可用**。它不仅覆盖了核心算法，而且测试已经抓到了一个真实回归，即 `phase_continuation` 过去曾经是 no-op，后来通过 `tests/test_binaural.py:206` 这种回归测试固定下来。这说明这块代码已经进入“持续维护”的状态，而不只是一次性实现。

`hrtf` 由 `dataset`、`rigid_sphere` 和 `sofa` 组成。`HRTFDataset` 把时域和频域接口连起来，见 `spherical_array_processing/hrtf/dataset.py:14`；`rigid_sphere_hrtf` 给出解析球头模型，见 `spherical_array_processing/hrtf/rigid_sphere.py:50`；`SOFA` 读写器则专门支持 `SimpleFreeFieldHRIR`，见 `spherical_array_processing/hrtf/sofa.py:1` 和 `spherical_array_processing/hrtf/sofa.py:22`。我对 `hrtf` 的判断是 **主体核心可用，但 SOFA 支持仍然是有意收窄的最小可用集**。这没有错，只是意味着它还不是那种“拿来就能吞各种 SOFA 生态文件”的成熟 HRTF 工具箱。

`room` 的覆盖相当完整。经典 shoebox、频段反射版本、FDN late tail、卷积式 reverb、ISO 3382 指标都在，见 `spherical_array_processing/room/__init__.py:10`；具体实现分别落在 `spherical_array_processing/room/shoebox.py:177`、`spherical_array_processing/room/banded.py:114`、`spherical_array_processing/room/fdn.py:117`、`spherical_array_processing/room/reverb.py:17` 和 `spherical_array_processing/room/metrics.py:231`。我把它评为 **核心可用**。不过如果把“房间仿真”理解成严肃声学模拟，它还明显停留在**研究基线工具**层面，因为直接取整的采样延迟、标量反射和简化的 FDN 让它更适合算法验证，不太适合高保真房间建模。

### 支撑与再现层：`plotting`、`regression`、`experimental`、`repro`

`plotting` 做得很克制，功能只有 MATLAB 风格样式和几个基础图，见 `spherical_array_processing/plotting/style.py:10` 和 `spherical_array_processing/plotting/politis_helpers.py:11`。我给 **只有基本功能**。它足够用于研究展示，但完全不是一个完整的可视化子系统。

`regression` 明显是开发工具而不是终端用户工具。它负责 MATLAB/Octave 探测、CLI 批处理和状态枚举，见 `spherical_array_processing/regression/matlab.py:41`、`spherical_array_processing/regression/matlab.py:61` 和 `spherical_array_processing/regression/status.py:6`。我给 **只有基本功能**，但会补一句：这块的存在本身是加分项，因为它说明作者在认真维护跨实现一致性。

`experimental` 顾名思义就是实验区。`foa_from_stereo` 明写自己是“Experimental stereo -> incomplete FOA estimator”，见 `spherical_array_processing/experimental/foa_from_stereo.py:45` 和 `spherical_array_processing/experimental/foa_from_stereo.py:50`；`foa_from_stereo_dl` 也不是成熟深度模型，而是“有 checkpoint 就做线性映射，没有就退回直接估计”，见 `spherical_array_processing/experimental/foa_from_stereo_dl.py:80` 到 `spherical_array_processing/experimental/foa_from_stereo_dl.py:93`。这一层我会明确标成 **还欠缺关键特性**。它现在更像研究接口占位，而不是可对外承诺的能力。

`repro` 很特别。它不是 runtime 核心，却是一个**非常大的再现层**，内部把 Politis、Rafaely 和 SHT 相关函数整批 Python 化了，见 `spherical_array_processing/repro/__init__.py:1`、`spherical_array_processing/repro/politis/functions.py:1`、`spherical_array_processing/repro/rafaely/math.py:1` 和 `spherical_array_processing/repro/sht/functions.py:1`。而且这层不是摆设，`tests/test_repro_layers.py:7` 到 `tests/test_repro_layers.py:239` 对它做了大量接口层烟雾测试。我对它的评价是：**对开发者非常有价值，对普通用户不是核心卖点**。如果从产品视角看，它属于“范围很广，但不该让终端用户先接触到”的那一层。

## 数学 / 物理正确性抽查

### `Wigner-D` 旋转

这部分实现是正确而且相当漂亮的。低阶直接求和来自 Sakurai 公式，见 `spherical_array_processing/sh/rotation.py:31`；高阶稳定后端通过对 `J_y` 做 Hermitian 对角化来构造 `exp(-i β J_y)`，见 `spherical_array_processing/sh/rotation.py:77`；完整的 `D^n_{m m'}` 组装写成 `e^{-imα} d e^{-im'γ}`，见 `spherical_array_processing/sh/rotation.py:159` 到 `spherical_array_processing/sh/rotation.py:187`；实基旋转则通过复实变换共轭得到，见 `spherical_array_processing/sh/rotation.py:261`。这套公式链条是对的，测试也验证了低阶与 Sakurai 一致、高阶保持酉性、旋转等价于空间旋转，见 `tests/test_cross_module_consistency.py:217`、`tests/test_cross_module_consistency.py:239` 和 `tests/test_cross_module_consistency.py:253`。我唯一保留的提醒是，低层 `wigner_small_d` 的默认 backend 仍然是 `sakurai`，见 `spherical_array_processing/sh/rotation.py:115`，而文档自己又承认它在 `n≈30` 以后会明显掉精度，见 `spherical_array_processing/sh/rotation.py:66`。这不是公式错误，而是**低层默认值略保守**。

### `MagLS` / `BiMagLS`

`MagLS` 的核心逻辑是标准的 alternating projection：先用 `pinv(Y)` 得到复杂最小二乘解，然后在截止频率之上只保留目标 HRTF 的幅值，反复投影到 SH 可实现子空间，见 `spherical_array_processing/binaural/magls.py:100` 到 `spherical_array_processing/binaural/magls.py:153`。这个实现的关键细节是 `phase_continuation` 被真正做成了跨频率传播前一频点相位，而不是伪开关，见 `spherical_array_processing/binaural/magls.py:116` 和 `tests/test_binaural.py:206`。从数学上说，这里没有明显的符号、归一化或耳侧语义问题。更进一步，`BiMagLS` 还单独拟合每个方向到每只耳的时延 SH 场，见 `spherical_array_processing/binaural/bimagls.py:59`。我对这块的判断是 **算法正确，工程上也比较成熟**。它仍然缺少的是更“产品化”的东西，例如感知加权、测量 HRTF 条件数控制和大规模数据库回归，而不是公式本身。

### shoebox 镜像声源

镜像源位置公式在 `spherical_array_processing/room/shoebox.py:81` 到 `spherical_array_processing/room/shoebox.py:88`，它使用按轴奇偶翻转源位置再加上房间长度整数倍的经典构造；每个像源的振幅按各壁面反射系数幂次乘积再除以传播距离，见 `spherical_array_processing/room/shoebox.py:159` 到 `spherical_array_processing/room/shoebox.py:166`。这在**简化的几何声学模型**下是对的，`tests/test_room.py:28` 也验证了零反射时只剩一个直接声路径，方向和时延都正确。真正的问题不是公式错，而是**物理细节被故意简化**：到达时间被 `round` 到整数采样点，见 `spherical_array_processing/room/shoebox.py:166`；这会引入可闻的时延量化误差，也让高频 comb 结构不够真实。再加上没有空气吸收、没有频率依赖反射相位、没有源/接收指向性，它更像一个**稳健的基线仿真器**，不是高保真声学引擎。

### UHJ 编解码

UHJ 这块做得非常规整。编码常数在模块头就明确写出，见 `spherical_array_processing/ambi/uhj.py:13` 到 `spherical_array_processing/ambi/uhj.py:16`；输入 ACN FOA 先转换成 FuMa 的 `W, X, Y, Z`，其中 `W` 做 `1/√2` 缩放，见 `spherical_array_processing/ambi/uhj.py:132` 到 `spherical_array_processing/ambi/uhj.py:149`；随后 `uhj_encode` 按 Gerzon 公式拼 `S` 与 `D`，见 `spherical_array_processing/ambi/uhj.py:206` 到 `spherical_array_processing/ambi/uhj.py:216`；`uhj_decode` 用经典近似逆式恢复 `W, X, Y`，并把 `Z` 置零，见 `spherical_array_processing/ambi/uhj.py:272` 到 `spherical_array_processing/ambi/uhj.py:283`。测试里既有单音解析公式校验，也有归一化不变性检查，见 `tests/test_ambi_uhj.py:61` 和 `tests/test_ambi_uhj.py:100`。所以这块我认为**常数、符号和通道语义都是对的**。它的局限也很清楚：这里只有 **UHJ-2**，没有 3 声道 / 4 声道变体，也没有真正的流式状态 API，虽然 FIR Hilbert 的文档已经提示了这一点，见 `spherical_array_processing/ambi/uhj.py:186` 到 `spherical_array_processing/ambi/uhj.py:193`。

### NFC-HOA 距离补偿

`NFC-HOA` 的实现是我比较放心的一块。它直接按 `h_n^{(2)}(kd) / h_n^{(2)}(kR)` 做比值，见 `spherical_array_processing/ambi/nfc.py:8` 到 `spherical_array_processing/ambi/nfc.py:16` 和 `spherical_array_processing/ambi/nfc.py:112` 到 `spherical_array_processing/ambi/nfc.py:131`；直流极限也按小参数 Hankel 渐近式替换成 `(R/d)^{n+1}`，见 `spherical_array_processing/ambi/nfc.py:117` 到 `spherical_array_processing/ambi/nfc.py:129`。`tests/test_nfc_hoa.py:20`、`tests/test_nfc_hoa.py:48` 和 `tests/test_nfc_hoa.py:111` 把极限、直接 Hankel 比值和 `n=0` 的闭式幅值都测了一遍。这说明它不仅概念正确，而且边界也处理得对。这里唯一的小提醒是直流检测写成了 `f == 0.0`，见 `spherical_array_processing/ambi/nfc.py:121`，所以如果用户传非常小但不等于零的频点，它走的是原始 Hankel 比值而不是闭式极限。这个行为是可以接受的，但最好在文档里再强调一下“DC special case 只发生在精确零频点”。

### Ledoit-Wolf 收缩

`Ledoit-Wolf` 的实现是标准而且干净的。样本协方差先 Hermitian 对称化，见 `spherical_array_processing/covariance/shrinkage.py:64` 到 `spherical_array_processing/covariance/shrinkage.py:65`；目标矩阵是等迹的标量单位阵，见 `spherical_array_processing/covariance/shrinkage.py:67` 到 `spherical_array_processing/covariance/shrinkage.py:68`；`γ̂` 和 `π̂` 都是按 Frobenius 范数展开来算，见 `spherical_array_processing/covariance/shrinkage.py:70` 到 `spherical_array_processing/covariance/shrinkage.py:90`；最后返回的矩阵再次被对称化，见 `spherical_array_processing/covariance/shrinkage.py:92` 到 `spherical_array_processing/covariance/shrinkage.py:95`。更重要的是，`tests/test_covariance.py:18` 直接拿 `sklearn.covariance.ledoit_wolf(..., assume_centered=True)` 做数值对照，这几乎把“实现是否正确”这个问题关死了。我的保留意见不是公式，而是**接口命名没把“已中心化假设”说得很突出**。在信号处理场景里这通常不是问题，但对一般统计用户来说，若快照含非零均值，这个函数会把均值能量也吞进协方差里。

### DirAC

`DirAC` 的直接声方向估计基于 `W*·[X,Y,Z]`，能量用 `0.5(|W|^2 + |X|^2 + |Y|^2 + |Z|^2)`，再加单极点时间平滑，见 `spherical_array_processing/dirac/analysis.py:131` 到 `spherical_array_processing/dirac/analysis.py:162`。对纯平面波来说，这套写法在当前包的默认平面波编码下能给出正确方向和较低扩散度，`tests/test_dirac.py:29` 和 `tests/test_dirac.py:44` 已经证明了这一点。从算法目的上说，它是对的。问题出在**归一化语义没有完全收束**。文档写“扩散度标定假设 ACN/SN3D FOA”，见 `spherical_array_processing/dirac/analysis.py:85`，但实现里既没有像 `ambi.intensity_vector` 那样先转换归一化，后者见 `spherical_array_processing/ambi/intensity.py:102`，也没有在接口层强制说明“只接受某一种归一化”。这就让 `DirAC` 成为一个**数学上可用、API 语义略飘**的模块。

### FDN late reverb

`FDN` 的核心递推和衰减公式是正确的。文档里写明 `α_i = 10^{-3 L_i /(RT60·f_s)}`，见 `spherical_array_processing/room/fdn.py:11` 到 `spherical_array_processing/room/fdn.py:17`，实现也按这个式子直接计算，见 `spherical_array_processing/room/fdn.py:109` 到 `spherical_array_processing/room/fdn.py:114`。测试里把输出 RT60 和目标 RT60 做了直接拟合比较，见 `tests/test_room_fdn.py:22` 到 `tests/test_room_fdn.py:29`。因此它的**衰减标定是对的**。真正让我担心的是它把“mixing matrix 应该正交/酉”写进了文档，见 `spherical_array_processing/room/fdn.py:3` 到 `spherical_array_processing/room/fdn.py:7` 和 `spherical_array_processing/room/fdn.py:143` 到 `spherical_array_processing/room/fdn.py:146`，但代码只检查形状，见 `spherical_array_processing/room/fdn.py:171` 到 `spherical_array_processing/room/fdn.py:176`。这意味着坏矩阵不会被拦下，用户完全可能自己把能量守恒条件破坏掉。

### rigid-sphere HRTF

刚性球 HRTF 的实现是**物理上自洽**的。它用刚性球模态系数 `b_n(kR)` 和复球谐叠加 HRTF，见 `spherical_array_processing/hrtf/rigid_sphere.py:121` 到 `spherical_array_processing/hrtf/rigid_sphere.py:143`；直流项专门按 `b_0(0)=4π`、高阶为零的极限修复，见 `spherical_array_processing/hrtf/rigid_sphere.py:132` 到 `spherical_array_processing/hrtf/rigid_sphere.py:139`；最后用 `ifftshift` 语义对应的 `fftshift` 把中心参考相位重新放回时域缓冲中心，见 `spherical_array_processing/hrtf/rigid_sphere.py:148` 到 `spherical_array_processing/hrtf/rigid_sphere.py:157`。这不是随便“做个 analytic toy model”，而是和同包的 modal simulator 对上了，见 `tests/test_rigid_sphere_hrtf.py:87` 到 `tests/test_rigid_sphere_hrtf.py:127`；它的 ITD 还和 Woodworth 近似做过比较，见 `tests/test_rigid_sphere_hrtf.py:50`。所以这块的**公式、常数和轴语义都没有明显疑点**。它的局限是模型本身的局限：没有 pinna、没有 torso、没有个体化结构，这一点属于模型边界，不属于实现错误。

## API 质量

如果只看“命名像不像科研代码”，这套 API 已经明显比一般研究仓库强。大部分函数的命名都相当直给，`srp_map`、`esprit_doa`、`radial_equalizer_tikhonov`、`magls_binaural_filters`、`nfc_hoa_distance_filter` 这类名字在科学计算语境下非常清楚，见 `spherical_array_processing/doa/srp.py:46`、`spherical_array_processing/doa/esprit.py:138`、`spherical_array_processing/encoding/radial_filters.py:16`、`spherical_array_processing/binaural/magls.py:26` 和 `spherical_array_processing/ambi/nfc.py:56`。类型注解也比较全面，`py.typed` 也有，见 `pyproject.toml:34`。异常信息大体上是统一的，很多函数都会把收到的 shape 直接带进 `ValueError` 里，像 `spherical_array_processing/ambi/io.py:56`、`spherical_array_processing/hrtf/sofa.py:72`、`spherical_array_processing/room/shoebox.py:139` 这种细节，对用户非常友好。

但 API 质量并不等于“没有坑”。最大的坑我前面已经提过一次，现在必须再强调一次：**全包的通道顺序和轴约定没有做到 100% 一致**。包的主线一直以 ACN 为核心，顶层文档也这么写，见 `README.md:136`；`ambi.intensity` 和 `dirac` 都按 ACN 的 `W, Y, Z, X` 索引读一阶分量，见 `spherical_array_processing/ambi/intensity.py:41` 和 `spherical_array_processing/dirac/analysis.py:11`。但 `diffuseness.intensity_vectors_from_foa` 却要求 `[W, X, Y, Z]`，见 `spherical_array_processing/diffuseness/estimators.py:8`。这不是术语差异，而是真会把横向和纵向分量对调的那种 API 风险。我做了一个独立数值检查：把 `encode_plane_wave` 生成的 ACN FOA 直接送进 `diffuseness.intensity_vectors_from_foa`，方向向量会发生 90 度的轴置换；而当前测试只测了 shape 和范围，没有测方向语义，见 `tests/test_independent_audit.py:671`。

第二个易错点是**轴默认值不统一**。包级 `stft` 明确输出 `F, M, T`，见 `spherical_array_processing/stft.py:56`；`ambi.intensity_vector` 的默认 `coeff_axis` 也适配 `F, 4, T`，见 `spherical_array_processing/ambi/intensity.py:60`；但 `dirac_analysis` 的默认 `coeff_axis` 却是 `-1`，见 `spherical_array_processing/dirac/analysis.py:75`。这意味着用户若把 `sap.stft.stft(...)` 的输出直接喂给 `dirac_analysis`，默认值是错的，必须显式传 `coeff_axis=1`。时域管线 `dirac_render_time_domain` 已经替用户补上了这一点，见 `spherical_array_processing/dirac/pipeline.py:104` 到 `spherical_array_processing/dirac/pipeline.py:110`，但独立使用分析函数时仍然容易踩坑。

第三个易错点是**角度约定默认值分裂**。`SHBasisSpec` 默认 `az_colat`，`SphericalGrid` 默认 `az_el`，这两个类型虽然都有注释说明，见 `spherical_array_processing/types.py:26` 和 `spherical_array_processing/types.py:55`，而且 SH basis 内部也会自动转换，见 `spherical_array_processing/sh/basis.py:109`，但对于用户来说仍然需要额外的心智负担。这个问题没有严重到算 bug，但它会持续制造“为什么我自己的角度数组看起来没问题，结果图像却偏了 90 度”的日常摩擦。

## 测试健康度

测试覆盖面总体上是**广而且层次分明**的。你能看到普通单元测试，比如 `tests/test_covariance.py:18` 对 `Ledoit-Wolf` 做外部对照；也能看到算法合同测试，比如 `tests/test_nfc_hoa.py:48` 直接对 Hankel 比值；还能看到跨模块一致性测试，比如 `tests/test_cross_module_consistency.py:48` 把自由场仿真、SH 仿真、SHT 和 PWD 串成一条链；再往上还有示例级 smoke test，见 `tests/test_example_end_to_end.py:13`，以及大而杂的独立审计测试，见 `tests/test_extended_audit.py:1` 与 `tests/test_independent_audit.py:1`。再加上 CI 还会单独检验 `sdist` 的测试子集，见 `.github/workflows/ci.yml:63`，这说明这套测试不是“堆数量”，而是真的在维护发布质量。

如果只问“533 个测试够不够”，我的回答是：**对一个 1.4 万行左右的科学 Python 包来说，已经相当够用，而且比很多公开研究库健康得多**。本地统计下，测试行数约为源码的一半多一点，这在数值库里是很不错的比例。更重要的是，很多测试不是 happy path。例如 `tests/test_decoding.py:213` 会主动制造共面布局和凸包覆盖失败；`tests/test_room_fdn.py:22` 会从目标 RT60 反推测得 RT60；`tests/test_binaural.py:206` 会盯住一个具体回归场景。这样的测试密度，比单纯堆 shape 检查更有价值。

但我也确实看到了几块**没有被测试照亮的边缘**。最明显的就是前面说的 `diffuseness` 通道顺序问题。`tests/test_independent_audit.py:671` 到 `tests/test_independent_audit.py:728` 只检验了形状和数值范围，没有任何一条测试去验证 `intensity_vectors_from_foa` 在包的默认 ACN 顺序下是否给出正确方向。第二个空白在 `DirAC` 归一化语义上，当前测试只覆盖了默认正交归一化平面波和随机扩散场，见 `tests/test_dirac.py:29` 到 `tests/test_dirac.py:57`，但没有显式对 `sn3d` 或 `n3d` 输入做同一物理场的对照。第三个空白在双耳部分：`MagLS` 和整条 binaural pipeline 的测试使用的都是合成 HRTF 或 analytic rigid-sphere HRTF，见 `tests/test_binaural.py:16` 和 `tests/test_ambi_to_binaural.py:16`，而不是一个真实测量 SOFA 数据集上的数值回归。这样做对单元测试完全合理，但它还不能替代“真实数据库鲁棒性”的验证。

## 发布就绪度

如果问题是“明天能不能发 PyPI 正式 `0.4.0`”，我的答案是：**技术上能发，工程上也基本撑得住，但我不会毫无保留地按正式版心态去发**。正面的部分很明确。`pyproject.toml:1` 到 `pyproject.toml:69` 的打包配置是规范的；可选依赖按 `audio`、`hrtf` 和 `image` 分开，见 `pyproject.toml:47`；`MANIFEST.in:1` 到 `MANIFEST.in:65` 明确控制了 `sdist` 内容；CI 还专门在打包后解开 `sdist` 再跑一遍子集测试，见 `.github/workflows/ci.yml:57` 到 `.github/workflows/ci.yml:69`。我自己在当前工作树上执行 `python3 -m build` 也确实得到了成功的 `wheel` 和 `sdist`。这一套流程已经明显强于“能 import 就上 PyPI”的级别。

真正让我犹豫的是几个发布体验问题。最先想到的是 **文档和示例没有进入发行工件**。README 很坦率地说 `scripts/`、`examples/` 和 `docs/` 是仓库资产，不在 runtime wheel 里，见 `README.md:5` 到 `README.md:10`；`MANIFEST.in:58` 到 `MANIFEST.in:60` 也确实把这些目录裁掉了。这意味着仓库用户和 PyPI 用户得到的是两种体验。仓库用户可以看 `examples/binaural_em32_to_ears.py` 并且它还被测试了，见 `tests/test_example_end_to_end.py:13`；PyPI 用户则拿不到这类脚本。对 beta 版这完全可以接受，但对“正式 0.4.0”来说，我会希望至少保留一个最小示例集或者把文档链接写得更明确。

第二个发布顾虑是依赖面还有一点**不够收敛**。顶层 import 已经尽量避免把 `matplotlib` 提前载入，见 `tests/test_import_contract.py:7`，但 `matplotlib` 仍然是硬依赖，见 `pyproject.toml:36` 到 `pyproject.toml:39`。这会让很多只关心数值处理的服务器端用户为 plotting 付出安装成本。更自然的做法，是把绘图改成一个额外依赖，就像 `soundfile` 和 `h5py` 一样，后两者已经被处理成可选依赖，见 `pyproject.toml:47` 到 `pyproject.toml:60`，而且代码里也做了懒检查，见 `spherical_array_processing/ambi/io.py:40` 和 `spherical_array_processing/hrtf/sofa.py:25`。

第三个发布顾虑是**runtime surface 仍然混入了不少开发者导向模块**。`pyproject.toml:63` 到 `pyproject.toml:65` 会把整个 `spherical_array_processing.*` 都打进 wheel，而 `regression` 与 `repro` 明显属于开发与再现层，见 `spherical_array_processing/regression/__init__.py:1` 和 `spherical_array_processing/repro/__init__.py:1`。这不是错误，包体也不大，但它会让正式版的公开表面显得略宽、略杂。

## 真正剩下的短板

抛开那些显眼的“文档还可以再补、示例还可以再多”这种常规建议，我觉得真正会在后续变成用户痛点的，是**语义一致性问题**。`diffuseness` 现在这个 `[W, X, Y, Z]` 顺序如果不尽快和 ACN 主线对齐，未来每扩展一个 API，都要额外解释一次通道顺序，见 `spherical_array_processing/diffuseness/estimators.py:8`。这种问题的麻烦不在当下，而在它会持续污染后续所有使用者的心智模型。

紧接着会变成痛点的，是**“离线科学工具”和“可交付产品模块”之间的边界还没有完全划清**。`shoebox_rir` 的整数采样延迟，见 `spherical_array_processing/room/shoebox.py:166`，和 `fdn_reverb` 对 mixing matrix 只验 shape 不验正交性，见 `spherical_array_processing/room/fdn.py:171`，在研究代码里都说得过去；但一旦用户把它们当成“可直接用于内容制作或高保真仿真”的生产模块，就会撞上真实限制。更有趣的是，这个包别的地方工程味已经很重了，所以用户更容易对这些模块产生“它们也该是产品级”的预期落差。

还有一个不那么显眼但会逐渐放大的问题，是**同一物理量在不同模块里被重复实现**。最典型的是强度和扩散度语义：`ambi.intensity_vector` 有一套显式的归一化统一逻辑，见 `spherical_array_processing/ambi/intensity.py:57` 到 `spherical_array_processing/ambi/intensity.py:117`；`dirac_analysis` 又自己写了一套简化版，见 `spherical_array_processing/dirac/analysis.py:131` 到 `spherical_array_processing/dirac/analysis.py:162`。短期内这会让代码更直接，长期看则会制造微妙分歧。科学库最怕的不是大 bug，而是两套都“差不多对”的语义慢慢漂开。

最后一个会在包继续长大后显现的问题，是**产品边界与目标用户画像仍然偏混合**。`repro` 层非常有价值，`experimental` 也合理，但它们和 `binaural`、`decoding`、`room` 这些面向终端用户的模块同住一个 wheel，会让项目越来越像“研究工作台 + 发布包”的叠加体。现在体量还不大，这种混合感是优点；再往后长，它会开始影响用户对稳定性等级的预期管理。

## 总评价

如果满分是 10 分，我会给这个版本 **8.4 / 10**。扣分主要落在三件事上。第一，**API 语义还没有完全收敛**，尤其是 `diffuseness` 的通道顺序和 `DirAC` 的归一化处理。第二，**若按正式版标准看，发行工件对文档和示例的交付还不够完整**。第三，**部分房间和实验模块仍然明显是研究基线而不是产品级实现**。

我的一句话结论是：**这是一个非常强的研究级 Python 球阵列处理库，已经明显越过“学生项目”，也比很多公开研究库更工程化；但要把 `0.4.0b11` 直接抬成“工业感十足的正式 0.4.0”，还需要先把接口一致性和发布表面再收一遍。**
