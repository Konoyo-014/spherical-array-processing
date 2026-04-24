# spherical-array-processing 0.4.0b14 独立评分卡

## 结论

这是一次**从头重读代码、重新跑测试、重新看发行工件**后的独立审阅，不沿用 b11 的旧分。我的判断是，`0.4.0b14` 已经从“很强的研究型 Python 移植包”明显推进到了“**数学内核成熟、API 主干基本收敛、测试体系非常强的 beta 包**”。我给它的当前分数是 **9.1/10**。

这个分数比我在 b11 阶段给出的 **8.4/10** 高，原因不是“后续补丁很多所以顺手加分”，而是我这次重新检查后确认，之前最伤分的那条 **API 语义不收敛** 已经基本收口了，尤其是在 FOA/DirAC 这条线上。与此同时，我也确认了两个仍然会阻止它进入 **9.5+** 区间的现实问题：其一是**发布工件和打包测试子集还没有完全自洽**，其二是**公开 API 仍然存在少量边缘不一致**，其中最典型的是 `directional` 阵列语义在不同层之间没有完全打通。

## 验证基线

我先跑了你指定的基线命令：`PYTHONPATH=. python3 -m pytest -q`。结果是 **557 passed**，总耗时约 **21.36s**，同时出现 **384 条 warning**。这些 warning 里大头是 `matplotlib` / `pyparsing` 的第三方弃用提示，以及你有意保留的 developer-only `FutureWarning`，不是核心数学错误。

紧接着我又跑了 `python3 -m build`。结果是 **wheel 和 sdist 都能成功构建**，构建出的工件是 `dist/spherical_array_processing-0.4.0b14-py3-none-any.whl` 和 `dist/spherical_array_processing-0.4.0b14.tar.gz`。

为了专门检查“发布就绪度”而不是只看仓库态，我还做了一次**干净虚拟环境的 sdist 复核**。具体做法是在临时目录解开 `dist/spherical_array_processing-0.4.0b14.tar.gz`，只按 `README.md:164` 的写法安装 `.[dev]`，然后运行被打包进 sdist 的 `tests/test_plotting_helpers.py`。结果在没有 `matplotlib` 的环境里**直接 collection 失败**，报错是 `ModuleNotFoundError: No module named 'matplotlib'`。这说明当前 sdist 的“打包测试子集”并不完全满足 README 所暗示的最小安装路径，这个点是我这次保留扣分的关键证据。

## 整体定位

从版图上看，这个包现在已经不是一个只做 SH basis 和几个波束形成器的小工具了。`spherical_array_processing/__init__.py:104` 到 `spherical_array_processing/__init__.py:146` 暴露出的稳定层已经覆盖了 **SH 基函数与变换、旋转、阵列采样与仿真、固定与自适应波束形成、DOA、diffuseness、coherence、编码、解码、binaural、ambi 格式、room、covariance、DirAC、HRTF、plotting**。这个覆盖面在纯 Python 的球形阵列处理包里已经相当完整。

更重要的是，它不是“功能很多但彼此松散”。我这次重读时的直观感受是，**内部约定开始真正统一**了。ACN 顺序、orthonormal 归一化、`(F, Q, T)` 的 STFT 张量布局、developer-only 层和稳定层的边界、plotting 作为 optional extra 的懒导入行为，这些以前容易互相打架的约定，现在大多数已经收口到同一个中心了。

与此同时，我也不想把它说得过满。这个包目前最成熟、最可靠的部分，仍然是 **SH / acoustics / array / DOA / decoding / ambi / covariance / DirAC / HRTF** 这一圈。`room` 模块已经摆脱了“明显研究基线”的状态，但它依然是一个**干净、可验证、适合实验的基线房间声学层**，还不是高保真房声引擎。`experimental`、`repro`、`regression` 现在被边界管理得更好了，但它们本身仍然是开发层，不应该算进稳定 API 的成熟度红利里。

## 数学正确性抽查

这次我没有只看测试名，而是额外做了独立数值抽查。结论是，**核心数学链条是可信的**。

- **SH 基函数正交性与 SHT 往返**。在 `gauss_legendre_sampling(12)` 上测试 `max_order=5` 的实球谐基，`Y^T W Y` 对单位阵的最大偏差约为 **2.0e-15**；随后做 `inverse_sht -> direct_sht` 往返，最大系数误差约为 **3.1e-15**。这和 `spherical_array_processing/sh/basis.py:110`、`spherical_array_processing/sh/transforms.py:8` 到 `spherical_array_processing/sh/transforms.py:77` 的实现是一致的，说明**基函数规范、权重使用和 WLS 直接变换**是自洽的。

- **Wigner 小 d 与 SH 旋转矩阵正交性**。对 `wigner_small_d(12, 0.73)` 检查 `d d^T`，最大残差约 **1.8e-15**；对 `sh_rotation_matrix_real(5, 0.3, 0.7, -0.2)` 检查 `R R^T`，最大残差约 **1.4e-15**。这与 `spherical_array_processing/sh/rotation.py:76` 到 `spherical_array_processing/sh/rotation.py:115` 的 `J_y` 本征分解路径吻合，说明 **b12 把 `jy` 作为默认后端** 不只是“更稳”，而是真正在数值上把高阶旋转的默认行为拉到了工程安全区。

- **开放球阵 SH 响应与直接平面波仿真一致性**。我把 `simulate_plane_wave_array_response` 和 `simulate_sh_array_response(..., array_type="open")` 做了直接比对，在 32 麦克风、`max_order=18`、`fft_len=512` 的配置下，最大响应差约 **2.7e-08**。这说明 `spherical_array_processing/array/simulation.py:97` 到 `spherical_array_processing/array/simulation.py:229` 的 Legendre 展开和 `spherical_array_processing/acoustics/radial.py` 里的 `B_n` 定义是对得上的。

- **PWD / MUSIC 对合成单平面波协方差的恢复**。我用一个复球谐平面波系数构造 rank-1 协方差，然后在 4000 点 Fibonacci 扫描栅格上测试 `pwd_spectrum` 和 `music_spectrum`。两者都在**扫描网格本身的极限误差**内命中了真 DOA，角误差约 **0.87°**。这和 `spherical_array_processing/doa/spectra.py:157` 到 `spherical_array_processing/doa/spectra.py:286` 的 steering convention 是一致的，也说明之前最容易出错的“共轭方向感”现在是对的。

- **EPAD 能量保持性**。对 64 扬声器、三阶解码，我检查了 `D^T D` 相对 `4π/L · I` 的偏差，最大误差约 **5.3e-16**。这直接对应 `spherical_array_processing/decoding/decoders.py:139` 到 `spherical_array_processing/decoding/decoders.py:173` 的 SVD 构造，说明 **EPAD 的能量保持条件在实现上是成立的**。

- **shoebox 直达路径时延**。在无反射房间中，我用 `shoebox_rir(..., interpolation="sinc")` 检查直达声的到达时刻，得到的重心时延相对几何真值的误差约 **3.2 微秒**。这和 `spherical_array_processing/room/shoebox.py:49` 到 `spherical_array_processing/room/shoebox.py:95` 的 Kaiser 窗 sinc 分数延迟插值是一致的，说明这次 sinc 插值不是表面修补，而是真的把**样点量化误差**从主路径上拿掉了。

- **DirAC 对纯平面波的 diffuseness**。对纯 FOA 平面波 STFT，`dirac_analysis` 给出的 diffuseness 最大值是 **0.0**。这和 `spherical_array_processing/dirac/analysis.py:149` 到 `spherical_array_processing/dirac/analysis.py:170` 的 textbook `ψ = 1 - ||I||/E` 实现一致，也能反向证明 `spherical_array_processing/ambi/intensity.py:57` 到 `spherical_array_processing/ambi/intensity.py:95` 的 `_canonical_foa_pv` 共享 canonical helper 的确把物理量语义统一起来了。

- **AIC / MDL 源数估计**。我额外构造了一个有三条主特征值的 16 维协方差，`estimate_n_sources(..., criterion="mdl")` 和 `criterion="aic"` 都恢复了 **3**。这和 `spherical_array_processing/doa/source_count.py:53` 到 `spherical_array_processing/doa/source_count.py:118` 的 Wax–Kailath 实现相符。

如果把这些抽查和现有测试套件叠加来看，我对这个包的数学可信度评价是：**核心算法已经明显高于“实现正确但靠测试兜底”的阶段，进入了“公式、约定、数值表现三者基本一致”的阶段**。

## API 质量

这次最大的正面变化，确实出现在 API 层。

我在 b11 时最担心的是，包里虽然已经形成了很多能力，但**少数关键接口在约定上还没完全收口**。现在回头看，这个问题已经基本修了。`spherical_array_processing/diffuseness/estimators.py:12` 到 `spherical_array_processing/diffuseness/estimators.py:82` 里，`intensity_vectors_from_foa` 的 `channel_order` 默认已经明确改成 `acn`；`spherical_array_processing/dirac/analysis.py:68` 到 `spherical_array_processing/dirac/analysis.py:128` 里，`dirac_analysis` 的默认 `coeff_axis` 也已经改成 `-2`，和包内 `stft` 的 `(F, Q, T)` 约定对齐。更关键的是，`spherical_array_processing/ambi/intensity.py:57` 到 `spherical_array_processing/ambi/intensity.py:95` 新出现的 `_canonical_foa_pv` 把 FOA 的 pressure / velocity 语义做成了共享 canonical helper，`dirac_analysis` 直接复用它，这意味着 **DirAC 和 intensity_vector 现在终于在“W、X/Y/Z 到底代表什么”这个问题上说同一种语言**。

developer-only 层的边界也比 b11 清晰得多。`spherical_array_processing/__init__.py:133` 到 `spherical_array_processing/__init__.py:146` 没有把 `repro`、`regression`、`experimental` 暴露成顶层稳定命名空间，而 `tests/test_import_contract.py:35` 到 `tests/test_import_contract.py:98` 又把这种边界写成了可执行合同。这个处理是成熟的，因为它没有粗暴删掉开发层，而是让**稳定层与开发层之间的心理边界和机器边界同时成立**。

不过 API 还没有完全无毛刺。我这次最明确保留的一个 API 扣分，是 **`directional` 阵列语义在不同模块之间还没有完全贯通**。`spherical_array_processing/array/simulation.py:94` 与 `spherical_array_processing/array/simulation.py:142` 已经公开支持 `"directional"`；底层的 `plane_wave_radial_bn` 也支持 `dir_coeff`。但 `ArrayGeometry` 在 `spherical_array_processing/types.py:101` 仍然把 `array_type` 限死为 `"open" | "rigid" | "cardioid"`，而 `spherical_array_processing/encoding/radial_filters.py:20`、`spherical_array_processing/encoding/radial_filters.py:73`、`spherical_array_processing/encoding/radial_filters.py:139` 也还停留在这三类。这不是大 bug，但它说明**公共类型层、仿真层和编码层对“可支持的阵列族”还没有完全统一**。

另一个小但真实的 API 问题，是 **FOA 强度相关接口仍然有双轨制残留**。`spherical_array_processing/ambi/intensity.py:98` 到 `spherical_array_processing/ambi/intensity.py:150` 这一套已经是更现代、更完整的 API，支持 `normalization`、`coeff_axis` 和 `physical_units`。但 `spherical_array_processing/diffuseness/estimators.py:12` 这条旧接口还保留着 last-axis-only、channel-order-only 的较窄合同。现在它的默认行为已经修对了，所以我不再把它算作严重语义问题；不过如果追求 9.5+，这条接口最好继续收敛，而不是长期双轨共存。

## 测试健康度

测试健康度是这个版本非常强的一面。按 `pytest --collect-only -q` 的结果，当前一共收集到 **557 个测试**。其中 `tests/test_independent_audit.py` 一份就有 **89 个**测试，`tests/test_extended_audit.py` 有 **39 个**，`tests/test_cross_module_consistency.py` 有 **36 个**，`tests/test_decoding.py` 有 **29 个**。这说明测试不是只堆在单模块单函数上，而是把**数学恒等式、交叉模块一致性、端到端流程、打包与导入合同**都覆盖到了。

我尤其认可两类测试。第一类是**独立审计型测试**，它们不是顺着实现写断言，而是从公式和外部恒等式重新推导期望值。第二类是**分层边界测试**，比如 `tests/test_import_contract.py:7` 到 `tests/test_import_contract.py:131` 这种，直接把“stable public API 到底是什么”写成了机器可执行的约束。这是成熟库才会做的事。

但测试体系并不是没有问题。最大的问题不是仓库态，而是**打包态的测试子集策展还不够严谨**。`MANIFEST.in:34` 和 `MANIFEST.in:36` 把 `tests/test_plotting_helpers.py` 与 `tests/test_rafaely_plot_wrappers.py` 收进了 sdist，可这两个文件在 `tests/test_plotting_helpers.py:1` 到 `tests/test_plotting_helpers.py:7` 以及 `tests/test_rafaely_plot_wrappers.py:1` 到 `tests/test_rafaely_plot_wrappers.py:7` 里都无条件导入了 `matplotlib`。与此同时，`pyproject.toml:61` 到 `pyproject.toml:63` 又把 `matplotlib` 放在可选的 `[plotting]` extra 里，而 `README.md:164` 到 `README.md:167` 的 release check 示例只写了 `.[dev]`。这使得“**仓库测试很强**”和“**打包后测试子集自洽**”变成了两件不同的事。前者我给高分，后者我保留扣分。

## 发布就绪度

如果把“发布就绪度”理解为“能否生成可安装工件，并且运行时依赖足够干净”，那这个版本已经相当不错。`pyproject.toml:36` 到 `pyproject.toml:63` 的依赖面很克制，运行时核心依赖只有 `numpy` 和 `scipy`，`matplotlib` 与 `h5py` 都放进了 optional extras，这在 scientific Python 包里是正确方向。`tests/test_import_contract.py:7` 到 `tests/test_import_contract.py:32` 还专门证明了 plotting 的懒导入行为，说明这个 optional 不是文档口号，而是实际合同。

问题在于，“发布就绪”不只是 build 成功，还包括**工件内容是否与文档、自测流程和稳定层叙事一致**。这方面还有两条没有完全做好。

第一条是前面说的 **sdist 测试子集不自洽**。这不是小节上的瑕疵，而是真正会让 clean-env 验证卡住的问题。第二条是 **runtime wheel 仍然携带 developer-only 层**。构建日志和 wheel 内容都能看到 `spherical_array_processing/repro/*`、`spherical_array_processing/regression/*`、`spherical_array_processing/experimental/*` 被打包进 wheel。确实，`tests/test_import_contract.py:35` 到 `tests/test_import_contract.py:72` 保证了它们不从顶层泄露；确实，`spherical_array_processing/regression/__init__.py:1`、`spherical_array_processing/repro/__init__.py:1`、`spherical_array_processing/experimental/__init__.py:1` 也都加了 `FutureWarning`。但这依然意味着 **工件边界上的“稳定层 / 开发层”没有彻底分离**。对 beta 阶段我可以接受，对 9.5+ 我会继续扣分。

另外一个老问题，我这次认为是**部分改善但没有彻底关闭**。那就是**文档与示例在安装工件里的交付**。`MANIFEST.in:58` 到 `MANIFEST.in:60` 明确把 `docs`、`examples`、`scripts` 都从 sdist 里 prune 掉了。这让运行时工件更轻，但也意味着使用者拿到发布包以后，除了 README 之外，很难直接获得一组可运行的最小示例。对很多成熟 scientific 包来说，这不是硬性缺陷；但你这个包的覆盖面已经很大，模块之间的组合也越来越丰富，到了这个阶段，**一个小而精的安装态示例集**会显著提升发布完成度。

## 哪些旧扣分点已经确认修复

下面这几条，是我在 b11 时明确扣过分、而这次确认可以收回的部分。

- **API 语义未收敛：已基本修复**。`diffuseness` 的 FOA 通道顺序问题现在有了显式 `channel_order` 并且默认回到 ACN，见 `spherical_array_processing/diffuseness/estimators.py:12`。`DirAC` 的默认 `coeff_axis` 现在改成 `-2`，见 `spherical_array_processing/dirac/analysis.py:68` 到 `spherical_array_processing/dirac/analysis.py:98`。这两条都是我在 b11 时最担心的“静默错位”型问题，现在已经不再是主要扣分项。

- **DirAC 强度 / 归一化语义分裂：已修复**。`spherical_array_processing/ambi/intensity.py:57` 到 `spherical_array_processing/ambi/intensity.py:95` 的 `_canonical_foa_pv`，配合 `spherical_array_processing/dirac/analysis.py:149` 到 `spherical_array_processing/dirac/analysis.py:160`，把 DirAC 与 intensity 的 canonical 语义统一了。我这次数值抽查也验证了纯平面波下 diffuseness 为零。

- **plotting 依赖过重：已明显改善**。`pyproject.toml:61` 到 `pyproject.toml:63` 把 `matplotlib` 放进了 `[plotting]` extra，而 `tests/test_import_contract.py:7` 到 `tests/test_import_contract.py:32` 又证明了顶层导入不会偷偷拉起 `matplotlib`。这一条我确认修好了。

- **房间模块存在明确正确性短板：已部分修复并降级为能力短板**。`spherical_array_processing/room/shoebox.py:18` 到 `spherical_array_processing/room/shoebox.py:95` 的 sinc 分数延迟插值，和 `spherical_array_processing/room/fdn.py:66` 到 `spherical_array_processing/room/fdn.py:106` 的 mixing matrix 正交性校验，确实修掉了两处容易把 baseline 实验做偏的点。现在的 `room` 我不再把它评成“研究基线里可能有明显数值坑”的状态，而是评成“**正确、干净、但特性仍然偏基线**”的状态。

- **developer-only 子模块边界模糊：大体修复**。现在稳定层和开发层的边界已经在导入合同、命名空间和 warning 语义上清楚了，这一点比 b11 有明显进步。

## 仍然存在的扣分点

我这次保留的扣分，主要集中在下面几项，而且它们都不是“锦上添花项”，而是真会影响 9.5+ 评分的点。

- **sdist 打包测试子集不自洽**。这条是当前版本最大的发布层扣分。`MANIFEST.in:34`、`MANIFEST.in:36` 把 plotting 测试收进 sdist，但 `pyproject.toml:61` 到 `pyproject.toml:63` 又把 plotting 依赖设成可选，而 `README.md:164` 到 `README.md:167` 还给出了只装 `.[dev]` 的 release check 示例。我已经在 clean venv 里复现出 `tests/test_plotting_helpers.py` 因缺少 `matplotlib` 而 collection 失败。这个问题不伤数学正确性，但它伤**发布可信度**。

- **runtime wheel 仍然携带 developer-only 层**。边界管理已经比以前清楚，但只要 `repro`、`regression`、`experimental` 还在 wheel 里，发布工件就仍然带着“稳定层之外的实现债”。这不是灾难，但它让包的工件边界仍然偏研究工作台，而不是完全收束的 runtime artifact。

- **`directional` 支持没有贯穿全部公共 API**。`spherical_array_processing/array/simulation.py:94` 与 `spherical_array_processing/array/simulation.py:142` 说支持 `directional`，但 `spherical_array_processing/types.py:101` 和 `spherical_array_processing/encoding/radial_filters.py:20` 这一侧还没有跟上。这会让用户在类型提示、文档期望和可调用实现之间遇到轻度失配。

- **安装工件里的示例交付仍然偏薄**。`MANIFEST.in:58` 到 `MANIFEST.in:60` 把 `docs`、`examples`、`scripts` 全部排除了。仓库态当然有大量脚本和复现实验，但安装态几乎完全依赖 README。对当前这个包的体量和覆盖面来说，我认为这仍然值得一点扣分。

- **FOA 强度 / diffuseness 的公共接口还有双轨残留**。`spherical_array_processing/ambi/intensity.py:98` 这条新 API 已经明显更好，但 `spherical_array_processing/diffuseness/estimators.py:12` 的老入口还没有彻底退场或完全包装成同一套合同。它现在已经不构成严重 bug 风险，但还构成少量认知负担。

## 评分

如果只看**数学内核**，我会给得更高，接近 9.5。真正把总分拉回 **9.1/10** 的，不是数值稳定性，而是**发布层面和 API 边缘一致性**还差最后一截。换句话说，这个版本离“高可信 beta”已经很近了，但离“你可以非常有底气地把它当成一个打磨完成的 scientific package 去发布”还差几个具体动作。

## 到 9.5/10 需要完成的事项

- **先把 sdist 打包测试子集做成真正自洽的一套。** 当前最直接的修法有两条，要么把 `tests/test_plotting_helpers.py:1` 和 `tests/test_rafaely_plot_wrappers.py:1` 改成 `pytest.importorskip("matplotlib")` 风格的可选依赖测试，要么就把它们从 `MANIFEST.in:34` 与 `MANIFEST.in:36` 的 sdist 测试子集里移出去。如果你坚持把 plotting 测试保留在打包子集里，那就需要同步修改 `README.md:164` 和 `pyproject.toml:50`，把 release check 的最小安装路径写成 `.[dev,plotting]`，而不是现在的 `.[dev]`。

- **把 developer-only 层从 runtime wheel 里真正剥离，或者至少拆成单独分发单元。** 现在 `pyproject.toml:65` 到 `pyproject.toml:67` 会把 `spherical_array_processing.*` 全量打进 wheel。要冲到 9.5+，我建议明确决定：要么把 `spherical_array_processing.repro`、`spherical_array_processing.regression`、`spherical_array_processing.experimental` 从 runtime wheel 里排除，只留在仓库态 / sdist / dev extra；要么把它们拆成单独的 developer package。做完以后，再同步调整 `tests/test_import_contract.py:35` 到 `tests/test_import_contract.py:72` 的合同测试，让“稳定层与开发层边界”不仅在命名空间上成立，也在工件边界上成立。

- **把 `directional` 阵列语义沿公共 API 全链路打通。** 现在 `spherical_array_processing/array/simulation.py:94` 和 `spherical_array_processing/array/simulation.py:142` 已经公开支持 `"directional"`，但 `spherical_array_processing/types.py:101` 与 `spherical_array_processing/encoding/radial_filters.py:20`、`spherical_array_processing/encoding/radial_filters.py:73`、`spherical_array_processing/encoding/radial_filters.py:139` 还没有跟上。要冲 9.5+，这件事不能再停留在“底层可以、外层没声明”的状态。要么把 `directional` 正式升级为一等公民并补全类型、文档、测试；要么反过来缩窄公开合同，只保留三种稳定阵列类型。

- **把 FOA 强度 / diffuseness 的公共入口收成一条主线。** 目前最合理的中心接口已经是 `spherical_array_processing/ambi/intensity.py:98` 这套支持 `normalization`、`coeff_axis`、`physical_units` 的 API，而 `spherical_array_processing/diffuseness/estimators.py:12` 这条旧接口还保留着较窄合同。到 9.5+，我建议二选一：要么在 `spherical_array_processing/diffuseness/estimators.py:12` 把它改成对 `ambi.intensity_vector` 的薄包装，并补上同样的语义参数；要么正式把它标成 legacy / deprecated，让包里只有一个 canonical 的 FOA-intensity 入口。

- **补齐安装态的最小示例交付。** `MANIFEST.in:58` 到 `MANIFEST.in:60` 现在直接 prune 了 `docs`、`examples`、`scripts`。这在 beta 阶段可以理解，但对一个已经覆盖这么多子模块的包来说，9.5+ 版本最好至少提供一组**运行时安全、无仓库私货依赖**的最小示例。最直接的落点，就是修改 `MANIFEST.in:58` 到 `MANIFEST.in:60` 的策略，并在 `README.md:151` 到 `README.md:173` 把“仓库全量测试 / 发布工件最小示例 / 可选 extras”三件事讲得更清楚。

## 当前分数

**9.1/10**

## 到 9.5/10 需要完成的事项

- 让 sdist 打包测试子集在 `.[dev]` 或明确声明的最小 extras 下真正自洽，优先处理 `MANIFEST.in:34`、`MANIFEST.in:36`、`tests/test_plotting_helpers.py:1`、`tests/test_rafaely_plot_wrappers.py:1`、`README.md:164`。
- 把 developer-only 层从 runtime wheel 中拆出去或隔离出去，核心落点在 `pyproject.toml:65` 与 `tests/test_import_contract.py:35`。
- 把 `directional` 阵列支持在 `spherical_array_processing/types.py:101`、`spherical_array_processing/encoding/radial_filters.py:20`、`spherical_array_processing/array/simulation.py:94` 这一整条链上统一起来。
- 收敛 FOA 强度 / diffuseness 公共接口，核心落点在 `spherical_array_processing/diffuseness/estimators.py:12` 与 `spherical_array_processing/ambi/intensity.py:98`。
- 补齐安装态的最小示例交付与文档说明，核心落点在 `MANIFEST.in:58`、`MANIFEST.in:59`、`MANIFEST.in:60` 与 `README.md:151`。
