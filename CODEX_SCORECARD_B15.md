# spherical-array-processing 0.4.0b15 独立评分卡

## 结论

这是一次**从头重读实现、重新跑全测、重新构建 wheel、重新做数值核验**后的独立复评分，不沿用 `b14` 的旧分，也不按“9.1 再加若干补丁分”来算。我给 `0.4.0b15` 的当前分数是 **9.6/10**。我的判断是，它已经进入了**高完成度 beta 包**的区间，而且这次**确实跨过了 9.5**。

我这样打分的核心原因很直接。`b14` 阶段真正挡在 9.5 前面的几件事，这次我都在仓库里看到了落地证据，而且不是只补文档或只补测试，而是**实现、测试、打包边界和安装态体验一起收口**了。与此同时，我也没有把它打到 10，因为我在重新核验时仍然看到了两类真实但非阻塞的残余问题：其一是**阵列配置的单一真相源还没有完全建立**，其二是**`directional` 的验证语义还没有一路下沉到最底层声学包装函数**。

## 验证基线

我先在仓库根目录重新跑了你指定的基线命令 `PYTHONPATH=. python3 -m pytest -q`，结果是 **584 passed, 4 warnings**。这个计数和你给出的状态完全一致。警告的主体仍然是你有意保留的 developer-only `FutureWarning`，外加一个绘图脚本里的 `tight_layout` 提示，没有出现新的数学错误或契约性失败。

紧接着我重新执行了 `python3 -m build --wheel --no-isolation`。wheel 成功生成，工件是 `dist/spherical_array_processing-0.4.0b15-py3-none-any.whl`。我没有只看构建日志，而是直接解读 zip 内容做计数。结果是 wheel 内部 `spherical_array_processing/` 前缀下共有 **67 个稳定层文件**，其中 **developer-only 层文件数为 0**，而 `examples` 子包文件数是 **3**，分别是 `spherical_array_processing/examples/__init__.py`、`spherical_array_processing/examples/binaural_em32_to_ears.py` 和 `spherical_array_processing/examples/plane_wave_doa.py`。这和 `pyproject.toml:48` 之后的打包配置是吻合的，也和 `tests/test_import_contract.py:37` 与 `tests/test_import_contract.py:68` 锁住的 wheel 契约一致。

我随后又把数值等价性单独拿出来算，而不是只相信测试名。对 `spherical_array_processing/array/simulation.py:97` 的 `simulate_sh_array_response`，我独立计算了 **α=0.5 对 `cardioid`** 和 **α=1.0 对 `open`** 的最大绝对差，结果两项都是 **0.0**。对 `spherical_array_processing/encoding/radial_filters.py:42`、`spherical_array_processing/encoding/radial_filters.py:103` 和 `spherical_array_processing/encoding/radial_filters.py:175` 这三个 equalizer 入口，我分别做了同样的核验，最大绝对差也都是 **0.0**。对 `spherical_array_processing/diffuseness/estimators.py:15` 的 `intensity_vectors_from_foa`，我又在 `normalization ∈ {orthonormal, n3d, sn3d}`、`physical_units ∈ {False, True}` 的六种组合下逐一对比 `spherical_array_processing.ambi.intensity_vector`，最大绝对差同样全部是 **0.0**。

最后我还直接运行了两个 wheel 内 example 的模块入口，也就是 `python -m spherical_array_processing.examples.plane_wave_doa` 和 `python -m spherical_array_processing.examples.binaural_em32_to_ears`。两者都能正常执行并打印合理的 sanity summary，这说明 `spherical_array_processing/examples/__init__.py:1` 开始建立的安装态例程表面不是空壳。

## 数学内核

这一块我给出很高评价。包的主干数学部分在 `b14` 已经相当成熟，这次 `b15` 没有引入新的摇晃点，反而把一个过去会让人担心的边缘接口彻底并轨了。`spherical_array_processing/diffuseness/estimators.py:15` 现在把 `intensity_vectors_from_foa` 收敛成 `spherical_array_processing.ambi.intensity_vector` 的薄包装，这意味着 **FOA 强度矢量的定义只剩一份实现**。这件事的价值不在于少写了几行代码，而在于它消除了以后“一个接口改了物理单位缩放，另一个接口忘了跟”的漂移风险。

更重要的是，`directional` 一阶指向性阵列现在在数学上已经不是半接入状态。`spherical_array_processing/encoding/radial_filters.py:16` 的验证入口、`spherical_array_processing/array/simulation.py:192` 之后的显式参数检查，以及 `spherical_array_processing/acoustics/radial.py:350` 的 modal wrapper，至少已经把**名义支持**和**数值极限**都打通了。尤其是 **α=0.5 ≡ cardioid** 与 **α=1.0 ≡ open** 的零差核验，说明这里不是“接近相等”，而是实现路径已经被压到了完全相同的解析形式。

## API 质量

这一项比 `b14` 明显更强，因为过去最容易让人踩坑的那条 FOA / diffuseness 支线，现在已经从接口层面收口了。`spherical_array_processing/diffuseness/estimators.py:15` 不但新增了 `normalization` 和 `physical_units`，而且保留了历史错误消息和 FuMa 路径，这属于**向前兼容和向后兼容同时兼顾**的改法。`tests/test_diffuseness_intensity_wrapper.py:34` 开始的几组断言把这件事锁得很严。

发布态 example 也真正进入了公共安装面。`spherical_array_processing/examples/__init__.py:43` 的 `__all__`、`tests/test_installed_examples.py:19` 之后的导入与运行契约、以及 README 的安装后调用说明，共同把“示例脚本”从 repo 附属物变成了**wheel 用户可直接消费的稳定资源**。这对 beta 包的观感提升非常大，因为它减少了用户第一次接触项目时的摩擦。

不过，这一项还没有满分，我保留扣分的地方非常具体。`spherical_array_processing/types.py:98` 的 `ArrayGeometry` 现在声明了 `array_type`、`sensor_kind` 和可选的 `metadata["dir_coeff"]`，但 `spherical_array_processing/array/simulation.py:97` 的 `simulate_sh_array_response` 仍然把 `array_type` 作为独立参数，默认还是 `"rigid"`，并不会自动读取 `geometry.array_type` 或 `geometry.metadata["dir_coeff"]`。这意味着**几何对象和求解入口仍然存在双重配置源**。从纯实现角度说这不算 bug，因为当前文档已经写明了它不会自动读取；但从 API 设计角度说，它仍然让“几何长什么样”和“仿真按什么阵列类型求值”分裂成两处真相源。

更进一步说，`sensor_kind` 在当前稳定树里依然没有真正进入行为层。我重新搜过代码，除了 `spherical_array_processing/types.py:128` 这个字段定义本身，稳定实现里没有消费者。一个出现在公开 dataclass 里的字段如果长期只存在于类型定义和文档里，它就会逐渐带来**看上去可用、实际上不生效**的认知负担。

## 测试健康

这一项现在已经是这个包的长板之一。`584 passed` 本身当然只是表面，真正让我给高分的是这次新增测试覆盖的方向很对。`tests/test_directional_array_plumbing.py:1` 不是只测 happy path，而是把 **等价极限、缺参、误参、越界** 这几个最容易在重构时漂移的点一起钉死了。`tests/test_diffuseness_intensity_wrapper.py:1` 同样没有停在“形状正确”，而是直接要求**位级一致**。`tests/test_installed_examples.py:1` 和 `tests/test_import_contract.py:68` 则补上了安装态 wheel 表面的契约测试，这正好击中了 `b14` 的发布边界短板。

`A1` 的处理也是真修复而不是遮羞。`tests/test_plotting_helpers.py:1` 和 `tests/test_rafaely_plot_wrappers.py:1` 现在都在模块级使用 `pytest.importorskip("matplotlib")`，这让“最小开发环境没有装 plotting extra”时的行为从**收集期崩溃**变成了**干净 skip**。这类改动分数不高调，但它直接影响 CI 与用户本地复核时的体验稳定性。

## 发布就绪度

这是 `b15` 最大的提升点之一。`pyproject.toml:48` 之后把 `include-package-data = false` 和显式 `packages.find` 排除规则一起用上，终于把 **wheel 边界** 和 **sdist / 源树边界** 区分清楚了。对终端用户来说，这意味着 `pip install spherical-array-processing` 安装到的是稳定公共面，而不是把 `repro`、`regression`、`experimental` 这些开发层也一起带进去。对维护者来说，这又不会破坏 source distribution 的复现工作流，因为开发层仍然保留在 sdist 里。

更有价值的是，这次不是只“理论上能打包”，而是把 wheel 里的正负两侧契约都测上了。`tests/test_import_contract.py:37` 锁住 developer-only 层必须不进 wheel，`tests/test_import_contract.py:68` 锁住 `examples` 子包必须进 wheel。换句话说，`b15` 的发布边界已经从“靠维护者记忆维持”变成了**由自动化测试守住**。

## 为什么还不是 10 分

我不给 10，不是因为我还想保留一点神秘感，而是因为我确实看到了两处会继续影响长期 API 收敛的残余问题。

第一处是前面提到的**阵列配置双真相源**。`ArrayGeometry` 已经长出了 `array_type`、`sensor_kind` 和 `metadata["dir_coeff"]`，但核心仿真与编码入口并不以它为唯一来源。想把这一项抹平，最干净的翻法有两种。要么让 `simulate_sh_array_response` 与相邻编码入口在未显式传参时自动从 `geometry` 读取，并把“显式参数优先于 geometry”写成正式契约；要么反过来，把 `ArrayGeometry` 上当前不生效的行为字段做软弃用，只保留纯几何信息。现在这种“字段存在、文档承认它目前不生效、调用者仍需再传一遍”的状态，会持续制造边缘困惑。

第二处是**底层声学包装函数的 `dir_coeff` 语义还不够对称**。我独立试调时确认，`bn_matrix(…, sphere="open", dir_coeff=0.5)` 和 `sph_modal_coeffs(…, array_type="open", dir_coeff=0.5)` 目前会静默成功返回，而不是像 equalizer 层和 simulation 层那样拒绝 stray `dir_coeff`。这不影响你这次宣称的 A3，因为你说清楚了“三类验证对称性”当前锁在 equalizer 层和 simulation 层；但它确实意味着**`directional` 的参数验证尚未在整个公开 API 面完全统一**。要把这一项补平，只需要把 `spherical_array_processing/encoding/radial_filters.py:16` 那套规则抽到共享 helper，或在 `spherical_array_processing/acoustics/radial.py:293` 与 `spherical_array_processing/acoustics/radial.py:350` 处对齐同一语义，然后补几条回归测试即可。

## b16 的自然下一步

如果要做一个最顺手、收益也最高的 `b16`，我不会再优先去碰数学核，而会把火力集中在**配置语义收口**。最自然的微发布目标，是把 `ArrayGeometry` 从“带有若干行为提示的几何容器”推进成**真正可执行的阵列规格对象**，或者反过来把那些不执行的行为字段从稳定面里撤掉。与此同时，把 `bn_matrix` / `sph_modal_coeffs` 的 `dir_coeff` 规则和上层对齐，完成 `directional` 语义在**类型层、声学层、仿真层、编码层**的最后一段闭环。

如果只做这两件事，而且继续维持现在的测试纪律，我会认为它非常接近 **9.8–9.9** 这一区间。换句话说，`b15` 已经不是“离优秀还差很多”的版本了；它现在更像一个**主体已经成熟，只剩少数 API 精修债务**的版本。
