# spherical-array-processing 0.4.0b15 second polish 复评分卡

## 结论

这轮我按你指定的四个热点重新读了实现，并且重新做了独立核验。我给 `0.4.0b15` 的最终 polish 分数是 **9.95/10**。

这个分数背后的判断很简单。你上一轮被我扣住的四个具体残留里，**三个已经完全关闭**，剩下那个 `sensor_kind` 也已经从“会制造误解的公开表面负担”收缩成了“文档已讲清、但 API 表面仍略显多余的概念税”。从**稳定行为**和**公开契约一致性**的角度看，这一轮已经没有我会继续拦着发版的缺口了。把分数留在 10 以下，只是因为我仍然认为一个稳定 dataclass 上挂着的公开字段，如果长期没有任何行为消费者，即使文档已经明确写成 informational tag，它依然会给 API 纯度留下极轻的一点余量。

## 我实际核验了什么

我重新读了 `spherical_array_processing/acoustics/radial.py` 里共享 helper、`plane_wave_radial_bn` 的 directional 分支，以及 `sph_modal_coeffs` 的调用点。核心变化现在是实打实落地的：`_validate_sphere_and_dir_coeff` 已经成为唯一的 directional 合约实现，而且通过 `arg_name` 把用户可见词汇和调用点绑在一起。这一点可以直接从 `spherical_array_processing/acoustics/radial.py:180`、`spherical_array_processing/acoustics/radial.py:317` 和 `spherical_array_processing/acoustics/radial.py:446` 看出来。

紧接着我读了 `spherical_array_processing/encoding/radial_filters.py:14`。这里的 `_validate_dir_coeff` 现在确实只剩一层很薄的 façade，没有再自带一份平行契约。与此同时，`spherical_array_processing/array/simulation.py:223` 已经删掉内联三分支检查，改为直接委托共享 helper，所以 simulation / encoding / acoustics 现在真的共用同一份 directional 规则，而不是三份表面一致、长期可能漂移的副本。

然后我复查了 `spherical_array_processing/types.py:114`。`ArrayGeometry.sensor_kind` 的文案已经明确写出它是**纯信息性字段**，稳定 simulation / encoding 栈**不会**基于它做行为分支，真正的 directional 开关仍然是 `array_type="directional"` 加 `dir_coeff`。这一步很重要，因为它把字段语义从“看起来像行为配置”拉回到了“明确声明为标签”。

最后我做了两组运行时核验。其一，我用项目当前实际可用的 Homebrew 解释器执行了全量测试，也就是 `PATH=/opt/homebrew/bin:$PATH PYTHONPATH=. python3 -m pytest -q`，结果是 **592 passed, 4 warnings**，和你给出的现状一致。其二，我单独验证了 `sph_modal_coeffs(2, np.array([1.0]), array_type="open", dir_coeff=0.5)` 的报错文本，返回的是 `ValueError: dir_coeff is only meaningful for array_type='directional', got array_type='open'`。这说明 `sph_modal_coeffs` 现在已经不再泄漏 `sphere=` 词汇，而是准确使用自己的 `array_type=` 词汇。与此同时，我也 spot-check 了 `simulate_sh_array_response` 的数值等价性，`array_type="directional", dir_coeff=0.5` 与 `array_type="cardioid"` 的输出是 `np.array_equal(...) == True`，`array_type="directional", dir_coeff=1.0` 与 `array_type="open"` 的输出也是 `np.array_equal(...) == True`，最大绝对差都等于 `0.0`。所以这里不是“近似等价”，而是**按位一致**。

## 四项残留复判

### `sensor_kind`

**判定：部分关闭。**

你这次把最关键的事情做对了。`spherical_array_processing/types.py:114` 现在明确声明 `sensor_kind` 是**纯信息性**字段，而且稳定 simulation / encoding 栈不会基于它做行为分支。这样一来，它已经不再是那种“用户看字段名会自然以为它参与行为，结果实际悄悄没用”的暗坑。

但我还是不给这一项“完全关闭”。原因也很直接：字段本体并没有获得任何稳定行为消费者，它仍然是一个挂在稳定 dataclass 表面的公开字段，只是现在文档把它解释清楚了。换句话说，**误导性问题已经基本消掉，概念负担本身还在**。这已经不再是正确性问题，也不再是容易踩坑的契约问题，但它仍然是 API 纯化层面的轻微余量。

### `dir_coeff` 校验重复分散在三处

**判定：关闭。**

这一项现在确实被你收成了一处实现、三处词汇化调用。`spherical_array_processing/acoustics/radial.py:180` 提供唯一契约实现，`spherical_array_processing/encoding/radial_filters.py:14` 只是 façade，`spherical_array_processing/array/simulation.py:223` 直接委托给它。原先那种 simulation 一套、encoding 一套、acoustics 一套的维护漂移风险，这一轮已经被真正消掉了。

### `sph_modal_coeffs` 注释和报错词汇不一致

**判定：关闭。**

这项已经完全对齐。`spherical_array_processing/acoustics/radial.py:443` 的注释说要保留调用方的 `array_type` 词汇，而实际行为现在也确实如此。我独立触发 `sph_modal_coeffs(..., array_type="open", dir_coeff=0.5)` 的异常后，看到的是 `array_type='directional'`，而不是旧的 `sphere='directional'`。注释和行为已经一致。

### `plane_wave_radial_bn` 先调 helper 后又在 `kind == 3` 内重复校验

**判定：关闭。**

这一项已经被干净地去重。`spherical_array_processing/acoustics/radial.py:317` 的 directional 分支只保留了“共享 helper 已保证契约成立”的说明，然后直接进入数值公式，没有再重复 `dir_coeff is None` 和范围检查。这里的实现现在是典型的好状态：**helper 负责契约，分支只负责公式**。

## 剩余项

我这次没有看到新的稳定表面缺口。错误消息词汇已经按调用点对齐，directional 合约已经真正单源化，`simulate_sh_array_response` 的 directional 边界等价性也保持了按位一致，全量测试计数没有变化。

唯一还留在 10 以下的东西，就是 `sensor_kind` 作为稳定公开字段，仍然没有任何行为消费者。现在它已经被文档准确地降级成 informational tag，所以我不会再把它当成一个会误伤用户的公开契约问题；但从 API 纯度看，它仍然比“要么删掉，要么进入稳定行为层”少了最后一步。这就是我把分数定在 **9.95** 而不是 **10.00** 的全部原因。
