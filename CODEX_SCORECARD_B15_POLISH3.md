# spherical-array-processing 0.4.0b15 第三轮 polish 复评分

## 结论

这一次我按你指定的范围重新读了 `ArrayGeometry` 的构造契约，也独立做了行为抽查和全量测试复跑。我的最终评分是 **10.0/10**。

核心原因很明确。前一轮把分数卡在 `9.95` 的唯一残余，是 **`sensor_kind` 仍然只是稳定公开 dataclass 上的一个信息字段，却没有进入任何稳定行为层**。这一次它已经进入了行为层，而且进入得很干净：**默认值不再是假定常量，而是由 `array_type` 自动推导；显式传入时不再只是“记下来”，而是接受构造期一致性校验；不一致会在 `ArrayGeometry(...)` 当场抛出 `ValueError`**。这已经构成了一个稳定、清晰、可测试的 **construction-time coherence invariant**。也正因为如此，前一轮留下的那个 API 纯度残余，现在我认为已经**完全关闭**。

## 复核结果

我重新读了 `spherical_array_processing/types.py` 里 `ArrayGeometry` 的字段声明、docstring 和 `__post_init__`。当前实现把 `sensor_kind` 从 `Literal["pressure", "directional"]` 改成了 `Literal["pressure", "directional"] | None = None`，随后在 `__post_init__` 中依据 `array_type` 推导出唯一合法值。对于 `"open"` 和 `"rigid"`，推导结果是 **`"pressure"`**；对于 `"cardioid"` 和 `"directional"`，推导结果是 **`"directional"`**。如果调用者显式给出的 `sensor_kind` 与推导值不一致，构造函数会抛出包含 **`inconsistent`** 的 `ValueError`。这说明 `sensor_kind` 现在已经不再是一个自由漂浮的标签，而是受 dataclass 构造规则约束的稳定契约字段。

我也重新读了 `tests/test_directional_array_plumbing.py` 里这轮新增的三组测试。参数化测试覆盖了四种 `array_type` 到 `sensor_kind` 的自动推导映射；显式一致用例验证了“允许显式声明，但必须与推导结果一致”的契约；不一致用例验证了 **`rigid + directional`** 和 **`directional + pressure`** 都会在构造时失败。测试设计和实现契约是一一对应的，没有看到“文档说了一套、测试只测了一半”的情况。

紧接着我做了你指定的行为抽查。`ArrayGeometry(radius_m=0.1, sensor_grid=<grid>, array_type="rigid").sensor_kind` 的实际结果是 **`"pressure"`**。`ArrayGeometry(..., array_type="directional", metadata={"dir_coeff": 0.5}).sensor_kind` 的实际结果是 **`"directional"`**。`ArrayGeometry(..., array_type="rigid", sensor_kind="directional")` 的实际结果是抛出 **`ValueError`**，而且错误消息明确包含 **`inconsistent`**。这三条都与目标契约完全一致。

最后我重新执行了 `PYTHONPATH=. python3 -m pytest -q`，结果是 **598 passed, 4 warnings**。也就是说，这个改动没有带来回归，而且测试总数与您给出的现状一致。

## 对 `sensor_kind` 残余项的最终判定

我的判断是：**这个残余项已经完全关闭**。

原因并不复杂。上一轮让我保留最后 `0.05` 分的，不是“这个字段必须深度参与声学公式求值”，而是“它如果作为稳定公开字段继续存在，就应该拥有一个稳定、可观察、受约束的行为角色，而不是只在文档里解释自己是什么”。现在它已经满足这个要求了。**自动推导** 让默认状态不再含糊，**构造期一致性校验** 让显式状态不再能制造内部自相矛盾的对象，而 **失败路径有明确异常语义** 让这个契约对调用者也是稳定可见的。到这一步，我认为它已经跨过了“进入行为层”的门槛。

更进一步说，当前设计还保留了很好的向后兼容性。旧调用若不显式传 `sensor_kind`，默认 `ArrayGeometry(...)` 仍然会因为默认 `array_type="rigid"` 而得到 **`sensor_kind="pressure"`**。这意味着你没有为了修复 API 纯度问题去打破既有稳定路径，这一点处理得很成熟。

## 新发现

这一次复核里，我**没有发现新的稳定表面问题**。我没有看到新的契约泄漏、测试与实现不一致，或因为 `sensor_kind` 进入行为层而引出的副作用性倒挂。相反，这个改动把字段冗余感收敛成了一个清晰的 **derived-and-validated invariant**，因此它已经不再构成评分上的残余扣分项。

如果后续还要继续吹毛求疵，那就已经不再是“是否到 10.0”的问题，而会进入“是否要做更激进的 API 极简化”这种偏风格偏哲学的讨论，比如未来要不要把 `sensor_kind` 彻底做成只读派生属性而不是可显式传入字段。但那已经不是当前版本的缺陷，也不是我愿意据此扣分的东西。
