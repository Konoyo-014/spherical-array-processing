# b14 A3 Review

## Verdict

**TAG**

A3 的目标现在已经成立，而且边界比最初提交时更干净。`repro`、`regression`、`experimental` 这三个子模块都清楚地声明了自己属于 **developer-only** 区域，直接导入时会发出 **一次性的 `FutureWarning`**，同时顶层 `sap` 命名空间继续把它们排除在稳定公开接口之外。更重要的是，稳定公开模块与开发专用模块之间的边界现在也被重新拉直了，普通用户走公开 API 时不会再被误伤到这类警告。

## What I Checked

我先复核了 `spherical_array_processing/__init__.py` 的文档段落。这里对 contract 的表述是准确的：它明确说明这三个子模块 **不属于稳定公共 API**，明确写出它们 **不会作为顶层属性暴露**，而且现在还补上了用户真正需要的动作建议，也就是优先使用上面列出的公开子模块；如果确实要依赖这些开发层接口，就应该 **pin 住精确版本**。这让“风险是什么”和“接下来该怎么做”在同一段里闭环了，没有停留在只做风险提示的层面。

紧接着我逐个复核了 `repro`、`regression` 和 `experimental` 的模块级 docstring 与 `warnings.warn(...)` 文案。原先的 warning 已经能说明“这不是稳定 API”，但对误用方来说还不够 **actionable**，因为它没有明确告诉用户应该退回到哪里。现在三个 warning 都把动作补齐了：`repro` 引导用户回到公开顶层子模块，`regression` 明确说终端用户代码不应依赖它，`experimental` 明确说优先选稳定 API，并且三者都给出同一个现实策略，也就是 **如果你非用不可，就 pin 精确版本**。对一个不小心写了 `import sap.repro` 的下游用户来说，这已经足够直接。

然后我专门检查了你担心的 subprocess 测试语义。`tests/test_import_contract.py` 里这个测试的总体思路是对的，因为 **warnings 的捕获是在子进程内部发生的**，所以不会被父进程里更早的 `import sap` 污染。也就是说，它确实在验证“一个全新 Python 进程里，先导入顶层包，再直接导入 developer-only 子模块，会发生什么”。不过我也看到一个真实缺口：原测试只验证“这三类 warning 都出现过”，却没有验证 **每个模块恰好只出现一次**。我已经把它补成严格断言，直接锁死为三个 warning、三个模块名，而且进一步校验 warning 文本里包含 **unstable API** 与 **pin version** 这两个关键信号。

## Issue Found And Fixed

这轮 review 里真正值得修的点，不在你新增的 developer-only 警告本身，而在一个更深的边界泄漏。`spherical_array_processing.encoding` 是稳定公开模块，但它的 `measured_filters.py` 之前直接从 `spherical_array_processing.repro.politis.functions` 导入实现。这个依赖链会让普通用户仅仅因为导入公开的 `encoding`，就看到 **developer-only 的 `FutureWarning`**。从 API 契约上说，这会把“公开稳定层”和“开发专用层”的区分冲淡掉，也会让你这次 A3 想建立的信号变得不再纯净。

我没有用“局部 suppress warning”这种表面补丁，因为那会带来另一个问题：如果公开模块先把 `repro` 静默导入进来了，用户后面再显式导入 `sap.repro` 时，反而可能因为模块缓存而看不到警告。真正的根因是 **稳定层直接依赖了开发层实现**。所以我把 `measured_array_equalizer` 需要的共享实现抽到了私有内部模块里，让公开 `encoding` 和开发专用 `repro` 都依赖这个内部实现，而不是让公开层去穿透开发层。这样一来，公开 API 不再误发 developer-only warning，而 direct import `sap.repro` 的一次性提示机制也仍然成立。

## Regression Status

我按你要求跑了聚焦测试，也跑了全量回归。聚焦集合包括 `tests/test_import_contract.py`、`tests/test_repro_layers.py`、`tests/test_foa_experimental.py`、`tests/test_foa_experimental_dl.py`，另外我补跑了直接受本次修复影响的 `tests/test_measured_equalizer.py`。这些都通过了。随后我又跑了全量 `pytest -q`，结果是 **557 passed**。相比你给出的 A3 初始状态，现在多出来的那个通过测试，就是我补上的“稳定公开子模块不应发出 developer-only `FutureWarning`”契约测试。

你特别点名的 `test_repro_layers`、`test_foa_experimental` 和 `test_foa_experimental_dl` 没有回退。它们仍然正常导入并重度使用这些 developer-only 子模块，只是在导入时会如预期发出 warning，这和 A3 想传达的语义是一致的。

## Final Read

所以我的最终判断是 **TAG**。A3 想解决的“这些子模块仍然随 wheel 分发，但必须把 **非稳定 API** 的信号说清楚”这个目标，现在已经真正完成了。更关键的是，稳定公开 API 与 developer-only API 的边界也已经被修正为自洽状态：**直接依赖 developer-only 模块会被提醒，正常使用公开模块不会被误报**。
