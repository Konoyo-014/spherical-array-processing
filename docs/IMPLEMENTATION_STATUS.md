# 实现状态（当前迭代）

当前提交完成的是 **仓库基础架构 + 核心数学层首批实现 + 回归/复现框架骨架**。这使后续继续迁移 Politis 与 Rafaely 的全部函数时，能够在统一的接口与测试约束下推进。

已经落地的核心能力包括 **坐标变换**、**复/实球谐矩阵与系数变换**、**Rafaely 风格径向函数 `Bn/BnMat`**、**基础波束形成权重与 MVDR/LCMV**、**PWD/MUSIC 光谱**、**扩散度基础估计**、**扩散场相干矩阵（omni）**、**Matlab 风格绘图样式**，以及 **双通道到不完备 FOA 的实验性 STFT 原型**。

最近一轮已经把 Rafaely 章节脚本映射补齐到 **39/39**（`ch1` 到 `ch7` 全覆盖）。与此同时，FOA 路线新增了 **学习型估计接口** 与 **训练/评测脚本**，形成“直接法稳健估计 + 学习型替代”的双通道技术路径。

这不是完整迁移终点，但已经把计划中最容易导致后续返工的 **基础约定层** 固定下来了。

在后续迭代中，`repro.politis` 层已经继续补齐到接近完整的 **MATLAB 名字级接口封装**，包括编码滤波设计、固定/自适应波束形成、DOA、扩散度与相干性函数，以及 `differentialGains` 这类原脚本型差分系数表的 Python 入口。当前该层的价值在于，它允许在不牺牲 Pythonic 内核设计的前提下，快速对照 `TEST_SCRIPTS.m` 迁移路径。

Rafaely 侧除了数学与绘图包装器之外，也已经新增若干 **章节脚本级示例**（`ch1` 与 `ch6`），用于验证图形组织方式、样式配置与参数约定（尤其是 `theta` 作为 colatitude 的约定）在 Python 端可以稳定运行。相关脚本已经纳入 `pytest` 的 smoke tests。

目前已补上的 Rafaely 图脚本示例已经覆盖 `ch1` 的 **全部 9 个脚本**（包括 `fig_example_function`、`fig_rotation`、`fig_sinc_and_cap`、`fig_truncated_cap` 在内），以及 `ch6` 的 **全部 8 个脚本**。其中 `fig_supercardioid_beampatterns` 在 Python 端通过广义特征值问题重建了文中方向性指标曲线，计算得到的 `F` 值与 MATLAB 图标题基本一致（例如 `N=1..4` 的结果约为 `11.4, 24.0, 37.7, 51.8` dB）。与此同时，`fig_DolphChebyshev_beampattern` 已经从早期占位图升级为使用 **Dolph-Chebyshev 球谐权重** 的实际实现，`fig_WNG_open_and_rigid` 则补上了 `kr=0` 处刚性球模态数值不稳定的稳健化处理，避免图形和回归受到 `NaN` 污染。对于 MATLAB 中依赖 `fmincon` 的 `fig_multiple_objective_beampatterns`，当前 Python 版本采用 **SciPy SLSQP** 进行带等式与不等式约束的数值优化，以保持功能路径和指标语义一致。当前 `examples/rafaely/ch1` 与 `examples/rafaely/ch6` 已与 MATLAB 源脚本形成一一对应，差异仅在 Python 文件名使用小写命名风格。

图像回归基础设施也已经进入可用状态。新增脚本 `/scripts/run_rafaely_image_regression.py` 可以在同一流程中执行 **MATLAB 参考图导出、Python 图像渲染、图像对比与报告生成**，并支持章节过滤、脚本过滤、`--skip-matlab` / `--skip-python` / `--compare-only` 等模式。考虑到本机 MATLAB `-batch` 启动耗时较长，脚本提供了 `--matlab-timeout-s` 参数并在超时时继续输出报告，避免流水线长时间挂起。当前已验证 Python-only 模式可稳定生成 `ch1/ch6` 基线图像产物（约 30 张 PNG）及 JSON/Markdown 报告。

最近一次迭代中，MATLAB 回归脚本进一步接入了 `probe_matlab_cli()` 轻量探针，会在正式导出前先检查 **MATLAB CLI 是否可无交互执行命令**。这解决了一个实际环境问题：在某些安装状态下，`matlab -batch` 会因为命令行登录（MathWorks 账号）而表现为“长时间无输出”，此前只能通过超时猜测原因；现在脚本会在探测到登录提示（含中文/英文提示文本）时明确标记 `login_required`，把探针结果写入 `report.json/report.md`，并自动跳过 MATLAB 导出但继续完成 Python 渲染与报告生成，从而保持回归流水线可用性。

为了让这条诊断链在图像回归之外也能单独使用，新增了 `/scripts/check_matlab_cli.py` 作为独立预检入口，可直接输出 JSON（`--json`）并使用退出码区分 `ok` 与阻塞状态（例如 `login_required`）。这使得在启动大批量 MATLAB 图导出之前，可以先快速确认命令行环境是否已经完成登录。与此同时，新增了 `tests/test_regression_tooling.py`，覆盖 `probe_matlab_cli()` 的 `ok/login_required/timeout/not_found` 状态判定、Rafaely 回归脚本对临时 `codex_reg_*.m` wrapper 的过滤逻辑，以及 Markdown 报告中 MATLAB CLI 探针字段的写入行为，从而把图像回归工具链本身也纳入自动回归范围。

在此基础上，`/scripts/run_rafaely_image_regression.py` 还新增了 **图像门禁阈值判定与退出码** 能力，支持 `--ssim-min`、`--rmse-max`、`--mae-max`、`--use-default-thresholds`（当前接入 `FigureReproConfig.image_ssim_threshold` 默认值）以及 `--fail-on-threshold`。报告文件现在会写入阈值配置、每对图像的门禁评估结果（`PASS/FAIL/N/A`）和汇总统计（evaluated/pass/fail），这使得脚本可以从“产出指标报告”提升为“可直接用于回归门禁判定”的工具。当前已验证在本机 MATLAB CLI 仍处于 `login_required` 状态时，脚本仍能正常输出带阈值配置与阈值汇总字段的报告，并保持退出码语义一致。

最近一次迭代中，图像回归脚本的门禁语义进一步收紧，新增了 `--require-pairs`（当没有任何 MATLAB/Python 图像对被比较时返回非零退出码）和 `--require-matlab-batch-ready`（当脚本本应使用 MATLAB 导出，但 CLI 探针表明 MATLAB 不可无交互执行时返回非零退出码）。这避免了批量回归场景下“因 MATLAB 登录阻塞导致零图像对，但流程仍被误判为成功”的问题。与此同时，JSON/Markdown 报告还增加了 **按脚本汇总**（`chapter/script` 级别的 pair 数量、same-shape 数量与阈值通过/失败统计），便于快速定位失败集中在哪些 Rafaely 图脚本上，而不必逐条翻阅全部图像对。

本轮继续收紧了图像回归的验收语义，新增了 `--require-figure-count-match`，用于在 **MATLAB 与 Python 渲染图像数量不一致** 时直接返回非零退出码（当前约定为 `6`）。这修复了一个潜在假阳性来源：此前脚本对图像对比采用 `min(n_matlab, n_python)` 截断，若某一侧少导出图像，流程仍可能对已有子集给出“通过”结论。现在脚本会在 JSON/Markdown 报告中显式写出每个脚本的 `matlab_image_count / python_image_count / paired_count / figure_count_match`，并在汇总区给出 **图像数量一致性统计**（match/mismatch/total），从而把“数量一致性”也纳入回归门禁。该路径已经通过自动测试与真实 `compare-only` 伪数据 smoke 验证，实际可触发退出码 `6`。

在真实 MATLAB 登录后的联调中，又修复了一个影响导出计数的路径问题：回归脚本此前使用相对 `output-dir` 时，MATLAB wrapper 会在 `src/Rafaely/matlab/fig/ch*/` 的当前工作目录下创建相对 `artifacts/`，导致 Python 侧在仓库根目录统计不到导出的 PNG。现在 `run_rafaely_image_regression.py` 已将输出目录统一解析为绝对路径（`Path(...).expanduser().resolve()`），并新增了对应测试覆盖。修复后，带严格门禁参数（`--require-matlab-batch-ready --require-pairs --require-figure-count-match`）的实跑结果已验证通过：`ch1` 成功导出并配对 **19** 张图（9 个脚本，数量不一致 0），`ch6` 成功导出并配对 **11** 张图（8 个脚本，数量不一致 0），合并 `ch1+ch6` 报告共 **30** 对图、**17/17** 脚本图像数量一致。为适配 MATLAB 冷启动时间，脚本还新增了 `--matlab-probe-timeout-s`（默认 60 秒）避免探针被 15 秒硬限制误判。

在上述真实导出产物基础上，`compare-only` 门禁也已经可用，并且能得到明确的数值判定结果。当前使用 `--rmse-max 0.36 --mae-max 0.25 --fail-on-threshold --require-pairs --require-figure-count-match` 在 `ch1+ch6` 合并报告上重跑后，得到 **30/30** 图像对通过阈值、图像数量一致性 **17/17** 脚本匹配、退出码为 0。这意味着回归脚本现在不仅能生成可读报告，也可以作为一个带退出码语义的稳定验收门禁直接用于批处理或 CI 集成。

最近一轮又补上了 **逐函数 MATLAB 对照** 的工具化能力：新增 `/scripts/run_function_conformance.py`，会在同一 MATLAB 会话里批量执行 Rafaely/Politis 的函数级 case，把 MATLAB 输出与 Python 输出逐项比较，并输出 `PASS/FAIL/SKIP_DEPENDENCY` 报告到 `artifacts/function_conformance_live/report.json` 与 `artifacts/function_conformance_live/report.md`。在你登录 MATLAB 后的真实实跑里，当前统计为 **63 个 case：33 通过、3 数值不一致、27 因源仓缺少外部依赖而跳过**。这 27 个跳过项集中在 `getSH/getTdesign/sphModalCoeffs/unitSph2cart/gaunt_mtx` 等源仓外部函数缺失，不是 Python 端随机失败。

在同一轮修复中，已经把多个核心不一致项收敛掉，包括 **Rafaely `sh2` 与导数链路、`Bn/BnMat` 复数 Hankel 约定、`gaussian_sampling` 栅格展开顺序**，以及 Politis 侧的 **cardioid/supercardioid/maxEV 权重闭式实现**、`evaluateSHTfilters` 的复共轭计算、和 `getDiffuseness_IE/TV/SV/CMD` 的公式对齐。当前剩余的 3 个不一致分别是：`uniform_sampling`（Python 仍用 Fibonacci 近似而非 MATLAB 内置点集表）、`platonic_solid`（面索引采用 Python 0-based 约定）、`extractAxisCoeffs`（Python 保持多列输入语义，而 MATLAB 原函数对多列场景存在历史实现差异）。这些差异现在都已被报告显式标注，可作为下一轮是否“完全 MATLAB 语义兼容”的明确决策点。

为了让“现在推进到哪儿了”可视化且可复用，本轮新增了 `/scripts/update_progress_dashboard.py` 和产物文档 `docs/PROGRESS_DASHBOARD.md`。该仪表盘会自动汇总 `docs/source_inventory.json`、函数级对照报告与图像回归报告，给出 **源码规模、Case 通过率、API 通过率、图像门禁结果和主要阻塞依赖符号**。配合 `python3 scripts/run_regression.py --update-progress` 可以一条命令刷新快照，后续每轮推进都能在同一位置查看进度变化。
