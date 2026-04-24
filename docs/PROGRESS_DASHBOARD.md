# Progress Dashboard

更新时间：`2026-02-27 17:29:13`

## 当前总览

这个仪表盘由源码清单、函数级 MATLAB 对照报告和图像回归报告自动汇总，用于回答“现在做到哪一步了”。

| 指标 | 当前值 |
|---|---:|
| Politis MATLAB 源函数文件数 | 52 |
| Rafaely 数学函数文件数 | 19 |
| Rafaely 绘图工具文件数 | 5 |
| Rafaely 图脚本文件数 | 39 |
| Python 导出 API 数（Rafaely+Politis） | 89 |
| 函数级对照 Case（总） | 71 |
| 函数级对照 Case（PASS） | 68 |
| 函数级对照 Case（FAIL） | 0 |
| 函数级对照 Case（EXPECTED_DIFFERENCE） | 3 |
| 函数级对照 Case（SKIP_DEPENDENCY） | 0 |
| 可比较 Case 通过率（PASS/(PASS+FAIL)） | 100.0% |
| API 级通过率（PASS/总API） | 84.3% |
| 图像对照图对数 | 30 |
| 图像阈值评估通过数 | 30/30 |
| 图脚本图像数量一致性 | 17/17 |

## 函数级对照状态

最新函数级报告：`/Users/konoyo/Desktop/CodexWorkspace/Spherical_Array_Process_Python/artifacts/function_conformance_live/report.json`。 当前严格 FAIL 已清零，仍有语义差异白名单项和源仓外部依赖导致的 SKIP。

| API 状态 | 数量 |
|---|---:|
| PASS | 75 |
| FAIL | 0 |
| EXPECTED_DIFFERENCE | 3 |
| SKIP | 11 |

## 图像回归状态

最新图像回归报告：`/Users/konoyo/Desktop/CodexWorkspace/Spherical_Array_Process_Python/artifacts/rafaely_image_regression_ch1_ch6_live/report.json`。 当前章节 `ch1+ch6` 已完成成对比较与数量一致性检查。

## 解释与下一步

剩余建设重点在两条线上同步推进。第一条线是继续减少函数级 FAIL，把语义差异项逐步收敛到可选兼容模式。第二条线是补齐 MATLAB 源仓缺失依赖的对照桥接，把当前 SKIP 的 case 变为可执行的可比较 case。

