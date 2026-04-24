# 并行迁移工作流（多 Codex 指挥）

这个文档对应你强调的“指挥者模式”。仓库里已经有 `scripts/build_source_inventory.py` 与 `scripts/generate_parallel_migration_tasks.py`，前者生成全量清单，后者把 MATLAB 源码按功能桶切成并行任务包。

推荐做法是先运行任务生成脚本，再按功能桶分配给不同的 Codex 实例。这样每个实例只改自己负责的模块，并且统一依赖当前主仓已经固定的基础层接口（`types`, `coords`, `sh`, `acoustics`, `repro`）。

每个子任务完成后都应至少运行仓库测试，并新增对应模块的测试用例；如果子任务涉及图形复刻，就再用 `scripts/compare_images.py` 与后续 MATLAB 导出图进行比对。这样你在“大爆炸交付”的节奏下，内部仍然能保持接口稳定和回归可控。

