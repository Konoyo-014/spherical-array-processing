#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.repro import politis as po
from spherical_array_processing.repro import rafaely as rf


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_missing_dependency_symbols(conformance_report: dict[str, Any] | None) -> list[tuple[str, int]]:
    if not conformance_report:
        return []
    symbols: Counter[str] = Counter()
    for row in conformance_report.get("cases", []):
        if row.get("status") != "skip_dependency":
            continue
        err = str(row.get("matlab_error", ""))
        for pat in (
            r"函数 '([^']+)'",
            r"function '([^']+)'",
            r"variable '([^']+)'",
            r"变量 '([^']+)'",
        ):
            m = re.search(pat, err, flags=re.IGNORECASE)
            if m:
                symbols[m.group(1)] += 1
                break
    return symbols.most_common()


def _format_pct(num: int, den: int) -> str:
    if den <= 0:
        return "N/A"
    return f"{(100.0 * num / den):.1f}%"


def _build_dashboard_markdown(
    source_inventory: dict[str, Any] | None,
    conformance_report: dict[str, Any] | None,
    image_report: dict[str, Any] | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    politis_src = int(source_inventory.get("politis", {}).get("count_m_files", 0)) if source_inventory else 0
    rafaely_math = int(source_inventory.get("rafaely", {}).get("math_count", 0)) if source_inventory else 0
    rafaely_plot = int(source_inventory.get("rafaely", {}).get("plot_count", 0)) if source_inventory else 0
    rafaely_fig = int(source_inventory.get("rafaely", {}).get("fig_count", 0)) if source_inventory else 0

    csum = conformance_report.get("case_summary", {}) if conformance_report else {}
    asum = conformance_report.get("api_summary", {}) if conformance_report else {}

    case_total = int(csum.get("total", 0))
    case_pass = int(csum.get("pass", 0))
    case_fail = int(csum.get("fail", 0))
    case_expected = int(csum.get("expected_difference", 0))
    case_skip = int(csum.get("skip_dependency", 0))

    api_total = int(asum.get("total", 0))
    api_pass = int(asum.get("pass", 0))
    api_fail = int(asum.get("fail", 0))
    api_expected = int(asum.get("expected_difference", 0))
    api_skip = int(asum.get("skip", 0))

    comparable_total = case_pass + case_fail
    comparable_pass_pct = _format_pct(case_pass, comparable_total)
    api_pass_pct = _format_pct(api_pass, api_total)

    image_summary = image_report.get("summary", {}) if image_report else {}
    image_pairs = int(image_summary.get("num_pairs", 0))
    image_th_eval = int(image_summary.get("threshold_evaluated_pairs", 0))
    image_th_pass = int(image_summary.get("threshold_pass_pairs", 0))
    image_scripts_total = int(image_summary.get("scripts_total", 0))
    image_scripts_match = int(image_summary.get("scripts_figure_count_match", 0))

    missing_symbols = _extract_missing_dependency_symbols(conformance_report)

    lines: list[str] = []
    lines.append("# Progress Dashboard")
    lines.append("")
    lines.append(f"更新时间：`{now}`")
    lines.append("")
    lines.append("## 当前总览")
    lines.append("")
    lines.append(
        "这个仪表盘由源码清单、函数级 MATLAB 对照报告和图像回归报告自动汇总，"
        "用于回答“现在做到哪一步了”。"
    )
    lines.append("")
    lines.append("| 指标 | 当前值 |")
    lines.append("|---|---:|")
    lines.append(f"| Politis MATLAB 源函数文件数 | {politis_src} |")
    lines.append(f"| Rafaely 数学函数文件数 | {rafaely_math} |")
    lines.append(f"| Rafaely 绘图工具文件数 | {rafaely_plot} |")
    lines.append(f"| Rafaely 图脚本文件数 | {rafaely_fig} |")
    lines.append(f"| Python 导出 API 数（Rafaely+Politis） | {len(rf.__all__) + len(po.__all__)} |")
    lines.append(f"| 函数级对照 Case（总） | {case_total} |")
    lines.append(f"| 函数级对照 Case（PASS） | {case_pass} |")
    lines.append(f"| 函数级对照 Case（FAIL） | {case_fail} |")
    lines.append(f"| 函数级对照 Case（EXPECTED_DIFFERENCE） | {case_expected} |")
    lines.append(f"| 函数级对照 Case（SKIP_DEPENDENCY） | {case_skip} |")
    lines.append(f"| 可比较 Case 通过率（PASS/(PASS+FAIL)） | {comparable_pass_pct} |")
    lines.append(f"| API 级通过率（PASS/总API） | {api_pass_pct} |")
    lines.append(f"| 图像对照图对数 | {image_pairs} |")
    lines.append(f"| 图像阈值评估通过数 | {image_th_pass}/{image_th_eval} |")
    lines.append(f"| 图脚本图像数量一致性 | {image_scripts_match}/{image_scripts_total} |")
    lines.append("")
    lines.append("## 函数级对照状态")
    lines.append("")
    if conformance_report:
        if case_fail > 0:
            tail = " 当前还有少量 FAIL 和一批源仓外部依赖导致的 SKIP。"
        elif case_expected > 0:
            tail = " 当前严格 FAIL 已清零，仍有语义差异白名单项和源仓外部依赖导致的 SKIP。"
        else:
            tail = " 当前严格 FAIL 已清零，剩余主要是源仓外部依赖导致的 SKIP。"
        lines.append(f"最新函数级报告：`{ROOT / 'artifacts' / 'function_conformance_live' / 'report.json'}`。{tail}")
    else:
        lines.append("尚未发现函数级对照报告。")
    lines.append("")
    lines.append("| API 状态 | 数量 |")
    lines.append("|---|---:|")
    lines.append(f"| PASS | {api_pass} |")
    lines.append(f"| FAIL | {api_fail} |")
    lines.append(f"| EXPECTED_DIFFERENCE | {api_expected} |")
    lines.append(f"| SKIP | {api_skip} |")
    lines.append("")
    if missing_symbols:
        lines.append("最常见的 MATLAB 外部依赖缺失符号如下，这些是当前 SKIP 的主因。")
        lines.append("")
        lines.append("| 缺失符号 | 触发次数 |")
        lines.append("|---|---:|")
        for sym, cnt in missing_symbols[:12]:
            lines.append(f"| `{sym}` | {cnt} |")
        lines.append("")
    lines.append("## 图像回归状态")
    lines.append("")
    if image_report:
        lines.append(
            f"最新图像回归报告：`{ROOT / 'artifacts' / 'rafaely_image_regression_ch1_ch6_live' / 'report.json'}`。"
            " 当前章节 `ch1+ch6` 已完成成对比较与数量一致性检查。"
        )
    else:
        lines.append("尚未发现图像回归报告。")
    lines.append("")
    lines.append("## 解释与下一步")
    lines.append("")
    lines.append(
        "剩余建设重点在两条线上同步推进。第一条线是继续减少函数级 FAIL，"
        "把语义差异项逐步收敛到可选兼容模式。第二条线是补齐 MATLAB 源仓缺失依赖的对照桥接，"
        "把当前 SKIP 的 case 变为可执行的可比较 case。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="更新进度仪表盘文档")
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=ROOT / "docs" / "source_inventory.json",
    )
    parser.add_argument(
        "--conformance-report",
        type=Path,
        default=ROOT / "artifacts" / "function_conformance_live" / "report.json",
    )
    parser.add_argument(
        "--image-report",
        type=Path,
        default=ROOT / "artifacts" / "rafaely_image_regression_ch1_ch6_live" / "report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "PROGRESS_DASHBOARD.md",
    )
    args = parser.parse_args(argv)

    source_inventory = _load_json(args.source_inventory)
    conformance_report = _load_json(args.conformance_report)
    image_report = _load_json(args.image_report)

    text = _build_dashboard_markdown(source_inventory, conformance_report, image_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
