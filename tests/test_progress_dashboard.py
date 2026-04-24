from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_missing_dependency_symbols():
    mod = _load_module(ROOT / "scripts" / "update_progress_dashboard.py")
    report = {
        "cases": [
            {"status": "skip_dependency", "matlab_error": "未定义与 'double' 类型的输入参数相对应的函数 'getSH'。"},
            {"status": "skip_dependency", "matlab_error": "undefined function 'getSH' for input arguments"},
            {"status": "skip_dependency", "matlab_error": "undefined function 'sphModalCoeffs' for input arguments"},
            {"status": "pass", "matlab_error": ""},
        ]
    }
    out = mod._extract_missing_dependency_symbols(report)
    assert out[0][0] == "getSH"
    assert out[0][1] == 2


def test_build_dashboard_markdown_contains_core_metrics():
    mod = _load_module(ROOT / "scripts" / "update_progress_dashboard.py")
    src = {
        "politis": {"count_m_files": 52},
        "rafaely": {"math_count": 20, "plot_count": 5, "fig_count": 35},
    }
    conf = {
        "case_summary": {"total": 10, "pass": 7, "fail": 1, "expected_difference": 1, "skip_dependency": 1},
        "api_summary": {"total": 20, "pass": 12, "fail": 3, "expected_difference": 2, "skip": 3},
        "cases": [],
    }
    img = {"summary": {"num_pairs": 30, "threshold_evaluated_pairs": 30, "threshold_pass_pairs": 30, "scripts_total": 17, "scripts_figure_count_match": 17}}
    md = mod._build_dashboard_markdown(src, conf, img)
    assert "Progress Dashboard" in md
    assert "Politis MATLAB 源函数文件数" in md
    assert "| 52 |" in md
    assert "可比较 Case 通过率" in md
    assert "87.5%" in md  # 7 / (7+1)
    assert "图脚本图像数量一致性" in md
    assert "EXPECTED_DIFFERENCE" in md
