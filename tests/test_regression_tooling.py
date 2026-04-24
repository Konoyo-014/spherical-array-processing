from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from spherical_array_processing.regression import matlab as matlab_mod
from spherical_array_processing.regression.matlab import MatlabProbeResult, MatlabRuntime


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_probe_matlab_cli_detects_login_required(monkeypatch):
    monkeypatch.setattr(matlab_mod, "detect_matlab", lambda: MatlabRuntime(executable="/tmp/matlab", source="test"))

    cp = subprocess.CompletedProcess(
        args=["matlab"],
        returncode=1,
        stdout="请输入您的 MathWorks 帐户电子邮件地址，然后按 Enter 键:\n登录失败。 是否要重试? y/n [n]\n",
        stderr="",
    )
    monkeypatch.setattr(matlab_mod.subprocess, "run", lambda *a, **k: cp)

    out = matlab_mod.probe_matlab_cli(timeout_s=1)
    assert out.status == "login_required"
    assert "login" in out.message.lower()
    assert "MathWorks" in out.stdout_tail


def test_probe_matlab_cli_detects_ok(monkeypatch):
    monkeypatch.setattr(matlab_mod, "detect_matlab", lambda: MatlabRuntime(executable="/tmp/matlab", source="test"))
    cp = subprocess.CompletedProcess(
        args=["matlab"],
        returncode=0,
        stdout="CODEX_MATLAB_PROBE_OK\n",
        stderr="",
    )
    monkeypatch.setattr(matlab_mod.subprocess, "run", lambda *a, **k: cp)

    out = matlab_mod.probe_matlab_cli(timeout_s=1)
    assert out.status == "ok"


def test_probe_matlab_cli_timeout(monkeypatch):
    monkeypatch.setattr(matlab_mod, "detect_matlab", lambda: MatlabRuntime(executable="/tmp/matlab", source="test"))

    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["matlab"], timeout=1, output="partial-out", stderr="partial-err")

    monkeypatch.setattr(matlab_mod.subprocess, "run", _raise)
    out = matlab_mod.probe_matlab_cli(timeout_s=1)
    assert out.status == "timeout"
    assert "partial-out" in out.stdout_tail
    assert "partial-err" in out.stderr_tail


def test_probe_matlab_cli_not_found(monkeypatch):
    monkeypatch.setattr(matlab_mod, "detect_matlab", lambda: None)
    out = matlab_mod.probe_matlab_cli(timeout_s=1)
    assert out.status == "not_found"


def test_regression_script_discovers_and_filters_temp_wrappers(tmp_path, monkeypatch):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    fake_root = tmp_path
    ch1 = fake_root / "src" / "Rafaely" / "matlab" / "fig" / "ch1"
    ch1.mkdir(parents=True)
    (ch1 / "fig_Pn.m").write_text("% test", encoding="utf-8")
    (ch1 / "codex_reg_tmp.m").write_text("% temp", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", fake_root)

    discovered = mod._discover_matlab_scripts(["ch1"])
    assert [p.name for p in discovered["ch1"]] == ["fig_Pn.m"]


def test_regression_markdown_report_includes_probe(tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    md_path = tmp_path / "report.md"
    metadata = {
        "output_dir": str(tmp_path),
        "chapters": ["ch1"],
        "matlab_available": True,
        "matlab_probe": {
            "status": "login_required",
            "message": "MATLAB CLI appears to require interactive MathWorks account login",
        },
    }
    mod._write_markdown_summary(md_path, pairs=[], metadata=metadata)
    text = md_path.read_text(encoding="utf-8")
    assert "MATLAB CLI探针状态" in text
    assert "login_required" in text


def test_threshold_evaluation_helpers_pass_fail():
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    pair_ok = mod.PairResult(
        chapter="ch1",
        script="fig_Pn",
        figure_index=1,
        matlab_path="m.png",
        python_path="p.png",
        same_shape=True,
        metrics={"rmse": 0.01, "mae": 0.005, "ssim": 0.99},
        metrics_resized_to_ref=None,
    )
    pair_fail = mod.PairResult(
        chapter="ch1",
        script="fig_Pn",
        figure_index=2,
        matlab_path="m2.png",
        python_path="p2.png",
        same_shape=True,
        metrics={"rmse": 0.2, "mae": 0.05, "ssim": 0.8},
        metrics_resized_to_ref=None,
    )
    thresholds = mod._threshold_config(ssim_min=0.95, rmse_max=0.1, mae_max=0.02)
    e_ok = mod._evaluate_pair_thresholds(pair_ok, thresholds)
    e_fail = mod._evaluate_pair_thresholds(pair_fail, thresholds)
    assert e_ok["evaluated"] is True and e_ok["pass"] is True
    assert e_fail["evaluated"] is True and e_fail["pass"] is False
    summary = mod._summarize_threshold_results([pair_ok, pair_fail], thresholds)
    assert summary["threshold_evaluated_pairs"] == 2
    assert summary["threshold_pass_pairs"] == 1
    assert summary["threshold_fail_pairs"] == 1


def test_per_script_summary_groups_pairs():
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    thresholds = mod._threshold_config(ssim_min=0.95, rmse_max=None, mae_max=None)
    pairs = [
        mod.PairResult("ch1", "fig_Pn", 1, "m1", "p1", True, {"ssim": 0.99, "rmse": 0.0, "mae": 0.0}, None),
        mod.PairResult("ch1", "fig_Pn", 2, "m2", "p2", True, {"ssim": 0.80, "rmse": 0.2, "mae": 0.1}, None),
        mod.PairResult("ch1", "fig_Pnm", 1, "m3", "p3", True, {"ssim": 0.97, "rmse": 0.01, "mae": 0.01}, None),
    ]
    rows = mod._summarize_pairs_by_script(pairs, thresholds)
    assert len(rows) == 2
    row_pn = next(r for r in rows if r["script"] == "fig_Pn")
    assert row_pn["num_pairs"] == 2
    assert row_pn["threshold_pass_pairs"] == 1
    assert row_pn["threshold_fail_pairs"] == 1


def test_build_per_script_comparison_summary_includes_count_mismatch(tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    discovered = {"ch1": [Path("/x/fig_Pn.m"), Path("/x/fig_Pnm.m")]}
    (tmp_path / "matlab" / "ch1").mkdir(parents=True)
    (tmp_path / "python" / "ch1").mkdir(parents=True)
    # fig_Pn: MATLAB 2, Python 1 (mismatch)
    for i in (1, 2):
        (tmp_path / "matlab" / "ch1" / f"fig_Pn_{i:02d}.png").write_bytes(b"x")
    (tmp_path / "python" / "ch1" / "fig_Pn_01.png").write_bytes(b"x")
    # fig_Pnm: both zero (match)
    rows = mod._build_per_script_comparison_summary(
        discovered=discovered,
        script_filter=set(),
        out_root=tmp_path,
        thresholds=mod._threshold_config(None, None, None),
        pairs=[],
    )
    assert len(rows) == 2
    row = next(r for r in rows if r["script"] == "fig_Pn")
    assert row["matlab_image_count"] == 2
    assert row["python_image_count"] == 1
    assert row["figure_count_match"] is False
    count_sum = mod._summarize_script_count_mismatches(rows)
    assert count_sum["scripts_figure_count_mismatch"] == 1


def test_regression_markdown_report_includes_threshold_summary(tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    md_path = tmp_path / "report.md"
    pair_ok = mod.PairResult(
        chapter="ch1",
        script="fig_Pn",
        figure_index=1,
        matlab_path="m.png",
        python_path="p.png",
        same_shape=True,
        metrics={"rmse": 0.01, "mae": 0.005, "ssim": 0.99},
        metrics_resized_to_ref=None,
    )
    metadata = {"output_dir": str(tmp_path), "chapters": ["ch1"], "matlab_available": False, "matlab_probe": None}
    thresholds = mod._threshold_config(ssim_min=0.95, rmse_max=None, mae_max=None)
    per_script_rows = [
        {
            "chapter": "ch1",
            "script": "fig_Pn",
            "matlab_image_count": 1,
            "python_image_count": 1,
            "paired_count": 1,
            "figure_count_match": True,
            "num_pairs": 1,
            "num_same_shape": 1,
            "num_with_metrics": 1,
            "num_with_resized_metrics": 0,
            "threshold_evaluated_pairs": 1,
            "threshold_pass_pairs": 1,
            "threshold_fail_pairs": 0,
        }
    ]
    mod._write_markdown_summary(md_path, pairs=[pair_ok], metadata=metadata, thresholds=thresholds, per_script_rows=per_script_rows)
    text = md_path.read_text(encoding="utf-8")
    assert "阈值评估统计" in text
    assert "PASS" in text
    assert "按脚本汇总" in text
    assert "count_match=True" in text


def test_check_matlab_cli_script_json_output(monkeypatch, capsys):
    mod = _load_module(ROOT / "scripts" / "check_matlab_cli.py")
    monkeypatch.setattr(mod, "detect_matlab", lambda: MatlabRuntime(executable="/tmp/matlab", source="test"))
    monkeypatch.setattr(
        mod,
        "probe_matlab_cli",
        lambda timeout_s=15: MatlabProbeResult(
            status="login_required",
            message="need login",
            stdout_tail="email address",
            stderr_tail="",
        ),
    )
    rc = mod.main(["--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 2
    assert payload["probe"]["status"] == "login_required"
    assert payload["matlab_runtime"]["source"] == "test"


def test_regression_script_fail_on_threshold_exit_code(monkeypatch, tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    monkeypatch.setattr(mod, "detect_matlab", lambda: None)
    monkeypatch.setattr(mod, "_discover_matlab_scripts", lambda chapters: {})
    monkeypatch.setattr(
        mod,
        "_write_report",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        mod,
        "_write_markdown_summary",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("# x", encoding="utf-8"),
    )
    monkeypatch.setattr(
        mod,
        "_summarize_threshold_results",
        lambda pairs, thresholds: {"threshold_evaluated_pairs": 1, "threshold_pass_pairs": 0, "threshold_fail_pairs": 1},
    )
    rc = mod.main(
        [
            "--compare-only",
            "--output-dir",
            str(tmp_path / "out"),
            "--fail-on-threshold",
            "--ssim-min",
            "0.95",
        ]
    )
    assert rc == 3


def test_regression_script_require_pairs_exit_code(monkeypatch, tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    monkeypatch.setattr(mod, "detect_matlab", lambda: None)
    monkeypatch.setattr(mod, "_discover_matlab_scripts", lambda chapters: {})
    monkeypatch.setattr(
        mod,
        "_write_report",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        mod,
        "_write_markdown_summary",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("# x", encoding="utf-8"),
    )
    rc = mod.main(["--compare-only", "--output-dir", str(tmp_path / "out"), "--require-pairs"])
    assert rc == 5


def test_regression_script_require_matlab_batch_ready_exit_code(monkeypatch, tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")

    class DummyRt:
        executable = "/tmp/matlab"
        source = "test"

    monkeypatch.setattr(mod, "detect_matlab", lambda: DummyRt())
    monkeypatch.setattr(
        mod,
        "probe_matlab_cli",
        lambda timeout_s=15: MatlabProbeResult(status="login_required", message="need login", stdout_tail="", stderr_tail=""),
    )
    monkeypatch.setattr(mod, "_discover_matlab_scripts", lambda chapters: {})
    monkeypatch.setattr(
        mod,
        "_write_report",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        mod,
        "_write_markdown_summary",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("# x", encoding="utf-8"),
    )
    rc = mod.main(["--output-dir", str(tmp_path / "out"), "--skip-python", "--require-matlab-batch-ready"])
    assert rc == 4


def test_regression_script_require_figure_count_match_exit_code(monkeypatch, tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")
    monkeypatch.setattr(mod, "detect_matlab", lambda: None)
    monkeypatch.setattr(mod, "_discover_matlab_scripts", lambda chapters: {})
    monkeypatch.setattr(
        mod,
        "_build_per_script_comparison_summary",
        lambda discovered, script_filter, out_root, thresholds, pairs: [
            {
                "chapter": "ch1",
                "script": "fig_Pn",
                "matlab_image_count": 2,
                "python_image_count": 1,
                "paired_count": 1,
                "figure_count_match": False,
                "num_pairs": 1,
                "num_same_shape": 1,
                "num_with_metrics": 1,
                "num_with_resized_metrics": 0,
                "threshold_evaluated_pairs": 1,
                "threshold_pass_pairs": 1,
                "threshold_fail_pairs": 0,
            }
        ],
    )
    monkeypatch.setattr(
        mod,
        "_write_report",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        mod,
        "_write_markdown_summary",
        lambda path, pairs, metadata, thresholds=None, per_script_rows=None: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text("# x", encoding="utf-8"),
    )
    rc = mod.main(["--compare-only", "--output-dir", str(tmp_path / "out"), "--require-figure-count-match"])
    assert rc == 6


def test_regression_main_resolves_output_dir_for_matlab_export(monkeypatch, tmp_path):
    mod = _load_module(ROOT / "scripts" / "run_rafaely_image_regression.py")

    class DummyRt:
        executable = "/tmp/matlab"
        source = "test"

    monkeypatch.setattr(mod, "detect_matlab", lambda: DummyRt())
    monkeypatch.setattr(
        mod,
        "probe_matlab_cli",
        lambda timeout_s=15: MatlabProbeResult(status="ok", message="ok", stdout_tail="", stderr_tail=""),
    )

    src_m = tmp_path / "src" / "Rafaely" / "matlab" / "fig" / "ch1" / "fig_Pn.m"
    src_m.parent.mkdir(parents=True, exist_ok=True)
    src_m.write_text("% dummy", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "_discover_matlab_scripts", lambda chapters: {"ch1": [src_m]})
    monkeypatch.setattr(mod, "_python_example_path_for_matlab_script", lambda chapter, matlab_script: tmp_path / "missing.py")

    captured = {"outdir": None}

    def fake_run(matlab_script, outdir, base_name, timeout_s=180):
        captured["outdir"] = outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / f"{base_name}_01.png").write_bytes(b"x")
        return 1, ""

    monkeypatch.setattr(mod, "_run_matlab_script_and_save", fake_run)

    rc = mod.main(["--chapters", "ch1", "--scripts", "fig_Pn", "--output-dir", "artifacts/relative_out", "--skip-python"])
    assert rc == 0
    assert captured["outdir"] is not None
    assert Path(captured["outdir"]).is_absolute()
