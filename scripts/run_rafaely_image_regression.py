#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spherical_array_processing.regression.image_compare import compare_grayscale_images, load_image_gray
from spherical_array_processing.regression.matlab import detect_matlab, probe_matlab_cli, run_matlab_batch
from spherical_array_processing.types import FigureReproConfig

try:
    from scipy.ndimage import zoom as nd_zoom
except Exception:  # pragma: no cover
    nd_zoom = None


CHAPTERS = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7")


@dataclass
class PairResult:
    chapter: str
    script: str
    figure_index: int
    matlab_path: str
    python_path: str
    same_shape: bool
    metrics: dict[str, float] | None
    metrics_resized_to_ref: dict[str, float] | None


def _threshold_config(
    ssim_min: float | None,
    rmse_max: float | None,
    mae_max: float | None,
) -> dict[str, float | None]:
    return {"ssim_min": ssim_min, "rmse_max": rmse_max, "mae_max": mae_max}


def _pair_metrics_for_evaluation(p: PairResult) -> tuple[str | None, dict[str, float] | None]:
    if p.metrics is not None:
        return "native", p.metrics
    if p.metrics_resized_to_ref is not None:
        return "resized_to_ref", p.metrics_resized_to_ref
    return None, None


def _evaluate_pair_thresholds(
    p: PairResult,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    mode, metrics = _pair_metrics_for_evaluation(p)
    out: dict[str, Any] = {
        "mode": mode,
        "evaluated": False,
        "pass": None,
        "checks": {},
        "reason": None,
    }
    if metrics is None:
        out["reason"] = "no comparable metrics available"
        return out

    active_checks = 0
    passed_checks = 0

    ssim_min = thresholds.get("ssim_min")
    if ssim_min is not None:
        if "ssim" in metrics:
            ok = float(metrics["ssim"]) >= float(ssim_min)
            out["checks"]["ssim"] = {"value": float(metrics["ssim"]), "min": float(ssim_min), "pass": ok}
            active_checks += 1
            passed_checks += int(ok)
        else:
            out["checks"]["ssim"] = {"value": None, "min": float(ssim_min), "pass": None, "reason": "ssim unavailable"}

    rmse_max = thresholds.get("rmse_max")
    if rmse_max is not None:
        if "rmse" in metrics:
            ok = float(metrics["rmse"]) <= float(rmse_max)
            out["checks"]["rmse"] = {"value": float(metrics["rmse"]), "max": float(rmse_max), "pass": ok}
            active_checks += 1
            passed_checks += int(ok)
        else:
            out["checks"]["rmse"] = {"value": None, "max": float(rmse_max), "pass": None, "reason": "rmse unavailable"}

    mae_max = thresholds.get("mae_max")
    if mae_max is not None:
        if "mae" in metrics:
            ok = float(metrics["mae"]) <= float(mae_max)
            out["checks"]["mae"] = {"value": float(metrics["mae"]), "max": float(mae_max), "pass": ok}
            active_checks += 1
            passed_checks += int(ok)
        else:
            out["checks"]["mae"] = {"value": None, "max": float(mae_max), "pass": None, "reason": "mae unavailable"}

    if active_checks == 0:
        out["reason"] = "no active thresholds configured or available"
        return out

    out["evaluated"] = True
    out["pass"] = passed_checks == active_checks
    return out


def _summarize_threshold_results(pairs: list[PairResult], thresholds: dict[str, float | None]) -> dict[str, int]:
    evaluated = 0
    passed = 0
    failed = 0
    for p in pairs:
        e = _evaluate_pair_thresholds(p, thresholds)
        if not e["evaluated"]:
            continue
        evaluated += 1
        if e["pass"]:
            passed += 1
        else:
            failed += 1
    return {
        "threshold_evaluated_pairs": evaluated,
        "threshold_pass_pairs": passed,
        "threshold_fail_pairs": failed,
    }


def _summarize_pairs_by_script(pairs: list[PairResult], thresholds: dict[str, float | None]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[PairResult]] = {}
    for p in pairs:
        buckets.setdefault((p.chapter, p.script), []).append(p)

    rows: list[dict[str, Any]] = []
    for (chapter, script), items in sorted(buckets.items()):
        th_sum = _summarize_threshold_results(items, thresholds)
        rows.append(
            {
                "chapter": chapter,
                "script": script,
                "num_pairs": len(items),
                "num_same_shape": sum(1 for p in items if p.same_shape),
                "num_with_metrics": sum(1 for p in items if p.metrics is not None),
                "num_with_resized_metrics": sum(1 for p in items if p.metrics_resized_to_ref is not None),
                **th_sum,
            }
        )
    return rows


def _build_per_script_comparison_summary(
    discovered: dict[str, list[Path]],
    script_filter: set[str],
    out_root: Path,
    thresholds: dict[str, float | None],
    pairs: list[PairResult],
) -> list[dict[str, Any]]:
    pair_rows = {
        (row["chapter"], row["script"]): row for row in _summarize_pairs_by_script(pairs, thresholds)
    }
    rows: list[dict[str, Any]] = []
    for ch, scripts in discovered.items():
        for mfile in scripts:
            if script_filter and mfile.stem not in script_filter:
                continue
            stem = mfile.stem
            m_dir = out_root / "matlab" / ch
            p_dir = out_root / "python" / ch
            m_count = len(list(m_dir.glob(f"{stem}_*.png"))) if m_dir.exists() else 0
            p_count = len(list(p_dir.glob(f"{stem}_*.png"))) if p_dir.exists() else 0
            pair_row = pair_rows.get((ch, stem), {})
            rows.append(
                {
                    "chapter": ch,
                    "script": stem,
                    "matlab_image_count": m_count,
                    "python_image_count": p_count,
                    "paired_count": min(m_count, p_count),
                    "figure_count_match": m_count == p_count,
                    "num_pairs": int(pair_row.get("num_pairs", 0)),
                    "num_same_shape": int(pair_row.get("num_same_shape", 0)),
                    "num_with_metrics": int(pair_row.get("num_with_metrics", 0)),
                    "num_with_resized_metrics": int(pair_row.get("num_with_resized_metrics", 0)),
                    "threshold_evaluated_pairs": int(pair_row.get("threshold_evaluated_pairs", 0)),
                    "threshold_pass_pairs": int(pair_row.get("threshold_pass_pairs", 0)),
                    "threshold_fail_pairs": int(pair_row.get("threshold_fail_pairs", 0)),
                }
            )
    return rows


def _summarize_script_count_mismatches(per_script_rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(per_script_rows)
    mismatches = sum(1 for r in per_script_rows if not r.get("figure_count_match", False))
    return {
        "scripts_total": total,
        "scripts_figure_count_match": total - mismatches,
        "scripts_figure_count_mismatch": mismatches,
    }


def _discover_matlab_scripts(chapters: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for ch in chapters:
        src = ROOT / "src" / "Rafaely" / "matlab" / "fig" / ch
        out[ch] = sorted(p for p in src.glob("*.m") if not p.name.startswith("codex_reg_"))
    return out


def _python_example_path_for_matlab_script(chapter: str, matlab_script: Path) -> Path:
    # Repository uses lowercase Python file names.
    return ROOT / "examples" / "rafaely" / chapter / f"{matlab_script.stem.lower()}.py"


def _load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_open_figures(outdir: Path, base_name: str, dpi: int = 150) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    fig_nums = sorted(plt.get_fignums())
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        target = outdir / f"{base_name}_{i:02d}.png"
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
    count = len(fig_nums)
    plt.close("all")
    return count


def _run_python_example_and_save(py_example: Path, outdir: Path, base_name: str) -> int:
    mod = _load_module_from_path(py_example)
    if not hasattr(mod, "main"):
        raise RuntimeError(f"{py_example} has no main()")
    main_fn = getattr(mod, "main")
    sig = inspect.signature(main_fn)
    kwargs: dict[str, Any] = {}
    if "show" in sig.parameters:
        kwargs["show"] = False
    if "print_table" in sig.parameters:
        kwargs["print_table"] = False
    # For very heavy scripts, use a lighter smoke configuration during regression rendering
    if py_example.stem == "fig_truncated_cap":
        if "max_order" in sig.parameters:
            kwargs["max_order"] = 24
        if "orders" in sig.parameters:
            kwargs["orders"] = (4, 8, 16, 24)
        if "resolution" in sig.parameters:
            kwargs["resolution"] = 18
    if py_example.stem == "fig_sh_balloon":
        if "resolution" in sig.parameters:
            kwargs["resolution"] = 24
    main_fn(**kwargs)
    return _save_open_figures(outdir, base_name)


def _matlab_wrapper_contents(script_name: str, outdir: Path, base_name: str, dpi: int = 150) -> str:
    outdir_posix = outdir.as_posix()
    # Code generated as a script so `clear all` inside target script does not break execution;
    # we re-declare variables only after `run(...)`.
    return f"""
try
    close all force;
    set(0,'DefaultFigureVisible','off');
    run('{script_name}');
    outdir = '{outdir_posix}';
    if ~exist(outdir,'dir'); mkdir(outdir); end
    figs = findall(0,'Type','figure');
    figs = sort(figs);
    for i = 1:numel(figs)
        f = figs(i);
        set(f,'PaperPositionMode','auto');
        fname = fullfile(outdir, sprintf('{base_name}_%02d.png', i));
        print(f, fname, '-dpng', '-r{dpi}');
    end
    close all force;
catch ME
    disp(getReport(ME,'extended'));
    exit(1);
end
""".strip()


def _run_matlab_script_and_save(matlab_script: Path, outdir: Path, base_name: str, timeout_s: int = 180) -> tuple[int, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    cwd = matlab_script.parent
    wrapper_body = _matlab_wrapper_contents(matlab_script.name, outdir, base_name)
    with tempfile.NamedTemporaryFile("w", suffix=".m", prefix="codex_reg_", dir=cwd, delete=False) as tf:
        tf.write(wrapper_body)
        wrapper_path = Path(tf.name)
    try:
        cmd_expr = f"run('{wrapper_path.name}')"
        cp = run_matlab_batch(cmd_expr, cwd=cwd, timeout_s=timeout_s)
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        if cp.returncode != 0:
            raise RuntimeError(f"MATLAB failed for {matlab_script.name}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        count = len(list(outdir.glob(f"{base_name}_*.png")))
        return count, stdout
    finally:
        try:
            wrapper_path.unlink(missing_ok=True)
        except Exception:
            pass


def _resize_to_shape(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray | None:
    if img.shape == target_shape:
        return img
    if nd_zoom is None:
        return None
    z0 = target_shape[0] / img.shape[0]
    z1 = target_shape[1] / img.shape[1]
    return nd_zoom(img, (z0, z1), order=1)


def _compare_outputs(chapter: str, script_stem: str, matlab_dir: Path, python_dir: Path) -> list[PairResult]:
    matlab_imgs = sorted(matlab_dir.glob(f"{script_stem}_*.png"))
    python_imgs = sorted(python_dir.glob(f"{script_stem}_*.png"))
    results: list[PairResult] = []
    n = min(len(matlab_imgs), len(python_imgs))
    for i in range(n):
        ref_p = matlab_imgs[i]
        cand_p = python_imgs[i]
        ref = load_image_gray(ref_p)
        cand = load_image_gray(cand_p)
        same_shape = ref.shape == cand.shape
        metrics = None
        metrics_resized = None
        if same_shape:
            metrics = compare_grayscale_images(ref, cand)
        else:
            cand_r = _resize_to_shape(cand, ref.shape)
            if cand_r is not None:
                metrics_resized = compare_grayscale_images(ref, cand_r)
        results.append(
            PairResult(
                chapter=chapter,
                script=script_stem,
                figure_index=i + 1,
                matlab_path=str(ref_p),
                python_path=str(cand_p),
                same_shape=same_shape,
                metrics=metrics,
                metrics_resized_to_ref=metrics_resized,
            )
        )
    return results


def _write_report(
    report_path: Path,
    pairs: list[PairResult],
    metadata: dict[str, Any],
    thresholds: dict[str, float | None] | None = None,
    per_script_rows: list[dict[str, Any]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    th = thresholds or _threshold_config(None, None, None)
    pair_payload = []
    for p in pairs:
        d = asdict(p)
        d["threshold_eval"] = _evaluate_pair_thresholds(p, th)
        pair_payload.append(d)
    payload = {
        "metadata": metadata,
        "thresholds": th,
        "per_script_summary": per_script_rows if per_script_rows is not None else _summarize_pairs_by_script(pairs, th),
        "pairs": pair_payload,
        "summary": {
            "num_pairs": len(pairs),
            "num_same_shape": sum(1 for p in pairs if p.same_shape),
            "num_with_metrics": sum(1 for p in pairs if p.metrics is not None),
            "num_with_resized_metrics": sum(1 for p in pairs if p.metrics_resized_to_ref is not None),
            **_summarize_threshold_results(pairs, th),
            **_summarize_script_count_mismatches(per_script_rows or []),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown_summary(
    md_path: Path,
    pairs: list[PairResult],
    metadata: dict[str, Any],
    thresholds: dict[str, float | None] | None = None,
    per_script_rows: list[dict[str, Any]] | None = None,
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    th = thresholds or _threshold_config(None, None, None)
    lines: list[str] = []
    lines.append("# Rafaely 图像回归报告")
    lines.append("")
    lines.append(f"输出目录: `{metadata['output_dir']}`")
    lines.append(f"章节: `{', '.join(metadata['chapters'])}`")
    lines.append(f"MATLAB可用: `{metadata['matlab_available']}`")
    probe = metadata.get("matlab_probe")
    if probe is not None:
        lines.append(f"MATLAB CLI探针状态: `{probe.get('status')}`")
        lines.append(f"MATLAB CLI探针信息: `{probe.get('message')}`")
    lines.append(
        "阈值配置: "
        f"`ssim_min={th.get('ssim_min')}, rmse_max={th.get('rmse_max')}, mae_max={th.get('mae_max')}`"
    )
    lines.append("")
    lines.append(f"图像对数量: **{len(pairs)}**")
    same_shape = sum(1 for p in pairs if p.same_shape)
    lines.append(f"同尺寸对数量: **{same_shape}**")
    th_sum = _summarize_threshold_results(pairs, th)
    count_sum = _summarize_script_count_mismatches(per_script_rows or [])
    lines.append(
        "阈值评估统计: "
        f"**evaluated={th_sum['threshold_evaluated_pairs']} / pass={th_sum['threshold_pass_pairs']} / fail={th_sum['threshold_fail_pairs']}**"
    )
    if count_sum["scripts_total"] > 0:
        lines.append(
            "图像数量一致性: "
            f"**match={count_sum['scripts_figure_count_match']} / mismatch={count_sum['scripts_figure_count_mismatch']} / total={count_sum['scripts_total']}**"
        )
    lines.append("")
    rows = per_script_rows if per_script_rows is not None else _summarize_pairs_by_script(pairs, th)
    if rows:
        lines.append("## 按脚本汇总")
        lines.append("")
        for row in rows:
            base = f"- `{row['chapter']}/{row['script']}`: pairs={row['num_pairs']}, same_shape={row['num_same_shape']}"
            if "matlab_image_count" in row and "python_image_count" in row:
                base += (
                    f", images(matlab/python/paired)="
                    f"{row['matlab_image_count']}/{row['python_image_count']}/{row['paired_count']}"
                    f", count_match={row.get('figure_count_match')}"
                )
            base += (
                ", "
                f"threshold(pass/fail/evaluated)="
                f"{row['threshold_pass_pairs']}/{row['threshold_fail_pairs']}/{row['threshold_evaluated_pairs']}"
            )
            lines.append(base)
        lines.append("")
    for p in pairs:
        th_eval = _evaluate_pair_thresholds(p, th)
        lines.append(f"## {p.chapter}/{p.script} #{p.figure_index:02d}")
        lines.append("")
        lines.append(f"MATLAB: `{p.matlab_path}`")
        lines.append(f"Python: `{p.python_path}`")
        if p.metrics is not None:
            lines.append(f"同尺寸比较: `{json.dumps(p.metrics, ensure_ascii=False)}`")
        elif p.metrics_resized_to_ref is not None:
            lines.append(f"缩放后比较: `{json.dumps(p.metrics_resized_to_ref, ensure_ascii=False)}`")
        else:
            lines.append("未比较: 尺寸不一致且当前环境缺少缩放依赖。")
        if th_eval["evaluated"]:
            lines.append(f"阈值判定: `{'PASS' if th_eval['pass'] else 'FAIL'}` (`mode={th_eval['mode']}`)")
        else:
            lines.append(f"阈值判定: `N/A` ({th_eval['reason']})")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MATLAB/Python figure regression for Rafaely ch1/ch6.")
    parser.add_argument("--chapters", default="ch1,ch2,ch3,ch4,ch5,ch6,ch7", help="Comma-separated chapters, e.g. ch1,ch6")
    parser.add_argument("--scripts", default="", help="Comma-separated MATLAB script stems to limit scope, e.g. fig_Pn,fig_rotation")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "rafaely_image_regression"),
        help="Output root for matlab/python renders and reports",
    )
    parser.add_argument("--skip-matlab", action="store_true", help="Skip MATLAB export")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python rendering")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing rendered images")
    parser.add_argument("--matlab-probe-timeout-s", type=int, default=60, help="Timeout for MATLAB CLI probe")
    parser.add_argument("--matlab-timeout-s", type=int, default=180, help="Timeout per MATLAB script export")
    parser.add_argument("--ssim-min", type=float, default=None, help="Minimum SSIM threshold for pass/fail evaluation")
    parser.add_argument("--rmse-max", type=float, default=None, help="Maximum RMSE threshold for pass/fail evaluation")
    parser.add_argument("--mae-max", type=float, default=None, help="Maximum MAE threshold for pass/fail evaluation")
    parser.add_argument(
        "--use-default-thresholds",
        action="store_true",
        help="Use FigureReproConfig defaults (currently enables default SSIM threshold)",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero if any threshold-evaluated pair fails",
    )
    parser.add_argument(
        "--require-pairs",
        action="store_true",
        help="Exit non-zero if no MATLAB/Python image pairs were compared",
    )
    parser.add_argument(
        "--require-matlab-batch-ready",
        action="store_true",
        help="Exit non-zero if MATLAB is requested but CLI probe is not batch-ready",
    )
    parser.add_argument(
        "--require-figure-count-match",
        action="store_true",
        help="Exit non-zero if any script has mismatched MATLAB/Python figure counts",
    )
    args = parser.parse_args(argv)

    chapters = [c.strip() for c in args.chapters.split(",") if c.strip()]
    for c in chapters:
        if c not in CHAPTERS:
            raise SystemExit(f"unsupported chapter: {c}")
    script_filter = {s.strip() for s in args.scripts.split(",") if s.strip()}
    out_root = Path(args.output_dir).expanduser().resolve()
    default_fig_cfg = FigureReproConfig()
    ssim_min = args.ssim_min
    if args.use_default_thresholds and ssim_min is None:
        ssim_min = float(default_fig_cfg.image_ssim_threshold)
    thresholds = _threshold_config(ssim_min=ssim_min, rmse_max=args.rmse_max, mae_max=args.mae_max)
    matlab_rt = detect_matlab()
    matlab_probe = None if matlab_rt is None else probe_matlab_cli(timeout_s=max(1, int(args.matlab_probe_timeout_s)))

    discovered = _discover_matlab_scripts(chapters)
    matlab_probe_blocked = False
    if not args.compare_only:
        if not args.skip_matlab and matlab_rt is None:
            print("MATLAB not found; skipping MATLAB export.", flush=True)
            matlab_probe_blocked = True
            args.skip_matlab = True
        elif not args.skip_matlab and matlab_probe is not None and matlab_probe.status != "ok":
            print(f"MATLAB CLI probe status: {matlab_probe.status}", flush=True)
            print(f"MATLAB CLI probe message: {matlab_probe.message}", flush=True)
            if matlab_probe.stdout_tail.strip():
                print("MATLAB CLI probe stdout tail:", flush=True)
                print(matlab_probe.stdout_tail.strip(), flush=True)
            if matlab_probe.stderr_tail.strip():
                print("MATLAB CLI probe stderr tail:", flush=True)
                print(matlab_probe.stderr_tail.strip(), flush=True)
            print("Skipping MATLAB export because MATLAB CLI is not batch-ready in this session.", flush=True)
            matlab_probe_blocked = True
            args.skip_matlab = True

        if not args.skip_matlab:
            print(f"Using MATLAB: {matlab_rt.executable} ({matlab_rt.source})", flush=True)
        for ch, scripts in discovered.items():
            for mfile in scripts:
                if script_filter and mfile.stem not in script_filter:
                    continue
                base_name = mfile.stem
                py_example = _python_example_path_for_matlab_script(ch, mfile)
                if not args.skip_matlab:
                    m_out = out_root / "matlab" / ch
                    print(f"[MATLAB] {ch}/{mfile.name}", flush=True)
                    try:
                        count, _ = _run_matlab_script_and_save(mfile, m_out, base_name, timeout_s=args.matlab_timeout_s)
                        print(f"  exported {count} figure(s)", flush=True)
                    except subprocess.TimeoutExpired:
                        print(f"  failed: MATLAB timed out after {args.matlab_timeout_s}s", flush=True)
                    except Exception as e:
                        print(f"  failed: {e}", flush=True)
                if not args.skip_python and py_example.exists():
                    p_out = out_root / "python" / ch
                    print(f"[Python] {ch}/{py_example.name}", flush=True)
                    try:
                        count = _run_python_example_and_save(py_example, p_out, base_name)
                        print(f"  rendered {count} figure(s)", flush=True)
                    except Exception as e:
                        print(f"  failed: {e}", flush=True)
                elif not args.skip_python:
                    print(f"[Python] missing example for {ch}/{mfile.stem}: {py_example}", flush=True)

    pairs: list[PairResult] = []
    for ch, scripts in discovered.items():
        for mfile in scripts:
            if script_filter and mfile.stem not in script_filter:
                continue
            m_dir = out_root / "matlab" / ch
            p_dir = out_root / "python" / ch
            if not m_dir.exists() or not p_dir.exists():
                continue
            pairs.extend(_compare_outputs(ch, mfile.stem, m_dir, p_dir))

    per_script_rows = _build_per_script_comparison_summary(discovered, script_filter, out_root, thresholds, pairs)

    metadata = {
        "output_dir": str(out_root),
        "chapters": chapters,
        "matlab_available": matlab_rt is not None,
        "matlab_runtime": None if matlab_rt is None else {"executable": matlab_rt.executable, "source": matlab_rt.source},
        "matlab_probe": None if matlab_probe is None else asdict(matlab_probe),
        "thresholds_active": thresholds,
        "matlab_probe_timeout_s": int(args.matlab_probe_timeout_s),
        "cwd": os.getcwd(),
    }
    _write_report(out_root / "report.json", pairs, metadata, thresholds=thresholds, per_script_rows=per_script_rows)
    _write_markdown_summary(out_root / "report.md", pairs, metadata, thresholds=thresholds, per_script_rows=per_script_rows)
    th_summary = _summarize_threshold_results(pairs, thresholds)
    count_summary = _summarize_script_count_mismatches(per_script_rows)
    print(f"Wrote report: {out_root / 'report.json'}", flush=True)
    print(f"Wrote summary: {out_root / 'report.md'}", flush=True)
    print(f"Compared pairs: {len(pairs)}", flush=True)
    print(
        "Threshold summary: "
        f"evaluated={th_summary['threshold_evaluated_pairs']} "
        f"pass={th_summary['threshold_pass_pairs']} "
        f"fail={th_summary['threshold_fail_pairs']}",
        flush=True,
    )
    print(
        "Figure-count summary: "
        f"match={count_summary['scripts_figure_count_match']} "
        f"mismatch={count_summary['scripts_figure_count_mismatch']} "
        f"total={count_summary['scripts_total']}",
        flush=True,
    )
    if args.require_matlab_batch_ready and matlab_probe_blocked:
        return 4
    if args.require_pairs and len(pairs) == 0:
        return 5
    if args.require_figure_count_match and count_summary["scripts_figure_count_mismatch"] > 0:
        return 6
    if args.fail_on_threshold and th_summary["threshold_fail_pairs"] > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
