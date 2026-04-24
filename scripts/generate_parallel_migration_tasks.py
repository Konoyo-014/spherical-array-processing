#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "parallel_tasks"


def classify_politis(name: str) -> str:
    n = name.lower()
    if n.startswith("arrayshtfilters") or "evaluateshtfilters" in n:
        return "encoding_and_equalization"
    if n.startswith("beamweights") or n.startswith("computevel") or n.startswith("extractaxis"):
        return "fixed_beamforming_weights"
    if n.startswith("sphmvdr") or n.startswith("sphlcmv") or n.startswith("sphnullformer") or "pmmw" in n:
        return "adaptive_beamforming"
    if n.startswith("sphmusic") or n.startswith("sphpwd") or n.startswith("sphsr") or n.startswith("sphesprit") or "intensityhist" in n:
        return "doa_estimation"
    if n.startswith("getdiffuseness") or n.startswith("getdiffcoh") or n.startswith("diffcoherence"):
        return "diffuseness_and_coherence"
    if n.startswith("plot"):
        return "plotting_helpers"
    if n.startswith("sparse_") or n.startswith("sorted_"):
        return "numerical_utilities"
    if n.startswith("spharray"):
        return "array_analysis"
    return "misc"


def classify_rafaely(path: Path) -> str:
    parts = path.parts
    if "math" in parts:
        return "rafaely_math"
    if "plot" in parts:
        return "rafaely_plot"
    if "fig" in parts:
        ch = next((p for p in parts if re.fullmatch(r"ch\d+", p)), "fig_misc")
        return f"rafaely_{ch}"
    return "rafaely_misc"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, list[dict[str, str]]] = {}

    politis_root = SRC / "Spherical-Array-Processing-MATLAB"
    for p in sorted(politis_root.glob("*.m")):
        bucket = classify_politis(p.stem)
        tasks.setdefault(bucket, []).append(
            {
                "source": str(p.relative_to(ROOT)),
                "target_module_hint": f"spherical_array_processing/repro/politis/{bucket}.py",
                "kind": "politis_function" if p.name != "TEST_SCRIPTS.m" else "politis_reference_script",
            }
        )

    rafaely_root = SRC / "Rafaely" / "matlab"
    for p in sorted(rafaely_root.rglob("*.m")):
        bucket = classify_rafaely(p)
        tasks.setdefault(bucket, []).append(
            {
                "source": str(p.relative_to(ROOT)),
                "target_module_hint": (
                    "examples/" + "/".join(p.relative_to(rafaely_root).with_suffix(".py").parts)
                    if "fig" in p.parts
                    else "spherical_array_processing/repro/rafaely/math.py"
                ),
                "kind": "rafaely_figure_script" if "fig" in p.parts else "rafaely_support_function",
            }
        )

    summary = {k: len(v) for k, v in sorted(tasks.items())}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")

    md = ["# Parallel Migration Task Packs", ""]
    for bucket in sorted(tasks):
        md.append(f"## {bucket}")
        md.append("")
        for item in tasks[bucket]:
            md.append(f"- `{item['source']}` -> `{item['target_module_hint']}` ({item['kind']})")
        md.append("")
    (OUT / "tasks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'tasks.json'}")
    print(f"Wrote {OUT / 'tasks.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

