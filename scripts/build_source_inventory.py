#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def matlab_function_signature(path: Path) -> str | None:
    text = path.read_text(errors="ignore")
    m = re.search(r"^function\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def main() -> int:
    src = ROOT / "src"
    politis = src / "Spherical-Array-Processing-MATLAB"
    rafaely = src / "Rafaely" / "matlab"
    payload: dict[str, object] = {"politis": {}, "rafaely": {}}

    if politis.exists():
        files = sorted(politis.glob("*.m"))
        payload["politis"] = {
            "count_m_files": len(files),
            "files": [
                {
                    "name": p.name,
                    "signature": matlab_function_signature(p),
                    "status": "planned" if p.name != "TEST_SCRIPTS.m" else "reference_script",
                }
                for p in files
            ],
        }
    if rafaely.exists():
        math_files = sorted((rafaely / "math").glob("*.m"))
        plot_files = sorted((rafaely / "plot").glob("*.m"))
        fig_files = sorted((rafaely / "fig").rglob("*.m"))
        payload["rafaely"] = {
            "math_count": len(math_files),
            "plot_count": len(plot_files),
            "fig_count": len(fig_files),
            "math_files": [p.name for p in math_files],
            "plot_files": [p.name for p in plot_files],
            "fig_files": [str(p.relative_to(rafaely)).replace("\\", "/") for p in fig_files],
        }

    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "source_inventory.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
