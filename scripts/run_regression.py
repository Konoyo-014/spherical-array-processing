#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.regression import detect_matlab, detect_octave


def _run(cmd: list[str]) -> int:
    cp = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(cp.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回归环境探测与进度文档更新入口")
    parser.add_argument("--update-progress", action="store_true", help="刷新 docs/source_inventory.json 与 docs/PROGRESS_DASHBOARD.md")
    args = parser.parse_args(argv)

    rt = detect_matlab()
    if rt is None:
        print("MATLAB: NOT FOUND (set MATLAB_BIN or MATLAB_ROOT to enable baseline regression)")
    else:
        print(f"MATLAB: FOUND via {rt.source} -> {rt.executable}")
    octave = detect_octave()
    if octave is None:
        print("Octave: NOT FOUND (optional)")
    else:
        print(f"Octave: FOUND via {octave.source} -> {octave.executable}")

    if args.update_progress:
        print("Refreshing source inventory...")
        if _run([sys.executable, "scripts/build_source_inventory.py"]) != 0:
            return 2
        print("Refreshing progress dashboard...")
        if _run([sys.executable, "scripts/update_progress_dashboard.py"]) != 0:
            return 3
        print(f"Updated: {ROOT / 'docs' / 'PROGRESS_DASHBOARD.md'}")
    else:
        print("Regression scaffold is ready. Add scenario scripts under scripts/ and tests/regression/.")
        print("Use --update-progress to refresh docs/PROGRESS_DASHBOARD.md.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
