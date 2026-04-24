#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.regression.matlab import detect_matlab, probe_matlab_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe MATLAB CLI batch readiness (login/timeout diagnostics).")
    parser.add_argument("--timeout-s", type=int, default=15, help="Probe timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Print probe result as JSON")
    args = parser.parse_args(argv)

    rt = detect_matlab()
    probe = probe_matlab_cli(timeout_s=args.timeout_s)
    payload = {
        "matlab_runtime": None if rt is None else {"executable": rt.executable, "source": rt.source},
        "probe": {
            "status": probe.status,
            "message": probe.message,
            "stdout_tail": probe.stdout_tail,
            "stderr_tail": probe.stderr_tail,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    else:
        if rt is None:
            print("MATLAB runtime: not found", flush=True)
        else:
            print(f"MATLAB runtime: {rt.executable} ({rt.source})", flush=True)
        print(f"Probe status: {probe.status}", flush=True)
        print(f"Probe message: {probe.message}", flush=True)
        if probe.stdout_tail.strip():
            print("STDOUT tail:", flush=True)
            print(probe.stdout_tail.strip(), flush=True)
        if probe.stderr_tail.strip():
            print("STDERR tail:", flush=True)
            print(probe.stderr_tail.strip(), flush=True)

    # Non-zero exit for states that block batch regression.
    return 0 if probe.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
