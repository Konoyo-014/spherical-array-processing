#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.regression.image_compare import compare_grayscale_images, load_image_gray


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: compare_images.py <reference> <candidate>")
        return 2
    ref = load_image_gray(argv[1])
    cand = load_image_gray(argv[2])
    metrics = compare_grayscale_images(ref, cand)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

