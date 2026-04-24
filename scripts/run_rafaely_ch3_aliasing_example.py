#!/usr/bin/env python3
from __future__ import annotations

from scripts.run_rafaely_missing_fig import run_named


def main(show: bool = True):
    return run_named("ch3_aliasing_example", show=show)


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
