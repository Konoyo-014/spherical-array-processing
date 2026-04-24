from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_rafaely_ch2_planewave_rigid_sphere import main


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
