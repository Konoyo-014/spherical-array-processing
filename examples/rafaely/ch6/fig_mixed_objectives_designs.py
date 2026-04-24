from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_rafaely_ch6_mixed_objectives_designs import main


if __name__ == "__main__":
    raise SystemExit(0 if main(print_table=True) else 1)
