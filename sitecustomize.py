"""Ensure the project root is importable when running scripts directly.

When launching `python model/train.py` from the project root, Python's
sys.path can start at `.../model`, which may prevent importing the top-level
`model` package as expected.

This file is auto-imported by Python at startup (via `site` module) if it's
present on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# If root isn't in sys.path, add it so `import model.*` works.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

