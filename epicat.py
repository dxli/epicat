#!/usr/bin/env python3
"""Convenience launcher so the tool can be run without installing it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicat.cli import main

if __name__ == "__main__":
    sys.exit(main())
