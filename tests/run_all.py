#!/usr/bin/env python
"""Run every unit test module. Exit 1 on any failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    test_files = sorted(p for p in ROOT.glob("test_*.py") if p.is_file())
    if not test_files:
        print("No test files found.")
        return 1
    failures = []
    for tf in test_files:
        print(f"\n--- {tf.name} ---")
        rc = subprocess.call([sys.executable, str(tf)], cwd=str(ROOT.parent))
        if rc != 0:
            failures.append(tf.name)
    print("\n====================")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(test_files)} test modules passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
