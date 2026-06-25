#!/usr/bin/env python
"""Ingest every upstream morphometric source into a canonical morphometrics.json.

Usage:
    python -m scripts.ingest_features --specimen-id enamel_001 \
        --out runs/ingest/morphometrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/ingest_features.py` from the repo root.
THIS = Path(__file__).resolve()
BIOMIMETIC_ROOT = THIS.parent.parent
if str(BIOMIMETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMIMETIC_ROOT))

from biomimetic_pipeline.ingest import merge  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen-id", required=True, help="Specimen identifier for provenance")
    ap.add_argument("--out", required=True, help="Output path for canonical morphometrics.json")
    ap.add_argument("--no-validate", action="store_true", help="Skip schema validation (faster)")
    args = ap.parse_args()

    out_path = merge.ingest_and_save(
        specimen_id=args.specimen_id,
        out_path=Path(args.out),
        validate_schema=not args.no_validate,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
