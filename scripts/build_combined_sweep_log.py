#!/usr/bin/env python
"""
build_combined_sweep_log.py
===========================

Purpose
-------
Merge the two per-arm layer-sweep logs (biomimetic + thick) into ONE
"all 12" master CSV, so the full N_BRIDGE_LAYERS sweep (both arms,
N = 4..9) can be read from a single file. Adds two provenance columns --
`arm` (bio | thick) and `rod_exception` (bool) -- plus a human-readable
`note` on any row whose rod diameter departs from its arm's canonical rod.

Why this exists
---------------
The sweep is run as two parallel arms, each harvested by `build_sweep_log.py`
into its OWN 6-row `sweep_log.csv` (`runs/sweep_layers_v2/` for bio,
`runs/sweep_layers_v2_thick/` for thick). Nothing on disk holds all 12
trials together, and the thick arm carries a SINGLE-ROW caveat that is
invisible inside a per-arm file:

    thick N=9 is built at ROD_DIAMETER = 2.75 mm (rod-rod gap 0.44 mm)
    instead of the thick arm's canonical 3.167 mm (gap 0.025 mm), because
    3.167 mm is geometrically UNMESHABLE at N=9 (OCCT rod-union degeneracy
    + sub-resolution sliver faces at the bridge<->rod junctions). See
    runs/sweep_layers_v2_thick/WHY_N9_THICK_NEEDS_OWN_ROD.txt.

Because that row's rod is thinner, its E_effective (2256.9 MPa) is LOWER
than N=8 thick (2464.4 MPa) -- NOT because N=9 is intrinsically less stiff
but because the load-bearing cross-section is smaller. The thick E_eff(N)
curve therefore has a DISCONTINUITY at N=9. This master log flags that row
explicitly so no downstream plot or table silently treats the thick arm as
"monotonic in N".

This script is a PURE TRANSFORM of the two per-arm logs -- it does NOT
re-read per-trial JSON. If any trial changed on disk, regenerate the
per-arm logs FIRST, then re-merge:
    conda run -n base python scripts/build_sweep_log.py --sweep-root runs/sweep_layers_v2
    conda run -n base python scripts/build_sweep_log.py --sweep-root runs/sweep_layers_v2_thick
    conda run -n base python scripts/build_combined_sweep_log.py

Inputs (CLI)
------------
  --bio-log    : per-arm bio CSV    (default runs/sweep_layers_v2/sweep_log.csv)
  --thick-log  : per-arm thick CSV  (default runs/sweep_layers_v2_thick/sweep_log.csv)
  --out        : combined CSV       (default runs/sweep_log_all12.csv)

Outputs
-------
  CSV with leading columns [arm, rod_exception], then EVERY column from the
  per-arm logs (union; bio column order first, any thick-only extras
  appended), then a trailing [note]. Rows grouped bio-first then thick, each
  ascending in N_BRIDGE_LAYERS. Prints a one-line-per-row summary to stdout.

Side effects: writes the combined CSV (creates parent dir if missing);
reads only the two input CSVs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent  # repo root = scripts/..

# Tolerance (mm) for deciding two rod diameters are the "same" spec. Rod
# values round-trip through CSV text at full float precision, so a row's rod
# matches the (full-precision) canonical value EXACTLY; this tolerance only
# guards against trailing-digit float noise. 1e-6 mm (1 nm) is ~400,000x
# smaller than the smallest intentional rod step in this study (the 0.42 mm
# jump from 3.167 -> 2.75 for thick N=9) and far below the 0.5 mm mesh size,
# so it can never blur a real design difference yet absorbs any round-off.
ROD_EQ_TOL_MM = 1e-6


# --- helpers -------------------------------------------------------------
def _read_arm_log(path: Path):
    """Read one per-arm sweep_log.csv -> (rows: list[dict], cols: list[str]).

    Preserves column ORDER (csv.DictReader.fieldnames is in file order) so the
    merged schema matches the source layout. Raises if the file is missing or
    has no data rows -- a silently-empty arm would produce a misleading
    "complete" master, so we fail loudly instead.
    """
    if not path.is_file():
        raise SystemExit(f"per-arm log not found: {path}")
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    if not rows:
        raise SystemExit(f"per-arm log has no data rows: {path}")
    return rows, cols


def _canonical_rod(rows):
    """Return the arm's canonical rod diameter (mm) = the MOST COMMON value.

    Rationale: within an arm the rod is meant to be CONSTANT across N; the one
    trial forced to deviate for meshability (thick N=9 @ 2.75) is the minority
    value. Taking the mode makes "exception" detection automatic and
    self-correcting -- we never hard-code "N == 9", so if the deviating N ever
    changes the flag follows it. We group by a rounded key (to count equal
    floats robustly) but RETURN a full-precision member value, so downstream
    comparisons against raw row rods are exact (diff == 0 for canonical rows).
    Returns None for an empty arm (defensive; never expected).
    """
    vals = []
    for r in rows:
        try:
            vals.append(float(r["ROD_DIAMETER"]))  # mm
        except (KeyError, TypeError, ValueError):
            continue  # row missing/garbled rod -> ignore for the mode
    if not vals:
        return None
    keys = [round(v, 6) for v in vals]  # 6 dp grouping key
    mode_key = Counter(keys).most_common(1)[0][0]  # most frequent rod
    for v in vals:  # first full-precision
        if round(v, 6) == mode_key:  # member of the mode
            return v
    return mode_key  # unreachable, but keeps the contract total


def _n_layers(row):
    """Parse N_BRIDGE_LAYERS as int for sorting; 0 if absent/garbled."""
    try:
        return int(float(row.get("N_BRIDGE_LAYERS", 0)))
    except (TypeError, ValueError):
        return 0


def _annotate(rows, arm, canon):
    """Prepend arm + rod_exception + append note to each row of one arm.

    A row is an "exception" when its rod departs from the arm's canonical rod
    by more than ROD_EQ_TOL_MM. Only thick N=9 (2.75 vs 3.167) trips this; all
    bio rows and thick N<=8 stay False. The note (empty for non-exceptions)
    carries the *why* into the data so any reader of the merged log sees that
    the row's stiffness is not comparable to its arm-mates.
    """
    out = []
    for r in rows:
        try:
            rod = float(r["ROD_DIAMETER"])  # mm
        except (KeyError, TypeError, ValueError):
            rod = None
        is_exc = rod is not None and canon is not None and abs(rod - canon) > ROD_EQ_TOL_MM
        note = ""
        if is_exc:
            note = (
                f"rod {rod:.3f} mm departs from {arm} canonical "
                f"{canon:.3f} mm (meshability fix); E_effective NOT "
                f"comparable within arm (thinner rod -> less cross-section). "
                f"See runs/sweep_layers_v2_thick/WHY_N9_THICK_NEEDS_OWN_ROD.txt"
            )
        # arm/rod_exception lead; **r preserves all source fields; note trails.
        out.append({"arm": arm, "rod_exception": is_exc, **r, "note": note})
    return out


def main() -> int:
    # --- 1. Parse CLI ----------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Merge per-arm layer-sweep logs into one all-12 master CSV."
    )
    ap.add_argument(
        "--bio-log", type=Path, default=ROOT / "runs" / "sweep_layers_v2" / "sweep_log.csv"
    )
    ap.add_argument(
        "--thick-log", type=Path, default=ROOT / "runs" / "sweep_layers_v2_thick" / "sweep_log.csv"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sweep_log_all12.csv")
    args = ap.parse_args()

    # --- 2. Load both per-arm logs --------------------------------------
    bio_rows, bio_cols = _read_arm_log(args.bio_log.resolve())
    thick_rows, thick_cols = _read_arm_log(args.thick_log.resolve())

    # Union of source columns: bio order first, then any thick-only extras
    # appended. Today both arms share an identical schema (same harvester), so
    # no extras are expected -- but we union defensively rather than silently
    # DROP a column if the two logs ever diverge.
    src_cols = list(bio_cols)
    for c in thick_cols:
        if c not in src_cols:
            src_cols.append(c)

    # --- 3. Annotate rows with arm + rod-exception flag -----------------
    bio_canon = _canonical_rod(bio_rows)  # mm; ~2.128 (bio rod is constant)
    thick_canon = _canonical_rod(thick_rows)  # mm; ~3.167 (the 5-of-6 majority)
    rows = _annotate(bio_rows, "bio", bio_canon) + _annotate(thick_rows, "thick", thick_canon)

    # --- 4. Order rows: bio block then thick block, each ascending in N --
    # Stable sort by (arm_rank, N): arm_rank keeps the bio block before thick;
    # within a block, ascending N_BRIDGE_LAYERS gives a readable E_eff(N) curve.
    arm_rank = {"bio": 0, "thick": 1}
    rows.sort(key=lambda r: (arm_rank.get(r["arm"], 9), _n_layers(r)))

    # --- 5. Write the combined CSV --------------------------------------
    # Leading provenance cols, then the full source schema, then the note col.
    out_cols = ["arm", "rod_exception"] + src_cols + ["note"]
    args.out.parent.mkdir(parents=True, exist_ok=True)  # runs/ exists; be safe
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- 6. Report -------------------------------------------------------
    n_exc = sum(1 for r in rows if r["rod_exception"])
    print(
        f"Wrote {args.out}  ({len(rows)} rows = {len(bio_rows)} bio + "
        f"{len(thick_rows)} thick, {len(out_cols)} cols, "
        f"{n_exc} rod-exception row(s))"
    )
    print(f"  bio   canonical rod = {bio_canon} mm")
    print(f"  thick canonical rod = {thick_canon} mm")
    for r in rows:
        flag = "  <== ROD EXCEPTION" if r["rod_exception"] else ""
        print(
            f"  {r['arm']:5} N={_n_layers(r)}  rod={r.get('ROD_DIAMETER')}  "
            f"E_eff={r.get('E_effective_MPa')}{flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
