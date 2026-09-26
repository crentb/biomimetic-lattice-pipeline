#!/usr/bin/env python
"""SI tables: every design-space cell and every mesh level, straight from data.

Purpose
-------
Writes LaTeX tabular bodies that supplemental.tex \\input{}s, so the SI lists
the exact values behind every point in Figs. 5 and S1 (referees can check
any plotted value without re-running the pipeline). Since the 2026-09-25
revision the stress ratio kappa_99 and the solid fraction phi_g are
computed over the gauge (plates excluded; compute_gauge_metrics.py); the
superseded whole-model ratio is listed alongside for traceability.

Inputs
------
data/fig5_values.json          (make_fig5_v2.py: arms + twist sweep)
data/mesh_convergence.json     (make_figS1_mesh_v2.py)

Outputs
-------
data/si_table_arms.tex, data/si_table_twist.tex, data/si_table_mesh.tex
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def fmt(x, nd=2):
    """Fixed-point number for LaTeX; negatives in math mode so they get a true minus sign."""
    if not (isinstance(x, (int, float)) and x == x):
        return "--"
    return f"${x:.{nd}f}$" if x < 0 else f"{x:.{nd}f}"


def main() -> None:
    v = json.loads((DATA / "fig5_values.json").read_text())
    # --- 1. Rod-diameter x bridge-layer study (H = 24 mm, f = 1) ----------------------
    # Family labels follow the v3 manuscript terminology: "measured diameter" (rods at
    # the measured 2.128-mm CAD diameter; stored arm key "bio") and "densified"
    # (3.167-mm rods, 0.025-mm surface clearance; stored arm key "thick").
    lines = []
    for r in sorted(v["arms"], key=lambda r: (r["arm"], r["N_BRIDGE_LAYERS"])):
        arm = "measured diameter" if r["arm"] == "bio" else "densified"
        lines.append(
            f"{arm} & {int(r['N_BRIDGE_LAYERS'])} & {fmt(r['phi_gauge'], 3)} & "
            f"{fmt(r['E_app'], 0)} & {fmt(r['E_app'] / r['phi_gauge'], 0)} & {fmt(r['SCF_p99'], 2)} & "
            f"{fmt(r['SCF_p99_whole'], 2)} & {fmt(r['SCF'], 1)} & {fmt(r['tens'], 3)} & "
            f"{fmt(r['rot'], 2)} \\\\"
        )
    head = (
        "\\begin{tabular}{l r r r r r r r r r}\n\\toprule\n"
        "family & $N$ & $\\phi_g$ & $E_{\\mathrm{app}}$ & $E_{\\mathrm{app}}/\\phi_g$ & $\\kappa_{99}$ & "
        "$\\kappa_{99}^{\\mathrm{all}}$ & peak & tension & rot. \\\\\n\\midrule\n"
    )
    (DATA / "si_table_arms.tex").write_text(
        head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    # --- 2. Twist sweep (N = 2, H = 20 mm) --------------------------------------------
    lines = []
    for r in sorted(v["decussation"], key=lambda r: r["f"]):
        mark = "$^{a}$" if "_j0" in str(r.get("run", "")) or abs(r["f"] - 0.15) < 1e-9 else ""
        lines.append(
            f"{fmt(r['f'], 2)}{mark} & {fmt(r['phi_gauge'], 3)} & {fmt(r['E_app'], 0)} & "
            f"{fmt(r['SCF'], 2)} & {fmt(r['SCF_whole'], 2)} & "
            f"{fmt(r['tens'], 3)} & {fmt(r['incl'], 1)} & {fmt(r['rot'], 2)} \\\\"
        )
    head = (
        "\\begin{tabular}{r r r r r r r r}\n\\toprule\n"
        "$f$ & $\\phi_g$ & $E_{\\mathrm{app}}$ (MPa) & $\\kappa_{99}$ & $\\kappa_{99}^{\\mathrm{all}}$ & "
        "tension frac. & incl.\\ ($^\\circ$) & rot.\\ ($^\\circ$/1\\%) \\\\\n\\midrule\n"
    )
    (DATA / "si_table_twist.tex").write_text(
        head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    # --- 3. Mesh convergence -----------------------------------------------------------
    m = json.loads((DATA / "mesh_convergence.json").read_text())
    lines = []
    for r in sorted(m, key=lambda r: -r["h_mm"]):
        lines.append(
            f"{r['h_mm']:g} & {r['n_tet']:,} & {fmt(r['E_app'], 1)} & {fmt(r['SCF'], 3)} & "
            f"{fmt(r['SCF_whole'], 3)} & {fmt(r['SCF_peak'], 2)} & {fmt(r['tens'], 4)} & "
            f"{fmt(r['incl'], 2)} & {fmt(r['rot'], 3)} \\\\"
        )
    head = (
        "\\begin{tabular}{r r r r r r r r r}\n\\toprule\n"
        "$h$ (mm) & tetrahedra & $E_{\\mathrm{app}}$ (MPa) & $\\kappa_{99}$ & $\\kappa_{99}^{\\mathrm{all}}$ & peak & "
        "tension frac. & incl.\\ ($^\\circ$) & rot.\\ ($^\\circ$/1\\%) \\\\\n\\midrule\n"
    )
    (DATA / "si_table_mesh.tex").write_text(
        head + "\n".join(lines).replace(",", "{,}") + "\n\\bottomrule\n\\end{tabular}\n"
    )
    print("SI tables written")


if __name__ == "__main__":
    main()
