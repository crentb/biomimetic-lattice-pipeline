#!/usr/bin/env python
"""Closed-loop design optimization: SI table, SI history figure, and summary values (v3).

Purpose
-------
Summarizes the Optuna TPE study run by
biomimetic_pipeline/scripts/run_inverse_design_2026_09.py, in which every
proposed design is built and solved by the full pipeline (feature-to-CAD
translation, CadQuery CAD with the integrity gate, Gmsh, SfePy) and scored by
    J = |E_app - E*| / E*  +  max(0, t* - t) / t*
(E_app apparent modulus, t tension-dominated gauge volume fraction; targets
E* and t* recorded with each trial). Produces:
  * an SI table listing every trial (design variables, responses, J, bridge
    offset used by the CAD integrity retry ladder);
  * an SI figure of the optimization history:
      A  objective J per trial with the best-so-far envelope; random start-up
         trials shaded;
      B  apparent modulus and tension-dominated fraction per trial against
         their targets;
      C  design variables (twist scale f, rod diameter d, bridge layers N) per
         trial, handedness as marker fill;
  * a JSON of the numbers quoted in the main text (best design, verification
    at 0.5 mm when available, counts).

Inputs
------
biomimetic_pipeline/runs/<run-root>/trials.jsonl       one record per trial
biomimetic_pipeline/runs/<run-root>/best.json          optimizer summary (if finished)
biomimetic_pipeline/runs/<run-root>/best_verified.json 0.5-mm verification (optional)

Outputs
-------
data/si_table_optimization.tex      booktabs rows (\\input by supplemental.tex)
data/optimization_values.json       numbers for the text
figures/figureS4.{pdf,png,tif}      optimization history (SI)

Usage (conda base, from manuscript/matter_v2/):  python scripts/make_si_optimization_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402

MANU = HERE.parent
BIO = MANU.parents[1]
DATA = MANU / "data"
OUT = MANU / "figures" / "figureS4"


def load(run_root: str):
    """Trial records (all states), optimizer summary, and verification record (or None)."""
    root = BIO / "runs" / run_root
    trials = [json.loads(x) for x in (root / "trials.jsonl").read_text().splitlines() if x.strip()]
    best = json.loads((root / "best.json").read_text()) if (root / "best.json").exists() else None
    ver = (
        json.loads((root / "best_verified.json").read_text())
        if (root / "best_verified.json").exists()
        else None
    )
    return trials, best, ver


def table(trials: list[dict]) -> str:
    """Booktabs body: one row per trial; failed trials show the failure instead of responses."""
    head = (
        "\\begin{tabular}{r r r r l r r r r r}\n\\toprule\n"
        "trial & $f$ & $N$ & $d$ (mm) & handedness & offset (mm) & $E_{\\mathrm{app}}$ (MPa) & "
        "tension frac. & rot.\\ ($^\\circ$/1\\%) & $J$ \\\\\n\\midrule\n"
    )
    rows = []
    for t in trials:
        lead = f"{t['trial']} & {t['f']:.3f} & {t['N']} & {t['d']:.3f} & {t['handedness']} & "
        if t.get("state") == "ok":
            rows.append(
                lead + f"{t['jitter_mm']:.2f} & {t['E_app']:.0f} & {t['tension']:.3f} & "
                f"${t['rotation_deg_per_pct']:+.2f}$ & {t['J']:.3f} \\\\"
            )
        else:
            rows.append(
                lead
                + "-- & \\multicolumn{4}{l}{failed ("
                + (
                    "CAD integrity"
                    if "integrity" in t.get("error", "").lower()
                    else "meshing or solver"
                )
                + ")} \\\\"
            )
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run-root", default="inverse_design_v3")
    ap.add_argument(
        "--startup-trials", type=int, default=6, help="random start-up trials of the TPE sampler"
    )
    args = ap.parse_args()
    trials, best, ver = load(args.run_root)
    ok = [t for t in trials if t.get("state") == "ok"]
    e_star, t_star = trials[0]["targets"]["E_app"], trials[0]["targets"]["tension"]

    # --- 1. Numbers for the text ---------------------------------------------------------------------
    b = min(ok, key=lambda t: t["J"])
    vals = {
        "n_trials": len(trials),
        "n_ok": len(ok),
        "n_failed": len(trials) - len(ok),
        "n_retried_bridge_offset": sum(1 for t in ok if len(t.get("attempts", [])) > 1),
        "targets": {"E_app": e_star, "tension": t_star},
        "best": {
            k: b[k]
            for k in (
                "trial",
                "f",
                "N",
                "d",
                "handedness",
                "E_app",
                "tension",
                "rotation_deg_per_pct",
                "inclination_deg",
                "J",
            )
        },
        "best_rel_error_E": (b["E_app"] - e_star) / e_star,
        "best_found_at_trial": b["trial"],
        "J_first_random": [t["J"] for t in trials[: args.startup_trials]],
        "wall_hours": sum(t.get("wall_s", 0.0) for t in trials) / 3600.0,
        "E_app_range_explored": [min(t["E_app"] for t in ok), max(t["E_app"] for t in ok)],
        "tension_range_explored": [min(t["tension"] for t in ok), max(t["tension"] for t in ok)],
        "optimizer_summary": best,
        "verification": ver,
    }
    (DATA / "optimization_values.json").write_text(json.dumps(vals, indent=1))
    (DATA / "si_table_optimization.tex").write_text(table(trials))

    # --- 2. SI figure: optimization history -----------------------------------------------------------
    S.apply()
    FW, FH = 174.0, 70.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    idx = np.array([t["trial"] for t in trials])
    J = np.array([t["J"] for t in trials])
    okm = np.array([t.get("state") == "ok" for t in trials])
    best_so_far = np.minimum.accumulate(np.where(okm, J, np.inf))

    # A -- objective per trial and the best-so-far envelope (log scale: J spans ~0.01 to ~3)
    axA = ax_mm(15, 16, 40, 44)
    axA.axvspan(-0.5, args.startup_trials - 0.5, color=S.TONE, alpha=0.45, lw=0, zorder=0)
    axA.plot(idx[okm], J[okm], "o", ms=3.4, color=S.NAVY_D, mec="white", mew=0.5, label="trial")
    if (~okm).any():
        axA.plot(
            idx[~okm],
            np.full((~okm).sum(), max(J[okm].max(), 1.0) * 1.3),
            "x",
            color=S.WINE_D,
            ms=4,
            label="failed",
        )
    axA.step(idx, best_so_far, where="post", color=S.GOLD_D, lw=1.2, label="best so far")
    axA.set_yscale("log")
    # plain tick labels on the log axis ("0.1", "1") instead of mathtext powers of ten
    axA.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axA.yaxis.set_minor_formatter(mticker.NullFormatter())
    axA.set_xlabel("trial")
    axA.set_ylabel("objective J")
    handles, labels = axA.get_legend_handles_labels()
    handles.append(Patch(color=S.TONE, alpha=0.45, lw=0))
    labels.append("random start-up")
    axA.legend(
        handles,
        labels,
        fontsize=5.2,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0,
        frameon=False,
    )
    S.recessive_grid(axA)
    label_mm("A", 1, 4)

    # B -- both responses indexed to their own target on ONE axis (no second y-scale):
    #      E_app / E* should reach 1; t / t* should reach at least 1.
    axB = ax_mm(72, 16, 40, 44)
    e = np.array([t["E_app"] if t.get("state") == "ok" else np.nan for t in trials]) / e_star
    tf = np.array([t["tension"] if t.get("state") == "ok" else np.nan for t in trials]) / t_star
    axB.axhline(1.0, color=S.RULE, lw=0.6, ls=(0, (2, 2)), zorder=0)
    axB.plot(
        idx,
        e,
        "o",
        ms=3.4,
        color=S.NAVY_D,
        mec="white",
        mew=0.5,
        label=f"apparent modulus / {e_star:.0f} MPa",
    )
    axB.plot(
        idx,
        tf,
        "s",
        ms=3.0,
        color=S.GOLD_D,
        mec="white",
        mew=0.5,
        label=f"tension-dominated fraction / {t_star:.2f}",
    )
    axB.set_xlabel("trial")
    axB.set_ylabel("response / target")
    axB.set_ylim(0, None)
    axB.legend(
        fontsize=5.2,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        handlelength=1.2,
        borderaxespad=0,
        frameon=False,
    )
    S.recessive_grid(axB)
    label_mm("B", 60, 4)

    # C -- design variables per trial; hollow markers = alternating handedness
    axC = ax_mm(129, 16, 41, 44)
    alt = np.array([t["handedness"] == "alternating" for t in trials])
    fv = np.array([t["f"] for t in trials])
    dv = np.array([t["d"] for t in trials])
    nv = np.array([t["N"] for t in trials])
    for vals_, c, m, lab in (
        (fv, S.NAVY_D, "o", r"twist scale $f$"),
        (dv / 2.128, S.GOLD_D, "s", r"$d/d_{\mathrm{meas}}$ (2.128 mm)"),
        (nv / 6.0, S.TEAL_D, "^", r"$N/6$"),
    ):
        axC.plot(idx[~alt], vals_[~alt], m, ms=3.2, color=c, mec="white", mew=0.4, label=lab)
        axC.plot(idx[alt], vals_[alt], m, ms=3.2, mfc="white", mec=c, mew=0.8)
    axC.set_xlabel("trial")
    axC.set_ylabel("design variable (scaled)")
    handles, labels = axC.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=S.MUTED, mew=0.8))
    labels.append("hollow = alternating")
    axC.legend(
        handles,
        labels,
        fontsize=5.2,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0,
        frameon=False,
    )
    S.recessive_grid(axC)
    label_mm("C", 117, 4)

    S.save(fig, OUT)
    print(
        json.dumps(
            {k: v for k, v in vals.items() if k not in ("optimizer_summary", "verification")},
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
