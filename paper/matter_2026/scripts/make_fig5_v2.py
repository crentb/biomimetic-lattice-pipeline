#!/usr/bin/env python
"""Figure 5 file (main-text Figure 6 in v3): design space -- rod diameter, bridge layers, and twist.

Purpose
-------
Row 1 -- architecture levers at the as-translated twist (f = 1.0), two families that
differ ONLY in rod diameter (measured diameter 2.13 mm vs densified 3.17 mm; same
3.19-mm pitch, 1.70-mm bridges), height-matched sweep N = 4..9 (H = 24 mm):
  A  apparent modulus E_app = F / (A_footprint * eps) vs bridge layers N.
  B  percentile stress ratio kappa_99 = P99(VM) / mean(VM), volume-weighted
     over the GAUGE WINDOW (lattice between the plates minus 1 mm at each
     plate; runs/_load_path_analysis/gauge_metrics.csv). Revision 2026-09-25:
     the earlier whole-model ratio (robust_scf.csv) let the lightly stressed
     solid plates dilute the mean. The stock metric (peak element / mean) is
     set by singular junction elements, is not mesh-converged, and is not
     used (Fig. S1).
  C  tension-dominated volume fraction vs N: bridges suppress the
     decussation-driven tension partition in both arms.
Row 2 -- the twist (decussation) lever at N = 2 (H = 20 mm), f = scale on the
translated ring-rotation profile. v3 (2026-09-26): with the sign-corrected PIV
trajectories the tilt-matched scale is f* = 1.03, i.e. the as-translated design, so
the separate f* band of v2 was removed; only the f = 1 reference line is drawn:
  D  E_app vs f.   E  tension-dominated volume fraction vs f.
  F  compression-twist coupling (platen rotation per 1 % strain) vs f.
Row 3 (v3, 2026-09-26) -- closed-loop Bayesian optimization (Optuna TPE; each design
built and solved by the full pipeline; run_inverse_design_2026_09.py):
  G  objective J per trial and best-so-far envelope (random start-up shaded).
  H  every evaluated design in the (E_app, tension-dominated fraction) plane with
     the target set (E_app = E*, fraction >= t*), the twist study at the measured
     diameter for context, and the selected design (search and 0.5-mm verification).
  I  render of the selected design (render_optimum_v3.py) with its variables.
Top strip: renders of the four corner designs (measured/densified x N = 4/9).

Why this exists
---------------
v1 Fig. 5 plotted a solid-area-normalised "E_eff" against solid fraction (a
double normalisation), used a dual-axis decussation panel, and its panel E
misapplied Jia & Wang Eq. 8 (a strength balance) with a hard-coded lattice
point outside its own shaded region. The v1 "4.2x tortuosity jump" came from
a legacy metric that is not a principal-stress trajectory and collapses to
~1.3-1.9 for every N >= 4 cell; it is not used.

Inputs
------
runs/sweep_log_all12_H24.csv                         two-arm sweep (H = 24)
runs/sweep_decussation_v2/sweep_log.csv              f = 0.05, 0.5, 1.0, 1.5
runs/sweep_decussation_dense_v3/sweep_log.csv        f = 0.6..1.25 (+0.15..0.55)
runs/_load_path_analysis/*.json                      partition, coupling
runs/_load_path_analysis/gauge_metrics.csv           kappa_99 (gauge), phi_gauge
runs/_load_path_analysis/robust_scf.csv              whole-model ratio (SI only)
figures/panels/render_{bio,thick}_N{4,9}.png
data/fig3_values.json                                f*
runs/inverse_design_v3/trials.jsonl, best_verified.json   optimization (row 3)
figures/panels/render_optimum.png                    selected design (row 3)

Outputs
-------
figures/figure5.{pdf,png,tif}; data/fig5_values.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
MANU = HERE.parent
BIO = MANU.parents[1]
sys.path.insert(0, str(HERE))
import _style_v2 as S  # noqa: E402

RUNS = BIO / "runs"
LPA = RUNS / "_load_path_analysis"
OUT = MANU / "figures" / "figure5"
VALUES = MANU / "data" / "fig5_values.json"


def lp(run_name: str) -> dict | None:
    f = LPA / (run_name.replace("/", "__") + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def robust_scf() -> dict:
    """run -> WHOLE-MODEL volume-weighted P99/mean ratio (compute_robust_scf.py; SI only)."""
    t = pd.read_csv(LPA / "robust_scf.csv")
    return dict(zip(t.run, t.SCF_p99))


def gauge() -> pd.DataFrame:
    """run -> gauge-window kappa_99 and gauge solid fraction (compute_gauge_metrics.py)."""
    return pd.read_csv(LPA / "gauge_metrics.csv").set_index("run")


OPT_ROOT = RUNS / "inverse_design_v3"
OPT_STARTUP = 6  # random start-up trials of the TPE sampler (driver default)


def optimization_record() -> dict | None:
    """Trials of the closed-loop study and the 0.5-mm verification of the best design (if run)."""
    log = OPT_ROOT / "trials.jsonl"
    if not log.exists():
        return None
    trials = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    ver = OPT_ROOT / "best_verified.json"
    return {"trials": trials, "verified": json.loads(ver.read_text()) if ver.exists() else None}


def decussation_table() -> pd.DataFrame:
    """All N = 2 decussation trials (v2 sweep + dense v3), one row per factor."""
    rows = []
    for log in (
        RUNS / "sweep_decussation_v2/sweep_log.csv",
        RUNS / "sweep_decussation_dense_v3/sweep_log.csv",
    ):
        if not log.exists():
            continue
        d = pd.read_csv(log)
        for _, r in d.iterrows():
            # label "decussation:factor=<f>[;jitter=<mm>]" -> f (jitter re-runs keep their f)
            f = float(re.search(r"factor=([0-9.]+)", str(r["label"])).group(1))
            j = lp(str(r["run_name"]))
            g = gauge()
            rn = str(r["run_name"])
            rows.append(
                {
                    "f": f,
                    "run": r["run_name"],
                    "E_eff": r["E_effective_MPa"],
                    "phi": r["solid_fraction"],
                    "phi_gauge": g.phi_gauge.get(rn, np.nan),
                    "SCF_whole": robust_scf().get(rn, np.nan),
                    "SCF": g.SCF_p99_gauge.get(rn, np.nan),
                    "tens": j["load_partition"]["tensile_volume_frac"] if j else np.nan,
                    "incl": j["load_partition"]["inclination_field_deg"] if j else np.nan,
                    "rot": j["top_plate"]["rotation_deg_per_pct_strain"] if j else np.nan,
                    "src": log.parent.name,
                }
            )
    t = pd.DataFrame(rows)
    # f = 1.0 exists in both sweeps (May run + Sept reproducibility re-run): keep the
    # original and record the re-run difference as the reproducibility check.
    t["pref"] = (t.src != "sweep_decussation_v2").astype(int)  # prefer the original May run
    t = (
        t.sort_values(["f", "pref"])
        .drop_duplicates("f", keep="first")
        .drop(columns="pref")
        .sort_values("f")
    )
    t["E_app"] = t.E_eff * t.phi
    return t


def main() -> None:
    S.apply()
    vals: dict = {}
    arms = pd.read_csv(RUNS / "sweep_log_all12_H24.csv")
    arms["E_app"] = arms.E_effective_MPa * arms.solid_fraction
    for col in ("tens", "rot"):
        arms[col] = np.nan
    rs, g = robust_scf(), gauge()
    arms["SCF_p99_whole"] = [rs.get(str(r), np.nan) for r in arms.run_name]
    arms["SCF_p99"] = [
        g.SCF_p99_gauge.get(str(r), np.nan) for r in arms.run_name
    ]  # kappa_99, gauge
    arms["phi_gauge"] = [g.phi_gauge.get(str(r), np.nan) for r in arms.run_name]
    for i, r in arms.iterrows():
        j = lp(str(r["run_name"]))
        if j:
            arms.at[i, "tens"] = j["load_partition"]["tensile_volume_frac"]
            arms.at[i, "rot"] = j["top_plate"]["rotation_deg_per_pct_strain"]
    bio = arms[arms.arm == "bio"].sort_values("N_BRIDGE_LAYERS")
    thk = arms[arms.arm == "thick"].sort_values("N_BRIDGE_LAYERS")
    dec = decussation_table()
    fstar = json.loads((MANU / "data/fig3_values.json").read_text())["f_star"]
    vals["arms"] = (
        arms[
            [
                "arm",
                "N_BRIDGE_LAYERS",
                "E_effective_MPa",
                "solid_fraction",
                "phi_gauge",
                "E_app",
                "SCF",
                "SCF_p99_whole",
                "SCF_p99",
                "tens",
                "rot",
            ]
        ]
        .round(4)
        .to_dict("records")
    )
    vals["decussation"] = dec.drop(columns=["run"]).round(4).to_dict("records")
    vals["f_star"] = fstar

    FW, FH = 174.0, 212.0
    fig = plt.figure(figsize=(FW * S.MM, FH * S.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)

    def label_mm(letter, x, y):
        fig.text(x / FW, 1 - y / FH, letter, fontsize=S.LABEL_PT, fontweight="bold", va="top")

    # --- top strip: corner-cell renders -------------------------------------------------
    for k, (name, lab) in enumerate(
        [
            ("bio_N4", "measured diameter, N = 4"),
            ("bio_N9", "measured diameter, N = 9"),
            ("thick_N4", "densified, N = 4"),
            ("thick_N9", "densified, N = 9"),
        ]
    ):
        a = np.asarray(Image.open(MANU / f"figures/panels/render_{name}.png").convert("RGBA"))
        ys, xs = np.where(a[..., 3] > 10)
        ax = ax_mm(8 + k * 42, 1, 34, 24)
        ax.imshow(a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.04,
            lab,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=S.SMALL_PT,
            color=S.NAVY_D if name.startswith("bio") else S.GOLD_D,
        )

    def arm_lines(ax, key):
        for d, c, m, lab in (
            (bio, S.NAVY_D, "o", "measured diameter (2.13 mm)"),
            (thk, S.GOLD_D, "s", "densified (3.17 mm)"),
        ):
            ax.plot(
                d.N_BRIDGE_LAYERS,
                d[key],
                marker=m,
                color=c,
                ms=3.6,
                mec="white",
                mew=0.5,
                label=lab,
            )
        ax.set_xticks(range(4, 10))
        ax.set_xlabel("bridge layers N")
        S.recessive_grid(ax)

    y1, h1 = 36, 40
    # A -- apparent modulus vs N
    axA = ax_mm(12, y1, 44, h1)
    arm_lines(axA, "E_app")
    axA.set_ylabel(r"apparent modulus $E_{app}$ (MPa)")
    axA.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.47),
        fontsize=5.6,
        handlelength=1.4,
        borderaxespad=0.2,
    )
    label_mm("A", 1, y1 - 4)

    # B -- percentile stress ratio kappa_99 (gauge window) vs N
    axB = ax_mm(70, y1, 44, h1)
    arm_lines(axB, "SCF_p99")
    axB.set_ylabel(r"$\kappa_{99} = P_{99}(\sigma_{VM})/\bar{\sigma}_{g}$")
    axB.set_ylim(1.5, 2.6)
    label_mm("B", 59, y1 - 4)

    # C -- tension partition vs N
    axC = ax_mm(128, y1, 42, h1)
    arm_lines(axC, "tens")
    axC.set_ylabel("tension-dominated volume fraction")
    axC.set_ylim(0, None)
    label_mm("C", 117, y1 - 4)

    # --- row 2: twist lever at N = 2 ---------------------------------------------------------
    y2, h2 = 96, 42

    def fstar_band(ax):
        # Dashed reference line at the as-translated design (f = 1). The tilt-matched
        # optimum f* (1.03, geometric verification) is within 0.03 of it and is not drawn.
        ax.axvline(1.0, color=S.RULE, lw=0.6, ls=(0, (2, 2)), zorder=0)

    def dec_line(ax, key, ylabel):
        m = np.isfinite(dec[key])
        ax.plot(dec.f[m], dec[key][m], marker="o", color=S.NAVY_D, ms=3.4, mec="white", mew=0.5)
        fstar_band(ax)
        ax.set_xlabel("twist scale f")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1.6)
        S.recessive_grid(ax)

    axD = ax_mm(12, y2, 44, h2)
    dec_line(axD, "E_app", r"apparent modulus $E_{app}$ (MPa)")
    ytop = axD.get_ylim()[1]
    axD.text(1.06, ytop * 0.97, "as translated\n(f = 1)", color=S.MUTED, fontsize=5.4, va="top")
    label_mm("D", 1, y2 - 4)

    axE = ax_mm(70, y2, 44, h2)
    dec_line(axE, "tens", "tension-dominated volume fraction")
    axE.set_ylim(0, None)
    label_mm("E", 59, y2 - 4)

    axF = ax_mm(128, y2, 42, h2)
    dec_line(axF, "rot", "platen rotation (° per 1 % strain)")
    label_mm("F", 117, y2 - 4)

    # --- row 3: closed-loop Bayesian optimization toward a target response -------------------
    opt = optimization_record()
    if opt is not None:
        y3, h3 = 158, 42
        trials = opt["trials"]
        ok = [t for t in trials if t.get("state") == "ok"]
        e_star, t_star = trials[0]["targets"]["E_app"], trials[0]["targets"]["tension"]
        best = min(ok, key=lambda t: t["J"])
        ver = opt["verified"]

        # G -- objective per trial (log scale) and best-so-far envelope
        axG = ax_mm(12, y3, 44, h3)
        idx = np.array([t["trial"] for t in trials])
        J = np.array([t["J"] for t in trials])
        okm = np.array([t.get("state") == "ok" for t in trials])
        axG.axvspan(-0.5, OPT_STARTUP - 0.5, color=S.TONE, alpha=0.45, lw=0, zorder=0)
        axG.plot(idx[okm], J[okm], "o", ms=3.2, color=S.NAVY_D, mec="white", mew=0.5)
        axG.step(
            idx,
            np.minimum.accumulate(np.where(okm, J, np.inf)),
            where="post",
            color=S.GOLD_D,
            lw=1.2,
        )
        axG.set_yscale("log")
        axG.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axG.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axG.yaxis.set_minor_formatter(mticker.NullFormatter())
        axG.set_xlabel("design evaluated (trial)")
        axG.set_ylabel("objective J")
        axG.text(
            0.97,
            0.95,
            "best so far",
            transform=axG.transAxes,
            ha="right",
            va="top",
            color=S.GOLD_D,
            fontsize=5.6,
        )
        S.recessive_grid(axG)
        label_mm("G", 1, y3 - 4)

        # H -- property plane: evaluated designs, target set, and the twist study for context
        axH = ax_mm(70, y3, 44, h3)
        m = np.isfinite(dec["E_app"]) & np.isfinite(dec["tens"])
        axH.plot(dec.E_app[m], dec.tens[m], "-", color=S.RULE, lw=0.7, zorder=1)
        axH.plot(
            dec.E_app[m],
            dec.tens[m],
            "o",
            ms=2.2,
            color=S.RULE,
            zorder=1,
            label="twist study (d = 2.13 mm, N = 2)",
        )
        alt = np.array([t["handedness"] == "alternating" for t in ok])
        E = np.array([t["E_app"] for t in ok])
        T = np.array([t["tension"] for t in ok])
        axH.plot(
            E[~alt],
            T[~alt],
            "o",
            ms=3.4,
            color=S.NAVY_D,
            mec="white",
            mew=0.5,
            zorder=3,
            label="optimizer, single handedness",
        )
        if alt.any():
            axH.plot(
                E[alt],
                T[alt],
                "o",
                ms=3.4,
                mfc="white",
                mec=S.NAVY_D,
                mew=0.8,
                zorder=3,
                label="optimizer, alternating",
            )
        top = max(0.40, float(np.nanmax(np.r_[T, dec.tens[m]])) * 1.08)
        axH.plot(
            [e_star, e_star],
            [t_star, top],
            color=S.GOLD_D,
            lw=2.4,
            alpha=0.55,
            solid_capstyle="butt",
            zorder=2,
            label="target set",
        )
        axH.plot(
            best["E_app"],
            best["tension"],
            "*",
            ms=8,
            color=S.GOLD_D,
            mec="white",
            mew=0.5,
            zorder=4,
            label="selected design",
        )
        if ver is not None:
            v = ver["verified"]
            axH.plot(
                v["E_app"],
                v["tension"],
                "*",
                ms=8,
                mfc="none",
                mec=S.INK,
                mew=0.6,
                zorder=5,
                label="selected, 0.5-mm mesh",
            )
        axH.set_ylim(0, top)
        axH.set_xlabel(r"apparent modulus $E_{app}$ (MPa)")
        axH.set_ylabel("tension-dominated volume fraction")
        axH.legend(fontsize=5.0, loc="upper right", handlelength=1.2, borderaxespad=0.3)
        S.recessive_grid(axH)
        label_mm("H", 59, y3 - 4)

        # I -- the selected design: render (same recipe as the top strip) and its variables
        rpng = MANU / "figures/panels/render_optimum.png"
        axI = ax_mm(122, y3 - 2, 48, h3 - 8)
        axI.axis("off")
        if rpng.exists():
            a = np.asarray(Image.open(rpng).convert("RGBA"))
            ys, xs = np.where(a[..., 3] > 10)
            axI.imshow(a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
        src = ver["verified"] if ver is not None else best
        hand = "alternating" if best["handedness"] == "alternating" else "single"
        fig.text(
            146 / FW,
            1 - (y3 + h3 - 4) / FH,
            f"f = {best['f']:.2f}, N = {best['N']}, d = {best['d']:.2f} mm, {hand} handedness\n"
            f"$E_{{app}}$ = {src['E_app']:.0f} MPa (target {e_star:.0f})\n"
            f"tension-dominated fraction {src['tension']:.2f} (target ≥ {t_star:.2f})",
            ha="center",
            va="top",
            fontsize=5.6,
            color=S.INK,
        )
        label_mm("I", 117, y3 - 4)
        vals["optimization"] = {
            "n_trials": len(trials),
            "n_ok": len(ok),
            "best": best,
            "verified": ver["verified"] if ver is not None else None,
        }
        fig.text(
            6 / FW,
            1 - 148.5 / FH,
            f"Closed-loop Bayesian optimization toward a target response "
            f"($E_{{app}}$ = {e_star:.0f} MPa, tension-dominated fraction ≥ {t_star:.2f}; H = 20 mm)",
            fontsize=S.BASE_PT,
            fontweight="bold",
            color=S.INK,
            va="top",
        )

    fig.text(
        6 / FW,
        1 - 28.5 / FH,
        "Rod diameter and bridge layers at the as-translated twist (f = 1, H = 24 mm)",
        fontsize=S.BASE_PT,
        fontweight="bold",
        color=S.INK,
        va="top",
    )
    fig.text(
        6 / FW,
        1 - 86.5 / FH,
        "Twist amplitude at N = 2 (H = 20 mm; all rings rotate in the same sense)",
        fontsize=S.BASE_PT,
        fontweight="bold",
        color=S.INK,
        va="top",
    )
    S.save(fig, OUT)
    VALUES.write_text(json.dumps(vals, indent=1, default=float))
    print(
        dec[["f", "E_app", "phi_gauge", "SCF_whole", "SCF", "tens", "incl", "rot"]]
        .round(3)
        .to_string(index=False)
    )
    print(
        arms[
            [
                "arm",
                "N_BRIDGE_LAYERS",
                "E_app",
                "phi_gauge",
                "SCF_p99_whole",
                "SCF_p99",
                "tens",
                "rot",
            ]
        ]
        .sort_values(["arm", "N_BRIDGE_LAYERS"])
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
