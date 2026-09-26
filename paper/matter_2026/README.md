# Matter manuscript (2026): figure, table, and check scripts

This folder holds the scripts behind the figures, Supplemental tables, and measurement checks of the
manuscript *Machine-Learning Translation of Tooth Enamel Microarchitecture into Additively
Manufactured Lattices with Programmable Mechanics* (Renteria, Grimm, Yunker, Parkinson, Arola;
submitted to *Matter*). The analysis and campaign drivers used by the paper are package-level
scripts in [`../../scripts/`](../../scripts/):

| Script | Role in the paper |
|---|---|
| `analyze_load_paths.py` | principal-stress trajectories, tension-dominated volume fraction, ring-resolved stresses, platen rotation |
| `compute_robust_scf.py`, `compute_gauge_metrics.py` | percentile stress ratio κ₉₉ and gauge-window normalization |
| `estimate_contact_onset.py` | strain at which neighboring rods of the densified lattices would touch |
| `run_revision_compute_2026_09.py` | twist-scale study and mesh-convergence campaign |
| `run_controls_2026_09.py` | alternating-handedness and rotation-constrained-platen controls |
| `run_inverse_design_2026_09.py` | closed-loop Bayesian design optimization (Optuna TPE) through the full CAD–FEA pipeline |

## Contents

| Path | Contents |
|---|---|
| `measurement/track_rods_piv_signcorrected.py` | PIV-guided rod tracking with the displacement sign corrected (Supplemental Section S1). `--sign +1` reproduces the original trajectories exactly |
| `scripts/_style_v2.py` | shared figure style (validated palette, Arial 6–9 pt, Cell Press widths) |
| `scripts/make_fig*.py`, `make_figS*.py` | main-text Figures 2–7 and Supplemental figures |
| `scripts/make_si_tables_v2.py` | Supplemental design-space and mesh tables |
| `scripts/si_measurement_checks.py`, `si_fabric_crosschecks.py` | tortuosity noise floor, band period, fabric axis, tracking fidelity, Woodcock fabric, clustering |
| `scripts/extract_canonical_som.py`, `remap_canonical_som_labels.py`, `som_signcheck.py` | regeneration of the original SOM band map and the sign-corrected band partition |
| `scripts/render_models_v2.py`, `make_thumbnails_v*.py` | off-screen renders of the digital twin and lattices; Figure 1 thumbnails |
| `tikz/` | LaTeX/TikZ sources of Figure 1 and the graphical abstract |

## Layout and data

The scripts were run from `<data root>/biomimetic_pipeline/manuscript/matter_v2/scripts/` and resolve
every input relative to their own location: `MANU` (their parent folder), `BIO = MANU.parents[1]`
(the folder holding `runs/`), and `REPO = MANU.parents[2]` (the folder holding the image-analysis
outputs `output_piv/`, `output_piv_signcorrected/`, `output_validation_100/`, and
`roi_imagestack_100/`). To re-run them, place this folder at that position relative to the
deposited data (to be archived at publication). Each script's docstring lists its inputs and
outputs; the rebuild order is

1. `measurement/track_rods_piv_signcorrected.py` (sign-corrected trajectories);
2. `extract_canonical_som.py`, `remap_canonical_som_labels.py`, `som_signcheck.py`;
3. the package-level analysis scripts above, per run directory;
4. `render_models_v2.py`, then `make_thumbnails_v3.py`;
5. `si_measurement_checks.py`, `si_fabric_crosschecks.py`, the `make_fig*.py` scripts, and
   finally `make_si_tables_v2.py`.

The copies in this repository follow its lint rules (ruff, black): statements separated by
semicolons are on separate lines and unused imports are removed. Their logic is identical to the
scripts that produced the published figures.
