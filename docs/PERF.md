# Performance Profile

## Flame-graph profile — crack-deflection metric (2026-07-15)

`biomimetic_pipeline/metrics/crack_deflection.py::compute` — the
streamline-tracing tortuosity metric — profiled with cProfile on a
fixed-seed synthetic `element_results_compression.csv` at genuine FEA
post-processing scale: 200,000 element centroids (50×50×80 jittered
lattice over a 10×10×20 mm box) with a plausible max-principal-stress
field. Flame graphs rendered with flameprof; Apple-Silicon macOS,
Python 3.13.

The metric runs once per design evaluation, so per-call cost is a
per-trial tax in sweep and closed-loop optimization campaigns. (The
CAD/FEA stages run externally in separate environments and are outside
this profile.)

| | Baseline | After |
|---|---|---|
| Wall time | 1.12 s | **0.09 s (12.4×)** |
| Function calls | 4,144,632 | 46,055 (90× fewer) |
| Result values | — | identical to 9 decimals, all six fields |

Baseline attribution:

| Phase | Cost | Share |
|---|---|---|
| `_trilinear` — 82,635 scalar calls (64 streamlines × ~430 steps × 3 components) | 0.52 s | 46% |
| `_load_elements` — `csv.DictReader`, one dict per row × 200k rows | 0.33 s | 30% |
| `_rasterize_to_grid` — per-element Python accumulation loop | 0.10 s | 9% |

The integrator was the expected hotspot; CSV parsing at 30% was the
surprise — it out-cost the rasterization loop the profile was expected
to indict (9%).

The fixes (all numerics-preserving, all in `crack_deflection.py`):

1. **Batched streamline integrator** — all 64 seeds advance together;
   trilinear cell indices and corner weights are computed once per step
   and shared across the three direction components. Same expressions,
   same break semantics (stall = stop without moving; top/xy-exit = stop
   after the move), same arc/chord bookkeeping.
2. **Fast CSV path** — header-resolved column indices + `np.loadtxt`
   (C parser); any parse error falls back to the original tolerant
   `DictReader` loop, so dirty files behave exactly as before.
3. **Vectorized rasterization** — `np.add.at` scatter-add
   (element-order accumulation, identical sums to the loop).

Equivalence fingerprint (identical in both runs): n=64 ·
mean 1.029302420 · p90 1.052157736 · max 1.055952942 ·
arc 20.912945589 mm · chord 20.316867959 mm.

At roughly 1 s saved per evaluation, a 500-trial optimization campaign
recovers about 8 minutes of pure metric overhead — and the metric no
longer competes with the FEA solver for the per-trial budget.

Flame graphs (open in a browser; box width is cumulative time, click to
zoom):

- [`profiling/flame_crack_baseline.svg`](profiling/flame_crack_baseline.svg)
- [`profiling/flame_crack_after.svg`](profiling/flame_crack_after.svg)

Numbers are from the machine above; expect variation elsewhere.
