"""Solve for the compressive displacement (strain) that produces a target
representative von-Mises stress in the lattice.

Given the mesh + material, we start with a seed displacement, run FEA once,
read the resulting representative VM stress, and scale the displacement
linearly (justified by small-strain linear elasticity) until within tolerance
of the target. No re-mesh, no re-CAD — only COMPRESS_DISP_MM changes.

Returns a StrainSolveResult with all iterations, the final accepted FEA
outputs, and a derived `critical_strain` = disp / specimen_height.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from biomimetic_pipeline.fea.fea_runner import FeaRunResult, run_one_iteration


@dataclass
class StrainSolveIteration:
    iter_idx: int
    disp_mm: float
    vm_value_mpa: float
    field: str
    within_tolerance: bool


@dataclass
class StrainSolveResult:
    accepted: bool
    target_mpa: float
    target_field: str
    final_iter: StrainSolveIteration
    iterations: List[StrainSolveIteration]
    fea_result: FeaRunResult
    critical_strain: float  # disp / specimen_height
    specimen_height_mm: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "target_mpa": self.target_mpa,
            "target_field": self.target_field,
            "critical_disp_mm": self.final_iter.disp_mm,
            "critical_strain": self.critical_strain,
            "specimen_height_mm": self.specimen_height_mm,
            "vm_value_at_critical_mpa": self.final_iter.vm_value_mpa,
            "iterations": [
                {
                    "iter": it.iter_idx,
                    "disp_mm": it.disp_mm,
                    "vm_value_mpa": it.vm_value_mpa,
                    "within_tolerance": it.within_tolerance,
                }
                for it in self.iterations
            ],
        }


RUNNER_SIG = Callable[..., FeaRunResult]


def solve_to_target_stress(
    mesh_source: Path,
    iter_root_dir: Path,
    specimen_height_mm: float,
    material_E_mpa: float,
    material_nu: float,
    target_mpa: float = 200.0,
    target_field: str = "avg_von_mises_MPa",
    seed_disp_mm: float = 1.0,
    tolerance_pct: float = 5.0,
    max_iters: int = 5,
    load_mode: str = "compression",
    element_type: str = "tet",
    runner: Optional[RUNNER_SIG] = None,
) -> StrainSolveResult:
    """Iteratively tune the applied displacement to hit a target VM stress.

    ``load_mode`` selects the load case ("compression" or "tension"); it is
    threaded straight through to the FEA runner. The bisection logic is
    load-case-agnostic -- it scales a displacement magnitude against the
    (always non-negative) von-Mises response -- so tension needs no special
    handling here.

    ``element_type`` ("tet" / "hex", added 2026-05-23 for the digital_twin
    model-type) is similarly load-case-agnostic and is just threaded through.
    For "hex", the fea_runner swaps the stock script's two tet-only cell
    kernels for 8-node hex equivalents -- the strain solver doesn't care.
    """
    iter_root_dir = Path(iter_root_dir)
    iter_root_dir.mkdir(parents=True, exist_ok=True)

    run_fea = runner if runner is not None else run_one_iteration

    disp = float(seed_disp_mm)
    iterations: List[StrainSolveIteration] = []
    latest_result: Optional[FeaRunResult] = None
    tol = float(target_mpa) * (float(tolerance_pct) / 100.0)

    for i in range(1, max_iters + 1):
        iter_dir = iter_root_dir / f"iter_{i}"
        fea_result = run_fea(
            mesh_source,
            iter_dir,
            material_E_mpa,
            material_nu,
            disp,
            load_mode=load_mode,
            element_type=element_type,
        )
        vm_value = _extract_target_field(fea_result, target_field)
        within = abs(vm_value - target_mpa) <= tol
        iterations.append(
            StrainSolveIteration(
                iter_idx=i,
                disp_mm=disp,
                vm_value_mpa=vm_value,
                field=target_field,
                within_tolerance=within,
            )
        )
        latest_result = fea_result
        if within:
            break
        if vm_value <= 1e-9:
            # Avoid divide-by-zero; double the displacement and retry.
            disp = disp * 2.0
        else:
            disp = disp * (target_mpa / vm_value)

    assert latest_result is not None
    final_iter = iterations[-1]
    critical_strain = float(final_iter.disp_mm) / max(float(specimen_height_mm), 1e-9)

    result = StrainSolveResult(
        accepted=final_iter.within_tolerance,
        target_mpa=float(target_mpa),
        target_field=target_field,
        final_iter=final_iter,
        iterations=iterations,
        fea_result=latest_result,
        critical_strain=critical_strain,
        specimen_height_mm=float(specimen_height_mm),
    )

    summary_path = iter_root_dir / "strain_solve_summary.json"
    summary = result.as_dict()
    summary["load_mode"] = load_mode
    summary_path.write_text(json.dumps(summary, indent=2))
    return result


def _extract_target_field(fea_result: FeaRunResult, field: str) -> float:
    if field == "avg_von_mises_MPa":
        return float(fea_result.avg_von_mises_mpa)
    # For percentile targets, compute on-the-fly from element_results CSV.
    return _percentile_from_element_results(fea_result.element_results_path, field)


def _percentile_from_element_results(path: Path, field: str) -> float:
    try:
        import numpy as np
    except ImportError:
        return float("nan")
    percentiles = {
        "VM_P50_MPa": 50.0,
        "VM_P90_MPa": 90.0,
        "VM_P95_MPa": 95.0,
        "VM_P99_MPa": 99.0,
        "VM_mean_MPa": None,
    }
    if field not in percentiles:
        return float("nan")

    import csv as _csv

    vm: List[float] = []
    vols: List[float] = []
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                vm.append(float(row["von_mises_MPa"]))
                vols.append(float(row["volume_mm3"]))
            except (KeyError, ValueError):
                continue
    if not vm:
        return float("nan")
    vm_arr = np.asarray(vm)
    vol_arr = np.asarray(vols)
    pct = percentiles[field]
    if pct is None:
        total = float(vol_arr.sum())
        return float((vm_arr * vol_arr).sum() / total) if total > 0 else float(vm_arr.mean())
    order = np.argsort(vm_arr)
    sorted_vm = vm_arr[order]
    sorted_w = vol_arr[order]
    cdf = np.cumsum(sorted_w) / max(sorted_w.sum(), 1e-12)
    idx = int(np.searchsorted(cdf, pct / 100.0))
    idx = min(idx, sorted_vm.size - 1)
    return float(sorted_vm[idx])
