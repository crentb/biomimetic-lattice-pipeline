"""Minimal LaTeX report for Phase 1.

Writes `report.tex` into run_dir/report/ using a template and compiles it via
pdflatex if available. Gracefully degrades to `.tex` only when pdflatex is
absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "specimen_report.tex"


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    cad_params = _maybe_json(run_dir / "cad_params.json")
    metrics = _maybe_json(run_dir / "metrics.json")

    ctx = _build_context(run_dir, cad_params, metrics)
    tex = _render_template(ctx)

    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex)

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        try:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                cwd=str(report_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                cwd=str(report_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.SubprocessError:
            pass
    return tex_path


def _maybe_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _latex_escape(s: Any) -> str:
    """Escape LaTeX-special characters so arbitrary user strings (run names,
    timestamps with colons, specimen IDs with underscores, file paths, etc.)
    render inside \\texttt{} / body text without triggering math-mode errors.
    """
    if s is None:
        return "--"
    text = str(s)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def _build_context(
    run_dir: Path, cad_params: Dict[str, Any], metrics: Dict[str, Any]
) -> Dict[str, Any]:
    prov = cad_params.get("provenance", {}) if isinstance(cad_params, dict) else {}
    pipeline_meta = metrics.get("_pipeline", {}) if isinstance(metrics, dict) else {}

    def fmt(key: str, default="--", prec: int = 3) -> str:
        v = metrics.get(key)
        if v is None or v != v:
            return default
        try:
            return f"{float(v):.{prec}g}"
        except (TypeError, ValueError):
            return str(v)

    def cad_fmt(key: str, default="--", prec: int = 3) -> str:
        v = cad_params.get(key)
        if v is None:
            return default
        try:
            return f"{float(v):.{prec}g}"
        except (TypeError, ValueError):
            return str(v)

    return {
        "specimen_id": _latex_escape(prov.get("morphometrics_specimen_id", "unknown")),
        "run_name": _latex_escape(pipeline_meta.get("run_name", run_dir.name)),
        "run_dir": _latex_escape(str(run_dir)),
        "timestamp": _latex_escape(pipeline_meta.get("run_timestamp_iso", "")),
        "rod_diameter": cad_fmt("ROD_DIAMETER"),
        "center_spacing": cad_fmt("CENTER_SPACING"),
        "n_rings": cad_fmt("N_RINGS", prec=2),
        "n_bridge_layers": cad_fmt("N_BRIDGE_LAYERS", prec=2),
        "bridge_diameter": cad_fmt("BRIDGE_DIAMETER"),
        "twist_type": _latex_escape(cad_params.get("TWIST_TYPE", "--")),
        "enamel_thickness": cad_fmt("ENAMEL_THICKNESS"),
        "biology_scale_factor": cad_fmt("biology_scale_factor", prec=2),
        "E_effective_mpa": fmt("E_effective_MPa"),
        "vm_mean_mpa": fmt("VM_mean_MPa"),
        "vm_p50_mpa": fmt("VM_P50_MPa"),
        "vm_p99_mpa": fmt("VM_P99_MPa"),
        "scf": fmt("SCF"),
        "reaction_force_n": fmt("reaction_force_N"),
        "specific_toughness": fmt("specific_toughness_mJ_per_MPa"),
        "compress_disp_mm": pipeline_meta.get("compress_disp_mm", "--"),
        "material_E_mpa": pipeline_meta.get("material_E_MPa", "--"),
        "material_nu": pipeline_meta.get("material_nu", "--"),
    }


def _render_template(ctx: Dict[str, Any]) -> str:
    template = _SPECIMEN_TEMPLATE
    # Simple %(key)s substitution (safer than Jinja for Phase 1 — no deps).
    return template % ctx


_SPECIMEN_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\title{Biomimetic Pipeline Specimen Report \\ \texttt{%(specimen_id)s}}
\author{biomimetic\_pipeline}
\date{%(timestamp)s}
\begin{document}
\maketitle

\section*{Run}
\begin{tabular}{ll}
\toprule
Run name    & \texttt{%(run_name)s} \\
Run dir     & \texttt{%(run_dir)s} \\
Timestamp   & %(timestamp)s \\
\bottomrule
\end{tabular}

\section*{CAD Parameters (mapped from morphometrics)}
\begin{tabular}{lr}
\toprule
ROD\_DIAMETER (mm)           & %(rod_diameter)s \\
CENTER\_SPACING (mm)         & %(center_spacing)s \\
BRIDGE\_DIAMETER (mm)        & %(bridge_diameter)s \\
ENAMEL\_THICKNESS (mm)       & %(enamel_thickness)s \\
N\_RINGS                     & %(n_rings)s \\
N\_BRIDGE\_LAYERS            & %(n_bridge_layers)s \\
TWIST\_TYPE                  & %(twist_type)s \\
biology\_scale\_factor       & %(biology_scale_factor)s \\
\bottomrule
\end{tabular}

\section*{FEA Summary (single-load Phase 1)}
\begin{tabular}{lr}
\toprule
Applied displacement (mm)    & %(compress_disp_mm)s \\
Material $E$ (MPa)           & %(material_E_mpa)s \\
Material $\nu$               & %(material_nu)s \\
Reaction force (N)           & %(reaction_force_n)s \\
Effective modulus (MPa)      & %(E_effective_mpa)s \\
$\overline{\mathrm{VM}}$ (MPa) & %(vm_mean_mpa)s \\
VM P50 (MPa)                 & %(vm_p50_mpa)s \\
VM P99 (MPa)                 & %(vm_p99_mpa)s \\
SCF                          & %(scf)s \\
Specific toughness (mJ/MPa)  & %(specific_toughness)s \\
\bottomrule
\end{tabular}

\paragraph{Notes.} Phase 1 uses a fixed applied displacement. The Phase 2
strain solver will instead solve for the displacement that produces a target
representative VM stress (100--200 MPa band) and report
\texttt{critical\_strain\_at\_200MPa} alongside a new crack-deflection metric.

\end{document}
"""
