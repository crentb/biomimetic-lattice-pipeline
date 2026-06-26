# Copyright 2026 Cameron B. Renteria
# SPDX-License-Identifier: Apache-2.0
import numpy as nm
import os
script_name = os.path.splitext(os.path.basename(__file__))[0]

# ------------------------------------------------------------
# MESH FOR COMPRESSION SAMPLE
# ------------------------------------------------------------
filename_mesh = "compound_enamel_lattice.msh"

# ------------------------------------------------------------
# MATERIAL PARAMETERS
# Read from environment variables so the pipeline can sweep materials
# without modifying this file.
#
#   MATERIAL_E   Young's modulus in MPa  (default: 85000 — hydroxyapatite)
#   MATERIAL_NU  Poisson's ratio         (default: 0.30)
#
# Common print materials:
#   Photopolymer resin  : E ≈ 2500–3500 MPa, nu ≈ 0.40
#   FDM PLA             : E ≈ 3500 MPa,       nu ≈ 0.36
#   Hydroxyapatite (HAp): E ≈ 85000 MPa,      nu ≈ 0.30  (biological reference)
# ------------------------------------------------------------
E  = float(os.environ.get("MATERIAL_E",  "85000"))   # MPa
nu = float(os.environ.get("MATERIAL_NU", "0.30"))

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

C11 = lam + 2.0 * mu   # diagonal (normal-normal)
C12 = lam              # off-diagonal (normal-normal)
C44 = mu               # diagonal (shear-shear)

print(f"[material] E={E:.0f} MPa  nu={nu:.3f}  "
      f"C11={C11:.2f}  C12={C12:.2f}  C44={C44:.2f}")

materials = {
    'm': ({'D': [
        [C11, C12, C12, 0.0, 0.0, 0.0],
        [C12, C11, C12, 0.0, 0.0, 0.0],
        [C12, C12, C11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C44],
    ]},),
}

# ------------------------------------------------------------
# REGIONS — z-limits read dynamically from the mesh
# ------------------------------------------------------------
from sfepy.discrete.fem import Mesh as _Mesh
_mesh = _Mesh.from_file(filename_mesh)
_z = _mesh.coors[:, 2]
_z_min, _z_max = float(_z.min()), float(_z.max())
_tol = 0.001  # mm

regions = {
    'Omega': 'all',
    'Bottom': (f'vertices in z < {_z_min + _tol:.6f}', 'facet'),
    'Top':    (f'vertices in z > {_z_max - _tol:.6f}', 'facet'),
}

# ------------------------------------------------------------
# FIELDS & VARIABLES
# ------------------------------------------------------------
fields = {
    'disp': ('real', 'vector', 'Omega', 1),
}

variables = {
    'u': ('unknown field', 'disp', 0),
    'v': ('test field',    'disp', 'u'),
}

# ------------------------------------------------------------
# BOUNDARY CONDITIONS — COMPRESSION IN Z
# ------------------------------------------------------------
# Convention: keep Bottom fixed, move Top *downward* (negative z).
# Override via COMPRESS_DISP_MM env var (always stored as negative).
_compress_disp_mm = float(os.environ.get("COMPRESS_DISP_MM", "10.0"))
compress_disp = -abs(_compress_disp_mm)  # always negative  (mm, 4x scaled geometry)

ebcs = {
    'fix_bottom': ('Bottom', {'u.all': 0.0}),
    'compress_top': ('Top',  {'u.2': compress_disp}),
}

# ------------------------------------------------------------
# INTEGRALS, EQUATIONS, SOLVERS
# ------------------------------------------------------------
integrals = {
    'i': 2,
}

equations = {
    'balance': "dw_lin_elastic.i.Omega(m.D, v, u) = 0",
}

solvers = {
    'ls': ('ls.scipy_direct', {}),
    # eps_a/eps_r set to engineering tolerance (1e-6). Default 1e-10 is
    # unreachable for direct solvers on large meshes and produces spurious
    # "precision lower than solver options" warnings.
    'newton': ('nls.newton', {
        'use_implicit_dof_elimination': False,
        'eps_a': 1e-6,
        'eps_r': 1e-6,
    }),
}

# ------------------------------------------------------------
# POST-PROCESSING (SAME AS TENSION, JUST INTERPRETED AS COMPRESSION)
# ------------------------------------------------------------

class SimpleStruct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def compute_von_mises(stress):
    out = []
    for s in stress:
        flat = []
        for item in s:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)

        flat = nm.array(flat, dtype=float).ravel()
        if flat.size < 6:
            flat = nm.concatenate([flat, nm.zeros(6 - flat.size)])

        sxx, syy, szz, sxy, syz, sxz = flat[:6]

        vm = nm.sqrt(
            0.5 * (
                (sxx - syy)**2 +
                (syy - szz)**2 +
                (szz - sxx)**2 +
                6.0 * (sxy**2 + syz**2 + sxz**2)
            )
        )
        out.append(vm)
    return out

def compute_principal_stresses(stress):
    out = []
    for s in stress:
        flat = []
        for item in s:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)

        flat = nm.array(flat, dtype=float).ravel()
        if flat.size < 6:
            flat = nm.concatenate([flat, nm.zeros(6 - flat.size)])

        sxx, syy, szz, sxy, syz, sxz = flat[:6]

        T = nm.array([
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz]
        ], dtype=float)

        vals = nm.linalg.eigvalsh(T)
        out.append(vals)
    return out

def compute_strain_energy_density(stress, strain):
    out = []
    for s, e in zip(stress, strain):
        sflat, eflat = [], []

        for item in s:
            if hasattr(item, "__len__"):
                sflat.extend(item)
            else:
                sflat.append(item)

        for item in e:
            if hasattr(item, "__len__"):
                eflat.extend(item)
            else:
                eflat.append(item)

        sflat = nm.array(sflat, dtype=float).ravel()
        eflat = nm.array(eflat, dtype=float).ravel()

        if sflat.size < 6:
            sflat = nm.concatenate([sflat, nm.zeros(6 - sflat.size)])
        if eflat.size < 6:
            eflat = nm.concatenate([eflat, nm.zeros(6 - eflat.size)])

        W = 0.5 * nm.dot(sflat[:6], eflat[:6])
        out.append(W)
    return out

def compute_element_volumes(problem):
    """
    Compute volume of each tetrahedral element (vectorized).
    Returns array of element volumes matching the cell ordering.
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)  # 3D cells -> vertices
    n_cells = len(conn.offsets) - 1
    tet_conn = conn.indices.reshape(n_cells, 4)

    v0 = coors[tet_conn[:, 0]]
    v1 = coors[tet_conn[:, 1]]
    v2 = coors[tet_conn[:, 2]]
    v3 = coors[tet_conn[:, 3]]

    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0

    cross = nm.cross(d2, d3)
    det = nm.sum(d1 * cross, axis=1)
    volumes = nm.abs(det) / 6.0

    return volumes


def compute_element_centroids(problem):
    """
    Compute centroid (x, y, z) of each tetrahedral element.
    Returns array of shape (n_elements, 3).
    """
    coors = problem.domain.get_mesh_coors()
    cmesh = problem.domain.cmesh

    conn = cmesh.get_conn(3, 0)
    n_cells = len(conn.offsets) - 1
    tet_conn = conn.indices.reshape(n_cells, 4)

    centroids = (coors[tet_conn[:, 0]] + coors[tet_conn[:, 1]]
                 + coors[tet_conn[:, 2]] + coors[tet_conn[:, 3]]) / 4.0
    return centroids


def compute_reaction_force_from_stress(problem, stress, element_volumes):
    """
    Compute reaction force by integrating stress over the volume.

    For a body in static equilibrium with no body forces:
      F_z = (1/H) * sum( sigma_zz_i * V_i )

    This is equivalent to integrating the traction over any cross-section
    and avoids the DOF-indexing issues with SfePy's residual vector.

    Returns [Fx, Fy, Fz] in Newtons.
    """
    coors = problem.domain.get_mesh_coors()
    height = coors[:, 2].max() - coors[:, 2].min()

    # Extract stress components per element
    n_el = len(stress)
    sxx_vals = nm.zeros(n_el)
    syy_vals = nm.zeros(n_el)
    szz_vals = nm.zeros(n_el)
    sxy_vals = nm.zeros(n_el)
    syz_vals = nm.zeros(n_el)
    sxz_vals = nm.zeros(n_el)

    for idx, s in enumerate(stress):
        flat = []
        for item in s:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)
        flat = nm.array(flat, dtype=float).ravel()
        if flat.size < 6:
            flat = nm.concatenate([flat, nm.zeros(6 - flat.size)])
        sxx_vals[idx] = flat[0]
        syy_vals[idx] = flat[1]
        szz_vals[idx] = flat[2]
        sxy_vals[idx] = flat[3]
        syz_vals[idx] = flat[4]
        sxz_vals[idx] = flat[5]

    # F_i = (1/H) * integral( sigma_iz dV )
    # For z-normal cross-section: F_x = int(sxz), F_y = int(syz), F_z = int(szz)
    Fx = nm.sum(sxz_vals * element_volumes) / height
    Fy = nm.sum(syz_vals * element_volumes) / height
    Fz = nm.sum(szz_vals * element_volumes) / height

    return nm.array([Fx, Fy, Fz])

def _reshape_cell_data(arr):
    if arr.ndim == 1:
        return arr[:, nm.newaxis, nm.newaxis, nm.newaxis]
    if arr.ndim == 2:
        return arr[:, nm.newaxis, :, nm.newaxis]
    return arr

def post_process(out, problem, state, extend=False):

    # Debug: count vertices in Top region
    top_region = problem.domain.regions['Top']
    num_top_vertices = top_region.get_entities(0).shape[0]
    print("\n=== DEBUG: Top region vertex count:", num_top_vertices, "===\n")
    
    # Check Mesh Boundaries
    coors = problem.domain.get_mesh_coors()
    print("Mesh bounding box:")
    print("  x:", coors[:,0].min(), "to", coors[:,0].max())
    print("  y:", coors[:,1].min(), "to", coors[:,1].max())
    print("  z:", coors[:,2].min(), "to", coors[:,2].max())

    # Stress (MPa)
    stress = problem.evaluate(
        'ev_cauchy_stress.i.Omega(m.D, u)',
        mode='el_avg',
        verbose=False,
    )

    # --- Compute element volumes and actual cross-section area ---
    element_volumes = compute_element_volumes(problem)
    total_volume = element_volumes.sum()
    coors = problem.domain.get_mesh_coors()
    height = coors[:, 2].max() - coors[:, 2].min()
    actual_area = total_volume / height  # mm^2

    # Bounding box area (for comparison)
    width  = coors[:, 0].max() - coors[:, 0].min()
    depth  = coors[:, 1].max() - coors[:, 1].min()
    bbox_area = width * depth  # mm^2

    # Extract all 6 stress components per element: [sxx, syy, szz, sxy, syz, sxz]
    n_el = len(stress)
    stress_components = nm.zeros((n_el, 6))
    for idx, s in enumerate(stress):
        flat = []
        for item in s:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)
        flat = nm.array(flat, dtype=float).ravel()
        if flat.size < 6:
            flat = nm.concatenate([flat, nm.zeros(6 - flat.size)])
        stress_components[idx] = flat[:6]

    szz_vals = stress_components[:, 2]

    # Volume-weighted average stress
    sigma_zz_vol_avg = nm.sum(szz_vals * element_volumes) / total_volume

    # Reaction force via volume-integrated stress (correct method)
    reaction = compute_reaction_force_from_stress(problem, stress, element_volumes)
    Fz_stress = reaction[2]  # N

    print("\n=== Reaction force from stress-volume integral ===")
    print("Total mesh volume     = {:.4f} mm^3".format(total_volume))
    print("Specimen height       = {:.4f} mm".format(height))
    print("Actual cross-sect area = {:.4f} mm^2".format(actual_area))
    print("Bounding box area     = {:.4f} mm^2  (NOT used)".format(bbox_area))
    print("Solid fraction        = {:.1f}%".format(100.0 * actual_area / bbox_area))
    print("")
    print("Vol-weighted avg sigma_zz = {:.6f} MPa".format(sigma_zz_vol_avg))
    print("Reaction Fz = {:.6f} N".format(Fz_stress))
    print("Reaction Fx = {:.6f} N".format(reaction[0]))
    print("Reaction Fy = {:.6f} N".format(reaction[1]))
    print("====================================================\n")

    # Strain (dimensionless)
    strain = problem.evaluate(
        'ev_cauchy_strain.i.Omega(u)',
        mode='el_avg',
        verbose=False,
    )

    # Extract all 6 strain components: [exx, eyy, ezz, gxy, gyz, gxz]
    strain_components = nm.zeros((n_el, 6))
    for idx, e in enumerate(strain):
        flat = []
        for item in e:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)
        flat = nm.array(flat, dtype=float).ravel()
        if flat.size < 6:
            flat = nm.concatenate([flat, nm.zeros(6 - flat.size)])
        strain_components[idx] = flat[:6]

    # Derived fields
    mises    = compute_von_mises(stress)
    p_stress = compute_principal_stresses(stress)
    energy   = compute_strain_energy_density(stress, strain)

    stress_np = _reshape_cell_data(nm.array(stress))
    strain_np = _reshape_cell_data(nm.array(strain))
    mises_np  = _reshape_cell_data(nm.array(mises))
    p_np      = _reshape_cell_data(nm.array(p_stress))
    energy_np = _reshape_cell_data(nm.array(energy))

    out['cauchy_stress'] = SimpleStruct(name='cauchy_stress', mode='cell', data=stress_np, var='u', step=0)
    out['von_mises']     = SimpleStruct(name='von_mises',     mode='cell', data=mises_np,  var='u', step=0)
    out['principal_stress'] = SimpleStruct(name='principal_stress', mode='cell', data=p_np, var='u', step=0)
    out['strain_energy'] = SimpleStruct(name='strain_energy', mode='cell', data=energy_np, var='u', step=0)

    out['reaction_force'] = SimpleStruct(
        name='reaction_force',
        mode='scalar',
        data=nm.array([Fz_stress], dtype=float),
        var='u',
        step=0,
    )

    nm.savetxt(
        "reaction_force_bottom_z_compression.txt",
        nm.array([Fz_stress]),
        header="Reaction force Fz in N (from stress-volume integral)"
    )

    out['cauchy_strain'] = SimpleStruct(
        name='cauchy_strain',
        mode='cell',
        data=strain_np,
        var='u',
        step=0,
    )

    # ------------------------------------------------------------
    # CSV EXPORTS
    # ------------------------------------------------------------
    disp_applied = compress_disp  # mm (negative for compression)

    global_csv = nm.array([
        disp_applied,
        sigma_zz_vol_avg,
        Fz_stress,
        nm.sum(nm.array(mises) * element_volumes) / total_volume,
        nm.sum(nm.array(energy) * element_volumes) / total_volume,
        nm.sum(nm.array(p_stress)[:, 0] * element_volumes) / total_volume,
        nm.sum(nm.array(p_stress)[:, 1] * element_volumes) / total_volume,
        nm.sum(nm.array(p_stress)[:, 2] * element_volumes) / total_volume,
    ])

    nm.savetxt(
        "global_results_compression.csv",
        global_csv.reshape(1, -1),
        delimiter=",",
        header="disp_mm,avg_sigma_zz_MPa,force_N,avg_von_mises_MPa,avg_energy_MPa,p1_MPa,p2_MPa,p3_MPa",
        comments=""
    )

    centroids = compute_element_centroids(problem)

    elem_data = nm.column_stack([
        stress_components,          # sxx, syy, szz, sxy, syz, sxz
        strain_components,          # exx, eyy, ezz, gxy, gyz, gxz
        nm.array(mises),
        nm.array(energy),
        nm.array(p_stress)[:, 0],
        nm.array(p_stress)[:, 1],
        nm.array(p_stress)[:, 2],
        element_volumes,
        centroids,
    ])

    nm.savetxt(
        "element_results_compression.csv",
        elem_data,
        delimiter=",",
        header=("sxx_MPa,syy_MPa,szz_MPa,sxy_MPa,syz_MPa,sxz_MPa,"
                "exx,eyy,ezz,gxy,gyz,gxz,"
                "von_mises_MPa,energy_MPa,p1_MPa,p2_MPa,p3_MPa,"
                "volume_mm3,cx_mm,cy_mm,cz_mm"),
        comments=""
    )

    # Multi-point F-d curve (linear elastic scaling — 20 incremental steps + zero point).
    # For linear elastic material F(δ) = K·δ, so scaling the single solve gives the
    # exact same result as running 20 separate solves. This produces a proper curve
    # for visualization and trapz energy integration in extract_metrics.py.
    N_LOAD_STEPS = 20
    disp_steps  = nm.linspace(0.0, disp_applied, N_LOAD_STEPS + 1)
    force_steps = Fz_stress * (disp_steps / disp_applied)

    nm.savetxt(
        "force_displacement_compression.csv",
        nm.column_stack([disp_steps, force_steps]),
        delimiter=",",
        header="displacement_mm,force_N",
        comments=""
    )

    return out

options = {
    'output_dir': '.',
    'save_format': 'vtk',   # 'vtkh' requires h5py and can silently fail; plain vtk always works
    'post_process_hook': 'post_process',
}