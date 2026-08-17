"""Accuracy diagnostics and ground-truth export for the BFS example."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np
from scipy.spatial import cKDTree

import utils.utils_flowsolver as flu


WALL_SAMPLE_X = np.linspace(0.02, 30.0, 1500)


def _boundary_index(fs, name: str) -> int:
    return int(fs.boundaries.loc[name].idx)


def boundary_measures(fs) -> dict[str, float]:
    one = dolfin.Constant(1.0)
    return {
        name: float(dolfin.assemble(one * fs.ds(_boundary_index(fs, name))))
        for name in fs.boundaries.index
    }


def prescribed_fluxes(fs, actuation: float) -> dict[str, float]:
    """Integrate the prescribed inlet and actuator expressions directly."""

    geometry = fs.params_mesh.user_data
    inlet_profile = dolfin.Expression(
        "4.0*uinf*(x[1]-y_step)*(y_top-x[1])/pow(y_top-y_step, 2)",
        degree=4,
        uinf=fs.params_flow.uinf,
        y_step=geometry["y_step"],
        y_top=geometry["y_top"],
    )
    fs.set_actuators_u_ctrl([actuation])
    actuator_v = fs.params_control.actuator_list[0].expression[1]
    return {
        "inlet_inward": float(
            dolfin.assemble(inlet_profile * fs.ds(_boundary_index(fs, "inlet")))
        ),
        "actuator_inward": float(
            dolfin.assemble(-actuator_v * fs.ds(_boundary_index(fs, "actuator")))
        ),
    }


def _free_dof_residual(fs) -> float:
    residual_form, _ = fs._make_varf_steady(initial_guess=fs.fields.UP0)
    residual = dolfin.assemble(residual_form)
    values = residual.get_local()
    for bc in fs._make_BCs().bcu:
        dofs = np.fromiter(bc.get_boundary_values().keys(), dtype=np.int64)
        values[dofs] = 0.0
    residual.set_local(values)
    residual.apply("insert")
    return float(dolfin.norm(residual) / math.sqrt(fs.W.dim()))


def _zero_crossings(x: np.ndarray, values: np.ndarray) -> list[float]:
    roots = []
    for left in range(len(x) - 1):
        f0 = values[left]
        f1 = values[left + 1]
        if not np.isfinite(f0 + f1) or f0 == f1:
            continue
        if f0 == 0.0:
            roots.append(float(x[left]))
        elif f0 * f1 < 0.0:
            roots.append(float(x[left] - f0 * (x[left + 1] - x[left]) / (f1 - f0)))
    return roots


def _negative_to_positive_crossings(
    x: np.ndarray, values: np.ndarray
) -> list[float]:
    roots = []
    for left in range(len(x) - 1):
        f0 = values[left]
        f1 = values[left + 1]
        if np.isfinite(f0 + f1) and f0 < 0.0 <= f1 and f0 != f1:
            roots.append(
                float(x[left] - f0 * (x[left + 1] - x[left]) / (f1 - f0))
            )
    return roots


def wall_observations(fs) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Sample lower/upper wall shear and locate sign changes."""

    dux_dy = dolfin.project(
        fs.fields.U0[0].dx(1), fs.P, solver_type="mumps"
    )
    lower = np.array([dux_dy(float(x), 0.0) for x in WALL_SAMPLE_X])
    upper = np.array([dux_dy(float(x), 2.0) for x in WALL_SAMPLE_X])
    observations = {
        "x": WALL_SAMPLE_X.copy(),
        "lower_du_dy": lower,
        "upper_du_dy": upper,
        "lower_skin_friction": 2.0 * lower / fs.params_flow.Re,
        "upper_skin_friction": -2.0 * upper / fs.params_flow.Re,
    }
    lower_reattachments = _negative_to_positive_crossings(WALL_SAMPLE_X, lower)
    roots = {
        "lower_wall_shear_zeros": _zero_crossings(WALL_SAMPLE_X, lower),
        "upper_wall_shear_zeros": _zero_crossings(WALL_SAMPLE_X, upper),
        "primary_lower_reattachment": (
            lower_reattachments[0] if lower_reattachments else None
        ),
    }
    return observations, roots


def common_velocity_samples(fs) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate velocity at a mesh-independent set of interior points."""

    geometry = fs.params_mesh.user_data
    upstream = np.array(
        [
            (x, y)
            for x in np.arange(geometry["x_in"] + 0.25, 0.0, 0.25)
            for y in np.linspace(1.05, 1.95, 10)
        ]
    )
    downstream = np.array(
        [
            (x, y)
            for x in np.arange(0.05, geometry["x_out"], 0.25)
            for y in np.linspace(0.05, 1.95, 20)
        ]
    )
    points = np.vstack((upstream, downstream))
    values = np.array([fs.fields.U0(float(x), float(y)) for x, y in points])
    return points, values


def solution_diagnostics(fs, actuation: float) -> tuple[dict, dict[str, np.ndarray]]:
    u = fs.fields.U0
    normal = dolfin.FacetNormal(fs.mesh)
    flux = {
        name: float(
            dolfin.assemble(
                dolfin.dot(u, normal) * fs.ds(_boundary_index(fs, name))
            )
        )
        for name in fs.boundaries.index
    }
    inlet_inward = -flux["inlet"]
    outlet_outward = flux["outlet"]
    actuator_outward = flux["actuator"]
    net_outward = float(sum(flux.values()))
    wall, roots = wall_observations(fs)
    points, sampled_velocity = common_velocity_samples(fs)

    diagnostics = {
        "actuation": float(actuation),
        "reynolds": float(fs.params_flow.Re),
        "mesh_vertices": int(fs.mesh.num_vertices()),
        "mesh_cells": int(fs.mesh.num_cells()),
        "velocity_dofs": int(fs.V.dim()),
        "pressure_dofs": int(fs.P.dim()),
        "mixed_dofs": int(fs.W.dim()),
        "kinetic_energy": float(
            0.5 * dolfin.assemble(dolfin.inner(u, u) * dolfin.dx)
        ),
        "divergence_l2": float(
            math.sqrt(dolfin.assemble(dolfin.div(u) ** 2 * dolfin.dx))
        ),
        "free_dof_residual_rms": _free_dof_residual(fs),
        "boundary_flux_outward": flux,
        "inlet_flux_inward": inlet_inward,
        "outlet_flux_outward": outlet_outward,
        "actuator_flux_inward": -actuator_outward,
        "net_flux_outward": net_outward,
        "relative_mass_imbalance": abs(net_outward) / max(abs(inlet_inward), 1e-30),
        "stagnation_locations": roots,
    }
    arrays = {
        **wall,
        "common_sample_points": points,
        "common_sample_velocity": sampled_velocity,
    }
    return diagnostics, arrays


def _velocity_to_mixed_mapping(fs, up) -> np.ndarray:
    v_coordinates = fs.V.tabulate_dof_coordinates()
    w_coordinates = fs.W.tabulate_dof_coordinates()
    mapping = np.zeros(fs.V.dim(), dtype=np.int64)
    for component in (0, 1):
        v_dofs = np.asarray(fs.V.sub(component).dofmap().dofs(), dtype=np.int64)
        w_dofs = np.asarray(
            fs.W.sub(0).sub(component).dofmap().dofs(), dtype=np.int64
        )
        tree = cKDTree(w_coordinates[w_dofs])
        distance, nearest = tree.query(v_coordinates[v_dofs], k=1)
        if float(np.max(distance)) > 1e-12:
            raise RuntimeError("Could not match velocity and mixed-space DOFs")
        mapping[v_dofs] = w_dofs[nearest]
    error = np.max(
        np.abs(up.vector().get_local()[mapping] - fs.fields.U0.vector().get_local())
    )
    if error > 1e-11:
        raise RuntimeError(f"Velocity/mixed DOF mapping error is {error}")
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def save_ground_truth(
    fs,
    run_dir: Path,
    mesh_name: str,
    actuation: float,
    continuation_history: list[dict],
    campaign: str = "mesh_convergence",
    write_steady_checkpoint: bool = True,
) -> dict:
    """Write raw arrays, observations, metadata, and optionally a checkpoint."""

    run_dir.mkdir(parents=True, exist_ok=True)
    if write_steady_checkpoint:
        steady_dir = run_dir / "steady"
        steady_dir.mkdir(exist_ok=True)
        flu.write_xdmf(steady_dir / "U0.xdmf", fs.fields.U0, "U0")
        flu.write_xdmf(steady_dir / "P0.xdmf", fs.fields.P0, "P0")

    up = fs.fields.UP0
    mapping = _velocity_to_mixed_mapping(fs, up)
    raw = {
        "U0_field_data.npy": fs.fields.U0.vector().get_local(),
        "P0_field_data.npy": fs.fields.P0.vector().get_local(),
        "UP0_field_data.npy": up.vector().get_local(),
        "V_dof_coordinates.npy": fs.V.tabulate_dof_coordinates(),
        "P_dof_coordinates.npy": fs.P.tabulate_dof_coordinates(),
        "W_dof_coordinates.npy": fs.W.tabulate_dof_coordinates(),
        "V_to_W_vel_mapping.npy": mapping,
        "vel_dofs_in_mixed.npy": np.asarray(fs.W.sub(0).dofmap().dofs()),
        "pres_dofs_in_mixed.npy": np.asarray(fs.W.sub(1).dofmap().dofs()),
    }
    actuator_bc_v = dolfin.DirichletBC(
        fs.V,
        dolfin.Constant((0.0, 0.0)),
        fs.get_subdomain("actuator"),
    )
    actuator_bc_w = dolfin.DirichletBC(
        fs.W.sub(0),
        dolfin.Constant((0.0, 0.0)),
        fs.get_subdomain("actuator"),
    )
    raw["actuator_dofs_V.npy"] = np.asarray(
        sorted(actuator_bc_v.get_boundary_values()), dtype=np.int64
    )
    raw["actuator_dofs_W.npy"] = np.asarray(
        sorted(actuator_bc_w.get_boundary_values()), dtype=np.int64
    )
    for filename, values in raw.items():
        np.save(run_dir / filename, values)

    diagnostics, arrays = solution_diagnostics(fs, actuation)
    np.save(run_dir / "common_sample_points.npy", arrays.pop("common_sample_points"))
    np.save(
        run_dir / "common_sample_velocity.npy",
        arrays.pop("common_sample_velocity"),
    )
    with (run_dir / "wall_observations.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = list(arrays)
        writer.writerow(names)
        writer.writerows(zip(*(arrays[name] for name in names)))

    np.save(run_dir / "actuation_history.npy", np.array([[0.0, actuation]]))
    np.save(run_dir / "disturbance_history.npy", np.array([[0.0, 0.0]]))
    with (run_dir / "timeseries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "time",
                "actuation",
                "kinetic_energy",
                "inlet_flux_inward",
                "outlet_flux_outward",
                "relative_mass_imbalance",
            ],
        )
        writer.writeheader()
        writer.writerow({"time": 0.0, **{key: diagnostics[key] for key in writer.fieldnames[1:]}})

    case_dir = Path(__file__).resolve().parent
    repo = case_dir.parents[2]
    metadata = case_dir / "data_input" / f"{mesh_name}_metadata.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "backward_facing_step",
        "campaign": campaign,
        "mesh_id": mesh_name,
        "mesh_path": str(fs.params_mesh.meshpath),
        "mesh_metadata_sha256": _sha256(metadata),
        "git": _git_state(repo),
        "reynolds": float(fs.params_flow.Re),
        "uinf_peak": float(fs.params_flow.uinf),
        "step_height": 1.0,
        "dt": float(fs.params_time.dt),
        "save_stride": 0,
        "finite_elements": {"velocity": "CG2", "pressure": "CG1"},
        "field_convention": "total",
        "actuation": float(actuation),
        "actuation_units": "nondimensional volume flux per unit span",
        "actuation_sign": "positive is blowing into the fluid; upper-wall v is negative",
        "actuator_profile": "unit-integral truncated shifted Gaussian",
        "disturbance": "none",
        "random_seed": None,
        "solver": {
            "nonlinear": "Picard/Newton continuation",
            "linear_solver": "MUMPS",
            "newton_max_iterations": 25,
            "newton_tolerances": "FEniCS 2019.1 defaults",
            "picard_max_iterations": 20,
            "picard_tolerance": 1e-8,
            "continuation_history": continuation_history,
        },
        "diagnostics": diagnostics,
        "files": sorted(
            [
                str(path.relative_to(run_dir))
                for path in run_dir.rglob("*")
                if path.is_file()
            ]
            + ["manifest.json"]
        ),
    }
    with (run_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return diagnostics


def compare_run_directories(run_dirs: list[Path]) -> list[dict]:
    """Compare each mesh result to the next finer result at common points."""

    comparisons = []
    for coarse, fine in zip(run_dirs[:-1], run_dirs[1:]):
        p0 = np.load(coarse / "common_sample_points.npy")
        p1 = np.load(fine / "common_sample_points.npy")
        if not np.array_equal(p0, p1):
            raise RuntimeError("Convergence runs use different common sample points")
        u0 = np.load(coarse / "common_sample_velocity.npy")
        u1 = np.load(fine / "common_sample_velocity.npy")
        wall0 = np.genfromtxt(
            coarse / "wall_observations.csv", delimiter=",", names=True
        )
        wall1 = np.genfromtxt(
            fine / "wall_observations.csv", delimiter=",", names=True
        )
        with (coarse / "manifest.json").open() as stream:
            d0 = json.load(stream)["diagnostics"]
        with (fine / "manifest.json").open() as stream:
            d1 = json.load(stream)["diagnostics"]
        relative_velocity = float(np.linalg.norm(u1 - u0) / np.linalg.norm(u1))
        energy_change = abs(d1["kinetic_energy"] - d0["kinetic_energy"]) / abs(
            d1["kinetic_energy"]
        )
        lower0 = d0["stagnation_locations"]["primary_lower_reattachment"]
        lower1 = d1["stagnation_locations"]["primary_lower_reattachment"]
        reattachment_change = None
        if lower0 is not None and lower1 is not None:
            reattachment_change = abs(lower1 - lower0) / abs(lower1)
        lower_wall_shear_change = float(
            np.linalg.norm(
                wall1["lower_skin_friction"] - wall0["lower_skin_friction"]
            )
            / np.linalg.norm(wall1["lower_skin_friction"])
        )
        upper_wall_shear_change = float(
            np.linalg.norm(
                wall1["upper_skin_friction"] - wall0["upper_skin_friction"]
            )
            / np.linalg.norm(wall1["upper_skin_friction"])
        )
        upper0 = d0["stagnation_locations"]["upper_wall_shear_zeros"]
        upper1 = d1["stagnation_locations"]["upper_wall_shear_zeros"]
        upper_stagnation_change = None
        if len(upper0) == len(upper1) and upper1:
            upper_stagnation_change = max(
                abs(x1 - x0) / abs(x1) for x0, x1 in zip(upper0, upper1)
            )
        comparisons.append(
            {
                "coarse": coarse.parent.name,
                "fine": fine.parent.name,
                "relative_common_point_velocity_l2": relative_velocity,
                "relative_kinetic_energy_change": float(energy_change),
                "primary_reattachment_relative_change": reattachment_change,
                "relative_lower_wall_shear_l2": lower_wall_shear_change,
                "relative_upper_wall_shear_l2": upper_wall_shear_change,
                "upper_stagnation_max_relative_change": upper_stagnation_change,
            }
        )
    return comparisons
