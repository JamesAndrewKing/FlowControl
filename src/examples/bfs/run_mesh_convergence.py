"""Compute steady BFS states and quantitative mesh-convergence diagnostics.

From the repository root, run for example::

    PYTHONPATH=src python src/examples/bfs/run_mesh_convergence.py \
        --output-root ../AdiabaticFlowControl/data/bfs
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import dolfin
import numpy as np

from examples.bfs.bfsdiagnostics import (
    boundary_measures,
    compare_run_directories,
    prescribed_fluxes,
    save_ground_truth,
)
from examples.bfs.bfsflowsolver import make_bfs_solver


LOGGER = logging.getLogger(__name__)
DEFAULT_REYNOLDS_PATH = (50.0, 100.0, 200.0, 300.0, 400.0, 500.0)
EXPECTED_BOUNDARY_LENGTHS = {
    "inlet": 1.0,
    "outlet": 2.0,
    "upper_wall": 59.2,
    "actuator": 0.8,
    "upstream_lower_wall": 10.0,
    "step_wall": 1.0,
    "downstream_lower_wall": 50.0,
}


def _actuation_slug(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def _validate_setup(fs) -> dict:
    measures = boundary_measures(fs)
    errors = {
        name: abs(measures[name] - expected)
        for name, expected in EXPECTED_BOUNDARY_LENGTHS.items()
    }
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"Incorrect geometric boundary measures: {errors}")
    fluxes = {}
    for actuation in (-0.01, 0.0, 0.01):
        values = prescribed_fluxes(fs, actuation)
        fluxes[str(actuation)] = values
        if abs(values["inlet_inward"] - 2.0 / 3.0) > 1e-10:
            raise RuntimeError(f"Inlet flux validation failed: {values}")
        if abs(values["actuator_inward"] - actuation) > 5e-5:
            raise RuntimeError(f"Actuator flux validation failed: {values}")
    fs.set_actuators_u_ctrl([0.0])
    return {"boundary_measures": measures, "prescribed_fluxes": fluxes}


def _newton(fs, actuation: float, initial_guess, max_iter: int = 25):
    fs.compute_steady_state(
        method="newton",
        max_iter=max_iter,
        u_ctrl=[actuation],
        initial_guess=initial_guess,
    )
    return fs.fields.UP0


def _solve_reynolds_path(fs, reynolds_path: tuple[float, ...]) -> list[dict]:
    history = []
    initial_guess = None
    for index, reynolds in enumerate(reynolds_path):
        fs.params_flow.Re = reynolds
        if index == 0:
            fs.compute_steady_state(
                method="picard",
                max_iter=15,
                tol=1e-8,
                u_ctrl=[0.0],
                initial_guess=initial_guess,
            )
            initial_guess = fs.fields.UP0
            history.append(
                {"parameter": "Re", "value": reynolds, "method": "Picard"}
            )
        try:
            initial_guess = _newton(fs, 0.0, initial_guess)
            method = "Newton"
        except RuntimeError:
            LOGGER.warning("Newton failed at Re=%g; applying Picard fallback", reynolds)
            fs.compute_steady_state(
                method="picard",
                max_iter=20,
                tol=1e-8,
                u_ctrl=[0.0],
                initial_guess=initial_guess,
            )
            initial_guess = _newton(fs, 0.0, fs.fields.UP0)
            method = "Picard+Newton fallback"
        history.append({"parameter": "Re", "value": reynolds, "method": method})
    return history


def _solve_actuation_path(
    fs,
    target: float,
    zero_state,
    subdivisions: int,
) -> list[dict]:
    history = []
    initial_guess = zero_state
    for actuation in np.linspace(0.0, target, subdivisions + 1)[1:]:
        try:
            initial_guess = _newton(fs, float(actuation), initial_guess)
            method = "Newton"
        except RuntimeError:
            fs.compute_steady_state(
                method="picard",
                max_iter=20,
                tol=1e-8,
                u_ctrl=[float(actuation)],
                initial_guess=initial_guess,
            )
            initial_guess = _newton(fs, float(actuation), fs.fields.UP0)
            method = "Picard+Newton fallback"
        history.append(
            {"parameter": "actuation", "value": float(actuation), "method": method}
        )
    return history


def _solve_from_mesh_checkpoint(
    fs,
    campaign_root: Path,
    source_mesh: str,
) -> list[dict]:
    """Interpolate a converged zero-control state onto a different mesh."""

    source_root = campaign_root / source_mesh
    source_run = source_root / _actuation_slug(0.0)
    with (source_run / "manifest.json").open() as stream:
        source_manifest = json.load(stream)
    if float(source_manifest["reynolds"]) != float(fs.params_flow.Re):
        raise ValueError("Source and target Reynolds numbers must match")

    source = make_bfs_solver(
        mesh_name=source_mesh,
        reynolds=fs.params_flow.Re,
        output_dir=source_root / "_interpolation_scratch",
        save_every=0,
        verbose=0,
    )
    source.load_steady_state(
        [source_run / "steady" / "U0.xdmf", source_run / "steady" / "P0.xdmf"]
    )
    velocity = dolfin.Function(fs.V)
    pressure = dolfin.Function(fs.P)
    dolfin.LagrangeInterpolator.interpolate(velocity, source.fields.U0)
    dolfin.LagrangeInterpolator.interpolate(pressure, source.fields.P0)
    initial_guess = fs.merge(velocity, pressure)
    del source, velocity, pressure
    gc.collect()

    try:
        _newton(fs, 0.0, initial_guess)
        method = "nonmatching interpolation + Newton"
    except RuntimeError:
        LOGGER.warning("Interpolated Newton solve failed; applying Picard fallback")
        fs.compute_steady_state(
            method="picard",
            max_iter=20,
            tol=1e-8,
            u_ctrl=[0.0],
            initial_guess=initial_guess,
        )
        _newton(fs, 0.0, fs.fields.UP0)
        method = "nonmatching interpolation + Picard+Newton fallback"
    return [
        {
            "parameter": "mesh_restart",
            "value": source_mesh,
            "method": method,
        }
    ]


def run(args) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    campaign_root = output_root / "mesh_convergence"
    campaign_root.mkdir(parents=True, exist_ok=True)
    for mesh_name in args.meshes:
        mesh_root = campaign_root / mesh_name
        scratch = mesh_root / "_solver_scratch"
        use_checkpoint = args.reuse_zero or args.restart_from_mesh is not None
        initial_reynolds = (
            args.reynolds_path[-1] if use_checkpoint else args.reynolds_path[0]
        )
        fs = make_bfs_solver(
            mesh_name=mesh_name,
            reynolds=initial_reynolds,
            output_dir=scratch,
            save_every=0,
            verbose=args.verbose,
        )
        setup = _validate_setup(fs)
        with (mesh_root / "setup_validation.json").open("w") as stream:
            json.dump(setup, stream, indent=2, sort_keys=True)

        if args.restart_from_mesh is not None:
            history_re = _solve_from_mesh_checkpoint(
                fs, campaign_root, args.restart_from_mesh
            )
        elif args.reuse_zero:
            zero_checkpoint = mesh_root / _actuation_slug(0.0) / "steady"
            fs.load_steady_state(
                [zero_checkpoint / "U0.xdmf", zero_checkpoint / "P0.xdmf"]
            )
            history_re = [
                {
                    "parameter": "restart",
                    "value": 0.0,
                    "method": "loaded converged zero-actuation checkpoint",
                }
            ]
        else:
            history_re = _solve_reynolds_path(fs, tuple(args.reynolds_path))
        zero_state = dolfin.Function(fs.W)
        zero_state.assign(fs.fields.UP0)

        ordered_actuations = [0.0] + [a for a in args.actuations if a != 0.0]
        for actuation in ordered_actuations:
            if actuation not in args.actuations:
                continue
            if actuation == 0.0:
                history = history_re
            else:
                fs.params_flow.Re = args.reynolds_path[-1]
                history = history_re + _solve_actuation_path(
                    fs,
                    actuation,
                    zero_state,
                    args.actuation_subdivisions,
                )
            run_dir = mesh_root / _actuation_slug(actuation)
            diagnostics = save_ground_truth(
                fs,
                run_dir=run_dir,
                mesh_name=mesh_name,
                actuation=actuation,
                continuation_history=history,
            )
            LOGGER.info(
                "%s a=%+.5f: mass error %.3e, lower roots %s",
                mesh_name,
                actuation,
                diagnostics["relative_mass_imbalance"],
                diagnostics["stagnation_locations"]["lower_wall_shear_zeros"],
            )
        del fs
        gc.collect()

    summary_meshes = args.summary_meshes or args.meshes
    discovered_runs: dict[str, list[Path]] = {}
    for mesh_name in summary_meshes:
        for manifest_path in (campaign_root / mesh_name).glob("a_*/manifest.json"):
            with manifest_path.open() as stream:
                actuation = str(json.load(stream)["actuation"])
            discovered_runs.setdefault(actuation, []).append(manifest_path.parent)

    summary = {
        "acceptance_threshold_relative": args.acceptance_threshold,
        "comparisons": {},
    }
    for actuation, run_dirs in discovered_runs.items():
        mesh_order = {name: index for index, name in enumerate(summary_meshes)}
        run_dirs.sort(key=lambda path: mesh_order[path.parent.name])
        if len(run_dirs) != len(summary_meshes):
            continue
        comparisons = compare_run_directories(run_dirs)
        for comparison in comparisons:
            bulk_metrics = [
                comparison["relative_common_point_velocity_l2"],
                comparison["relative_kinetic_energy_change"],
            ]
            if comparison["primary_reattachment_relative_change"] is not None:
                bulk_metrics.append(
                    comparison["primary_reattachment_relative_change"]
                )
            if comparison["upper_stagnation_max_relative_change"] is not None:
                bulk_metrics.append(comparison["upper_stagnation_max_relative_change"])
            wall_metrics = [
                comparison["relative_lower_wall_shear_l2"],
                comparison["relative_upper_wall_shear_l2"],
            ]
            comparison["passes_bulk_threshold"] = (
                max(bulk_metrics) < args.acceptance_threshold
            )
            comparison["passes_wall_threshold"] = (
                max(wall_metrics) < args.acceptance_threshold
            )
            comparison["passes_threshold"] = (
                comparison["passes_bulk_threshold"]
                and comparison["passes_wall_threshold"]
            )
        summary["comparisons"][actuation] = comparisons
    with (campaign_root / "mesh_convergence_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    parser.add_argument(
        "--output-root",
        required=default_output is None,
        default=default_output,
        help="BFS data root (normally AdiabaticFlowControl/data/bfs)",
    )
    parser.add_argument(
        "--meshes", nargs="+", default=["bfs_1", "bfs_2", "bfs_3"]
    )
    parser.add_argument("--actuations", nargs="+", type=float, default=[0.0])
    parser.add_argument(
        "--reynolds-path",
        nargs="+",
        type=float,
        default=list(DEFAULT_REYNOLDS_PATH),
    )
    parser.add_argument("--actuation-subdivisions", type=int, default=4)
    parser.add_argument(
        "--reuse-zero",
        action="store_true",
        help="Load each existing zero-actuation checkpoint instead of recomputing it",
    )
    parser.add_argument(
        "--restart-from-mesh",
        help="Interpolate an existing zero-control checkpoint from this mesh",
    )
    parser.add_argument(
        "--summary-meshes",
        nargs="+",
        help="Ordered mesh list used to rebuild the convergence summary",
    )
    parser.add_argument("--acceptance-threshold", type=float, default=0.005)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    if args.reuse_zero and args.restart_from_mesh is not None:
        parser.error("--reuse-zero and --restart-from-mesh are mutually exclusive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    for noisy_logger in ("FFC", "UFL", "dijitso", "utils.utils_flowsolver"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    run(parse_args())
