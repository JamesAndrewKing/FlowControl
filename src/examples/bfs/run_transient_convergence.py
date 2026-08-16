"""Run the BFS inlet-packet amplitude, mesh, and time-step checks.

The four streamwise inlet modes are based on ``sqrt(2) sin(2*pi*k*eta)``,
``k=1..4``, where ``eta=y-1``.  A negligible parabolic correction removes the
finite-element interpolation's residual mass flux.  The default study reuses
five simulations:

* ``bfs_3`` at ``dt = 0.01, 0.005, 0.0025``;
* ``bfs_4`` at ``dt = 0.005``;
* ``bfs_3`` at ``dt = 0.005`` and half disturbance amplitude.

Use ``--smoke`` for a twenty-step end-to-end check.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np
import scipy.sparse as sparse

from examples.bfs.bfsflowsolver import make_bfs_solver
from flowcontrol.flowfield import BoundaryConditions


LOGGER = logging.getLogger(__name__)
MODE_WEIGHTS = np.array([1.0, -0.65, 0.4, -0.25])
MODE_FREQUENCIES = np.array([0.25, 0.5, 0.75, 1.0])
SENSOR_CENTERS = np.linspace(0.0, 30.0, 61)


def _slug(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p").replace("+", "p")


def _run_id(dt: float, amplitude: float) -> str:
    return f"dt_{_slug(dt)}_delta_{_slug(amplitude)}"


def _git_state(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _make_inlet_expression(fs):
    expression = dolfin.Expression(
        (
            "sqrt_two*(c1*sin(two_pi*(x[1]-y_step)) + "
            "c2*sin(2.0*two_pi*(x[1]-y_step)) + "
            "c3*sin(3.0*two_pi*(x[1]-y_step)) + "
            "c4*sin(4.0*two_pi*(x[1]-y_step))) - "
            "6.0*(x[1]-y_step)*(1.0-(x[1]-y_step))*"
            "(c1*z1+c2*z2+c3*z3+c4*z4)",
            "0.0",
        ),
        element=fs.V.ufl_element(),
        y_step=fs.params_mesh.user_data["y_step"],
        sqrt_two=math.sqrt(2.0),
        two_pi=2.0 * math.pi,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        c4=0.0,
        z1=0.0,
        z2=0.0,
        z3=0.0,
        z4=0.0,
    )
    inlet_bc = dolfin.DirichletBC(
        fs.W.sub(0), expression, fs.get_subdomain("inlet")
    )
    fs.bc = BoundaryConditions(bcu=[inlet_bc] + fs.bc.bcu[1:], bcp=[])
    return expression


def _set_coefficients(expression, coefficients: np.ndarray) -> None:
    for index, value in enumerate(coefficients, start=1):
        setattr(expression, f"c{index}", float(value))


def _packet_coefficients(
    time_value: float,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    if time_value <= 0.0 or time_value >= duration:
        return np.zeros(4)
    phase = time_value / duration
    window = math.sin(math.pi * phase) ** 2
    carrier_time = time_value - 0.5 * duration
    return (
        amplitude
        * MODE_WEIGHTS
        * window
        * np.cos(MODE_FREQUENCIES * carrier_time)
    )


def _make_modes_discretely_solenoidal(
    fs, expression
) -> tuple[list[float], list[float]]:
    markers = dolfin.MeshFunction(
        "size_t", fs.mesh, fs.mesh.topology().dim() - 1, 0
    )
    fs.get_subdomain("inlet").mark(markers, 1)
    inlet_ds = dolfin.Measure("ds", domain=fs.mesh, subdomain_data=markers)
    eta = dolfin.SpatialCoordinate(fs.mesh)[1] - fs.params_mesh.user_data["y_step"]
    bubble_flux = float(dolfin.assemble(6.0 * eta * (1.0 - eta) * inlet_ds(1)))
    corrections = []
    for mode in range(4):
        coefficients = np.zeros(4)
        coefficients[mode] = 1.0
        _set_coefficients(expression, coefficients)
        raw_flux = float(
            dolfin.assemble(
                expression[0] * inlet_ds(1),
                form_compiler_parameters={"quadrature_degree": 12},
            )
        )
        correction = raw_flux / bubble_flux
        setattr(expression, f"z{mode + 1}", correction)
        corrections.append(correction)

    flux_errors = []
    for mode in range(4):
        _set_coefficients(expression, np.eye(4)[mode])
        flux_errors.append(
            abs(float(dolfin.assemble(expression[0] * inlet_ds(1))))
        )
    _set_coefficients(expression, np.zeros(4))
    if max(flux_errors) > 1e-10:
        raise RuntimeError(f"Inlet modes do not have zero flux: {flux_errors}")
    return flux_errors, corrections


def _wall_shear_operator(fs) -> tuple[sparse.csr_matrix, np.ndarray]:
    markers = dolfin.MeshFunction(
        "size_t", fs.mesh, fs.mesh.topology().dim() - 1, 0
    )
    bounds = []
    for index, center in enumerate(SENSOR_CENTERS, start=1):
        left = max(0.0, center - 0.25)
        right = center + 0.25
        sensor = dolfin.CompiledSubDomain(
            "on_boundary && near(x[1], 0.0, tol) && "
            "x[0] >= left-tol && x[0] <= right+tol",
            left=left,
            right=right,
            tol=100.0 * dolfin.DOLFIN_EPS,
        )
        sensor.mark(markers, index)
        bounds.append((left, right))

    ds_sensor = dolfin.Measure("ds", domain=fs.mesh, subdomain_data=markers)
    test = dolfin.TestFunction(fs.V)
    one = dolfin.Constant(1.0)
    rows = []
    for index in range(1, len(SENSOR_CENTERS) + 1):
        length = float(dolfin.assemble(one * ds_sensor(index)))
        if length <= 0.0:
            raise RuntimeError(f"Wall sensor {index} has zero measure")
        vector = dolfin.assemble(test[0].dx(1) * ds_sensor(index)).get_local()
        rows.append(sparse.csr_matrix(vector / length))
    return sparse.vstack(rows, format="csr"), np.asarray(bounds)


def _load_result(run_dir: Path) -> dict:
    with np.load(run_dir / "diagnostics.npz") as data:
        arrays = {name: data[name] for name in data.files}
    with (run_dir / "manifest.json").open() as stream:
        manifest = json.load(stream)
    return {"run_dir": run_dir, "manifest": manifest, **arrays}


def _run_case(
    output_root: Path,
    base_root: Path,
    mesh_name: str,
    dt: float,
    amplitude: float,
    final_time: float,
    packet_duration: float,
    diagnostic_dt: float,
    field_save_dt: float,
    reuse_existing: bool,
) -> dict:
    run_dir = (
        output_root
        / "transient_convergence"
        / mesh_name
        / _run_id(dt, amplitude)
    )
    if reuse_existing and (run_dir / "diagnostics.npz").exists():
        LOGGER.info("Reusing %s", run_dir)
        return _load_result(run_dir)

    steps = int(round(final_time / dt))
    if not math.isclose(steps * dt, final_time, abs_tol=1e-12):
        raise ValueError("Final time must be an integer multiple of dt")
    diagnostic_stride = max(1, int(round(diagnostic_dt / dt)))
    if not math.isclose(diagnostic_stride * dt, diagnostic_dt, abs_tol=1e-12):
        raise ValueError("Diagnostic interval must be an integer multiple of dt")
    save_every = 0
    if field_save_dt > 0.0:
        save_every = int(round(field_save_dt / dt))
        if not math.isclose(save_every * dt, field_save_dt, abs_tol=1e-12):
            raise ValueError("Field-save interval must be an integer multiple of dt")

    run_dir.mkdir(parents=True, exist_ok=True)
    fs = make_bfs_solver(
        mesh_name=mesh_name,
        reynolds=500.0,
        output_dir=run_dir,
        save_every=save_every,
        verbose=0,
    )
    fs.params_time.dt = dt
    fs.params_time.num_steps = steps
    fs.params_time.Tfinal = final_time
    base_dir = (
        base_root
        / "mesh_convergence"
        / mesh_name
        / "a_p0p00000"
        / "steady"
    )
    fs.load_steady_state([base_dir / "U0.xdmf", base_dir / "P0.xdmf"])
    inlet_expression = _make_inlet_expression(fs)
    mode_flux_errors, mode_flux_corrections = _make_modes_discretely_solenoidal(
        fs, inlet_expression
    )
    shear_operator, sensor_bounds = _wall_shear_operator(fs)
    base_shear = np.asarray(
        shear_operator @ fs.fields.U0.vector().get_local()
    ).ravel()

    fs.initialize_time_stepping(ic=None)
    times = np.arange(steps + 1, dtype=float) * dt
    coefficients = np.zeros((steps + 1, 4))
    energy = np.zeros(steps + 1)
    wall_times = [0.0]
    wall_perturbation = [np.zeros(len(SENSOR_CENTERS))]
    started = time.perf_counter()
    for step in range(1, steps + 1):
        coefficients[step] = _packet_coefficients(
            times[step], amplitude, packet_duration
        )
        _set_coefficients(inlet_expression, coefficients[step])
        fs.step(u_ctrl=[0.0])
        energy[step] = float(fs.timeseries.loc[fs.iter, "dE"])
        if step % diagnostic_stride == 0 or step == steps:
            wall_times.append(times[step])
            wall_perturbation.append(
                np.asarray(
                    shear_operator @ fs.fields.u_.vector().get_local()
                ).ravel()
            )
        if step % max(1, steps // 20) == 0:
            LOGGER.info(
                "%s dt=%g: %d/%d, E=%.4e",
                mesh_name,
                dt,
                step,
                steps,
                energy[step],
            )
    runtime = time.perf_counter() - started
    fs.write_timeseries()

    wall_times_array = np.asarray(wall_times)
    wall_perturbation_array = np.asarray(wall_perturbation)
    np.savez_compressed(
        run_dir / "diagnostics.npz",
        time=times,
        energy=energy,
        disturbance_coefficients=coefficients,
        wall_time=wall_times_array,
        wall_perturbation=wall_perturbation_array,
        wall_total=wall_perturbation_array + base_shear,
        wall_sensor_centers=SENSOR_CENTERS,
        wall_sensor_bounds=sensor_bounds,
        base_wall_shear=base_shear,
    )

    case_dir = Path(__file__).resolve().parent
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "backward_facing_step",
        "campaign": "transient_convergence",
        "mesh_id": mesh_name,
        "reynolds": 500.0,
        "base_actuation": 0.0,
        "field_convention": "total fields in XDMF; perturbation diagnostics in NPZ",
        "dt": dt,
        "final_time": final_time,
        "steps": steps,
        "field_save_stride": save_every,
        "diagnostic_stride": diagnostic_stride,
        "disturbance_amplitude": amplitude,
        "disturbance_packet": {
            "duration": packet_duration,
            "window": "sin(pi*t/duration)^2 on the compact interval",
            "modal_temporal_frequencies": MODE_FREQUENCIES.tolist(),
            "modal_weights": MODE_WEIGHTS.tolist(),
        },
        "inlet_modes": {
            "count": 4,
            "formula": "sqrt(2)*sin(2*pi*k*(y-1)), k=1,...,4",
            "component": "streamwise velocity",
            "parabolic_zero_flux_corrections": mode_flux_corrections,
            "zero_flux_errors": mode_flux_errors,
        },
        "wall_sensors": {
            "count": len(SENSOR_CENTERS),
            "centers": SENSOR_CENTERS.tolist(),
            "quantity": "segment-averaged du/dy on downstream lower wall",
        },
        "runtime_seconds": runtime,
        "peak_energy": float(np.max(energy)),
        "peak_time": float(times[int(np.argmax(energy))]),
        "git": _git_state(case_dir.parents[2]),
    }
    manifest["files"] = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    manifest["files"].append("manifest.json")
    with (run_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return _load_result(run_dir)


def _compare(reference: dict, refined: dict) -> dict:
    refined_energy = np.interp(
        reference["time"], refined["time"], refined["energy"]
    )
    energy_error = float(
        np.linalg.norm(reference["energy"] - refined_energy)
        / max(np.linalg.norm(refined_energy), 1e-30)
    )
    refined_wall = np.column_stack(
        [
            np.interp(
                reference["wall_time"],
                refined["wall_time"],
                refined["wall_perturbation"][:, sensor],
            )
            for sensor in range(len(SENSOR_CENTERS))
        ]
    )
    wall_error = float(
        np.linalg.norm(reference["wall_perturbation"] - refined_wall)
        / max(np.linalg.norm(refined_wall), 1e-30)
    )
    peak_reference = float(np.max(reference["energy"]))
    peak_refined = float(np.max(refined["energy"]))
    peak_error = abs(peak_reference - peak_refined) / max(peak_refined, 1e-30)
    peak_time_reference = float(
        reference["time"][int(np.argmax(reference["energy"]))]
    )
    peak_time_refined = float(
        refined["time"][int(np.argmax(refined["energy"]))]
    )
    return {
        "reference_run": str(reference["run_dir"]),
        "refined_run": str(refined["run_dir"]),
        "relative_energy_history_l2": energy_error,
        "relative_peak_energy_change": float(peak_error),
        "peak_time_change": abs(peak_time_reference - peak_time_refined),
        "relative_wall_shear_history_l2": wall_error,
        "passes": energy_error < 0.01
        and peak_error < 0.01
        and wall_error < 0.005,
    }


def _linearity(full: dict, half: dict) -> dict:
    half_energy = np.interp(full["time"], half["time"], half["energy"])
    quadratic_error = float(
        np.linalg.norm(full["energy"] - 4.0 * half_energy)
        / max(np.linalg.norm(full["energy"]), 1e-30)
    )
    peak_ratio = float(np.max(full["energy"]) / max(np.max(half_energy), 1e-30))
    return {
        "full_run": str(full["run_dir"]),
        "half_run": str(half["run_dir"]),
        "peak_energy_ratio": peak_ratio,
        "relative_quadratic_scaling_error": quadratic_error,
        "passes": quadratic_error < 0.01 and abs(peak_ratio - 4.0) < 0.04,
    }


def run_study(args) -> dict:
    output_root = Path(args.output_root).expanduser().resolve()
    base_root = Path(args.base_root or args.output_root).expanduser().resolve()
    if args.smoke:
        dt = args.dts[0]
        result = _run_case(
            output_root,
            base_root,
            "bfs_3",
            dt,
            args.amplitude,
            20.0 * dt,
            args.packet_duration,
            dt,
            0.0,
            False,
        )
        return {"smoke_run": str(result["run_dir"])}

    configurations = [
        ("bfs_3", args.dts[0], args.amplitude),
        ("bfs_3", args.dts[1], args.amplitude),
        ("bfs_3", args.dts[2], args.amplitude),
        ("bfs_4", args.dts[1], args.amplitude),
        ("bfs_3", args.dts[1], 0.5 * args.amplitude),
    ]
    results = {}
    for mesh_name, dt, amplitude in configurations:
        key = (mesh_name, dt, amplitude)
        results[key] = _run_case(
            output_root,
            base_root,
            mesh_name,
            dt,
            amplitude,
            args.final_time,
            args.packet_duration,
            args.diagnostic_dt,
            args.field_save_dt,
            args.reuse_existing,
        )

    coarse = results[("bfs_3", args.dts[0], args.amplitude)]
    reference = results[("bfs_3", args.dts[1], args.amplitude)]
    fine = results[("bfs_3", args.dts[2], args.amplitude)]
    verification = results[("bfs_4", args.dts[1], args.amplitude)]
    half = results[("bfs_3", args.dts[1], 0.5 * args.amplitude)]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "linearity": _linearity(reference, half),
        "time_step": {
            f"{args.dts[0]}_to_{args.dts[1]}": _compare(coarse, reference),
            f"{args.dts[1]}_to_{args.dts[2]}": _compare(reference, fine),
        },
        "spatial_bfs_3_to_bfs_4": _compare(reference, verification),
    }
    summary["passes"] = (
        summary["linearity"]["passes"]
        and all(item["passes"] for item in summary["time_step"].values())
        and summary["spatial_bfs_3_to_bfs_4"]["passes"]
    )
    summary_path = output_root / "transient_convergence" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    parser.add_argument(
        "--output-root",
        required=default_output is None,
        default=default_output,
    )
    parser.add_argument(
        "--base-root",
        help="Root containing mesh_convergence checkpoints (defaults to output root)",
    )
    parser.add_argument(
        "--dts", nargs=3, type=float, default=[0.01, 0.005, 0.0025]
    )
    parser.add_argument("--amplitude", type=float, default=1e-4)
    parser.add_argument("--final-time", type=float, default=80.0)
    parser.add_argument("--packet-duration", type=float, default=4.0)
    parser.add_argument("--diagnostic-dt", type=float, default=0.1)
    parser.add_argument("--field-save-dt", type=float, default=5.0)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    for noisy_logger in ("FFC", "UFL", "dijitso", "utils.utils_flowsolver"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    print(json.dumps(run_study(parse_args()), indent=2, sort_keys=True))
