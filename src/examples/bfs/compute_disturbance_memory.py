"""Measure the frozen BFS response memory of a small inlet packet.

The fixed points are loaded from FlowControl.  Diagnostics are written locally
unless an external research-data root is supplied.  The reported memory time
is the last crossing of a fixed fraction of the peak perturbation energy.
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np

from examples.bfs.bfsflowsolver import make_bfs_solver
from examples.bfs.run_transient_convergence import (
    MODE_FREQUENCIES,
    MODE_WEIGHTS,
    _git_state,
    _make_inlet_expression,
    _make_modes_discretely_solenoidal,
    _packet_coefficients,
    _set_coefficients,
)


LOGGER = logging.getLogger(__name__)


def actuation_slug(value):
    sign = "p" if value >= 0.0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def memory_statistics(times, energy, tail_fraction, packet_duration):
    peak_index = int(np.argmax(energy))
    peak_energy = float(energy[peak_index])
    if peak_energy <= np.finfo(float).tiny:
        raise RuntimeError("The inlet packet produced zero perturbation energy")

    threshold = tail_fraction * peak_energy
    above = np.flatnonzero(energy >= threshold)
    last_index = int(above[-1])
    censored = last_index == len(times) - 1
    if censored:
        crossing_time = float(times[-1])
    else:
        time_0, time_1 = times[last_index : last_index + 2]
        energy_0, energy_1 = energy[last_index : last_index + 2]
        crossing_time = float(
            time_0
            + (threshold - energy_0) * (time_1 - time_0)
            / (energy_1 - energy_0)
        )

    return {
        "peak_energy": peak_energy,
        "peak_time": float(times[peak_index]),
        "tail_fraction": float(tail_fraction),
        "tail_threshold": float(threshold),
        "memory_time": crossing_time,
        "memory_after_packet": max(0.0, crossing_time - packet_duration),
        "horizon_censored": bool(censored),
    }


def run_case(args, actuation, output_root, local_root):
    case = actuation_slug(actuation)
    run_dir = output_root / case
    diagnostics_path = run_dir / "diagnostics.npz"
    manifest_path = run_dir / "manifest.json"
    if args.reuse_existing and diagnostics_path.exists() and manifest_path.exists():
        with manifest_path.open() as stream:
            existing = json.load(stream)
        requested = (
            ("base_actuation", actuation),
            ("dt", args.dt),
            ("final_time", args.final_time),
            ("disturbance_amplitude", args.amplitude),
            ("reynolds", args.reynolds),
        )
        if all(
            math.isclose(existing[name], value, abs_tol=1e-12)
            for name, value in requested
        ) and math.isclose(
            existing["disturbance_packet"]["duration"],
            args.packet_duration,
            abs_tol=1e-12,
        ) and math.isclose(
            existing["memory"]["tail_fraction"],
            args.tail_fraction,
            abs_tol=1e-12,
        ):
            return existing
        LOGGER.info("Existing data in %s use different parameters; rerunning", run_dir)

    fixed_point = local_root / "fixed_points" / case
    velocity_file = fixed_point / "U0.xdmf"
    pressure_file = fixed_point / "P0.xdmf"
    if not velocity_file.exists() or not pressure_file.exists():
        raise FileNotFoundError(f"Missing local fixed point in {fixed_point}")

    steps = int(round(args.final_time / args.dt))
    if not math.isclose(steps * args.dt, args.final_time, abs_tol=1e-12):
        raise ValueError("--final-time must be an integer multiple of --dt")

    run_dir.mkdir(parents=True, exist_ok=True)
    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=run_dir,
        num_steps=steps,
        dt=args.dt,
        save_every=0,
        verbose=0,
    )
    fs.load_steady_state([velocity_file, pressure_file])
    inlet = _make_inlet_expression(fs)
    flux_errors, flux_corrections = _make_modes_discretely_solenoidal(fs, inlet)
    fs.initialize_time_stepping(ic=None)

    times = np.arange(steps + 1, dtype=float) * args.dt
    energy = np.zeros(steps + 1)
    coefficients = np.zeros((steps + 1, len(MODE_WEIGHTS)))
    progress_stride = max(1, min(steps // 20, 100))
    started = time.perf_counter()
    for step in range(1, steps + 1):
        coefficients[step] = _packet_coefficients(
            times[step], args.amplitude, args.packet_duration
        )
        _set_coefficients(inlet, coefficients[step])
        fs.step(u_ctrl=[0.0])
        energy[step] = float(fs.timeseries.loc[fs.iter, "dE"])
        if step % progress_stride == 0 or step == steps:
            LOGGER.info(
                "a=%+.5f: %d/%d, t=%.2f, E=%.4e",
                actuation,
                step,
                steps,
                times[step],
                energy[step],
            )
    runtime = time.perf_counter() - started
    fs.write_timeseries()
    timeseries_file = Path(fs.paths["timeseries"]).name

    memory = memory_statistics(
        times, energy, args.tail_fraction, args.packet_duration
    )
    np.savez_compressed(
        diagnostics_path,
        time=times,
        perturbation_energy=energy,
        disturbance_coefficients=coefficients,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "backward_facing_step",
        "campaign": "disturbance_memory",
        "mesh_id": args.mesh,
        "reynolds": float(args.reynolds),
        "base_actuation": float(actuation),
        "fixed_point": str(fixed_point),
        "dt": float(args.dt),
        "final_time": float(args.final_time),
        "steps": steps,
        "disturbance_amplitude": float(args.amplitude),
        "disturbance_packet": {
            "duration": float(args.packet_duration),
            "window": "sin(pi*t/duration)^2 on the compact interval",
            "modal_temporal_frequencies": MODE_FREQUENCIES.tolist(),
            "modal_weights": MODE_WEIGHTS.tolist(),
        },
        "inlet_modes": {
            "count": len(MODE_WEIGHTS),
            "formula": "sqrt(2)*sin(2*pi*k*(y-1)), k=1,...,4",
            "component": "streamwise velocity",
            "parabolic_zero_flux_corrections": flux_corrections,
            "zero_flux_errors": flux_errors,
        },
        "memory": memory,
        "runtime_seconds": runtime,
        "git": _git_state(Path(__file__).resolve().parents[3]),
        "files": ["diagnostics.npz", "manifest.json", timeseries_file],
    }
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest


def write_summary(output_root, manifests):
    rows = []
    for manifest in manifests:
        rows.append(
            {
                "actuation": manifest["base_actuation"],
                **manifest["memory"],
            }
        )
    with (output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    censored = any(row["horizon_censored"] for row in rows)
    lower_bound = max(row["memory_time"] for row in rows)
    memory_time = None if censored else lower_bound
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "last time perturbation energy crosses "
            f"{100.0 * rows[0]['tail_fraction']:g}% of its peak"
        ),
        "cases": rows,
        "horizon_censored": censored,
        "T_m": memory_time,
        "T_m_lower_bound": lower_bound,
        "epsilon_a": [0.05, 0.1, 0.2],
        "omega_a_max": (
            None
            if memory_time is None
            else [value / memory_time for value in (0.05, 0.1, 0.2)]
        ),
    }
    with (output_root / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


def main(args):
    example_dir = Path(__file__).resolve().parent
    local_root = example_dir / "data_output" / args.mesh
    if args.smoke:
        args.actuations = [0.0]
        args.final_time = 20.0 * args.dt
        args.packet_duration = 5.0 * args.dt
        output_root = local_root / "_memory_smoke"
    elif args.output_root is None:
        output_root = local_root / "disturbance_memory"
    else:
        output_root = (
            Path(args.output_root).expanduser().resolve()
            / "disturbance_memory"
            / args.mesh
        )
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = [
        run_case(args, actuation, output_root, local_root)
        for actuation in args.actuations
    ]
    summary = write_summary(output_root, manifests)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=os.environ.get("ADIABATIC_BFS_DATA_ROOT"),
        help="optional external research-data root",
    )
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument(
        "--actuations", nargs="+", type=float, default=[-0.01, 0.0, 0.01]
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=250.0)
    parser.add_argument("--packet-duration", type=float, default=4.0)
    parser.add_argument("--amplitude", type=float, default=1e-4)
    parser.add_argument("--tail-fraction", type=float, default=0.05)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dt <= 0.0 or args.final_time <= 0.0:
        parser.error("--dt and --final-time must be positive")
    if args.packet_duration <= 0.0 or args.packet_duration >= args.final_time:
        parser.error("--packet-duration must lie between zero and --final-time")
    if args.amplitude <= 0.0:
        parser.error("--amplitude must be positive")
    if not 0.0 < args.tail_fraction < 1.0:
        parser.error("--tail-fraction must lie between zero and one")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for noisy_logger in ("FFC", "UFL", "dijitso", "utils.utils_flowsolver"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
