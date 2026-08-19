"""Generate one disturbance-free, slowly actuated BFS trajectory.

The run starts from the zero-actuation fixed point.  A bounded multisine is
applied as the boundary perturbation, so the total actuator amplitude is the
prescribed ``a(t)``.  Exact first and second derivatives are saved together
with dense lower-wall observations and optional total-field checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np

import utils.utils_flowsolver as flu
from examples.bfs.bfsflowsolver import make_bfs_solver
from examples.bfs.run_transient_convergence import (
    _git_state,
    _wall_shear_operator,
)


LOGGER = logging.getLogger(__name__)
FREQUENCY_RATIOS = np.array([1.0, 0.83, 0.67, 0.49])
MODE_WEIGHTS = np.array([1.0, 1.0, 0.7, 0.7])


def value_slug(value):
    return f"{value:.5g}".replace("-", "m").replace(".", "p")


def actuation_slug(value):
    sign = "p" if value >= 0.0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def actuation_from_slug(name):
    token = name.removeprefix("a_")
    sign = 1.0 if token[0] == "p" else -1.0
    return sign * float(token[1:].replace("p", "."))


def make_trajectory(times, epsilon, memory_time, amplitude, seed):
    """Return a zero-mean multisine with ``a(0)=0`` and exact derivatives."""

    rng = np.random.default_rng(seed)
    phase_1, phase_2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
    phases = np.array([phase_1, -phase_1, phase_2, -phase_2])
    omega_max = epsilon / memory_time
    frequencies = omega_max * FREQUENCY_RATIOS
    angles = times[:, None] * frequencies + phases

    raw = np.einsum("ij,j->i", np.sin(angles), MODE_WEIGHTS)
    raw_dot = np.einsum(
        "ij,j->i", np.cos(angles), MODE_WEIGHTS * frequencies
    )
    raw_ddot = np.einsum(
        "ij,j->i", -np.sin(angles), MODE_WEIGHTS * frequencies**2
    )
    if abs(raw[0]) > 1e-12:
        raise RuntimeError("Paired multisine phases did not give a(0)=0")
    scale = amplitude / np.max(np.abs(raw))
    return {
        "actuation": scale * raw,
        "actuation_rate": scale * raw_dot,
        "actuation_acceleration": scale * raw_ddot,
        "omega_max": omega_max,
        "frequencies": frequencies,
        "phases": phases,
        "scale": scale,
    }


def critical_wall_table(fs, fixed_point_root, wall_operator):
    branch = []
    for path in fixed_point_root.glob("a_[mp]*"):
        if (path / "U0.xdmf").exists() and (path / "P0.xdmf").exists():
            branch.append((actuation_from_slug(path.name), path))
    branch.sort()
    if len(branch) < 3:
        raise FileNotFoundError(f"No fixed-point branch under {fixed_point_root}")

    actuation = []
    wall_shear = []
    for value, path in branch:
        fs.load_steady_state([path / "U0.xdmf", path / "P0.xdmf"])
        actuation.append(value)
        wall_shear.append(
            np.asarray(wall_operator @ fs.fields.U0.vector().get_local()).ravel()
        )
    return np.asarray(actuation), np.asarray(wall_shear)


def stride_steps(final_step, stride):
    steps = np.arange(0, final_step + 1, stride, dtype=int)
    if steps[-1] != final_step:
        steps = np.append(steps, final_step)
    return steps


def write_total_fields(fs, run_dir, time_value, append):
    velocity = dolfin.Function(fs.V)
    pressure = dolfin.Function(fs.P)
    velocity.vector()[:] = (
        fs.fields.U0.vector()[:] + fs.fields.u_.vector()[:]
    )
    pressure.vector()[:] = (
        fs.fields.P0.vector()[:] + fs.fields.p_.vector()[:]
    )
    velocity.vector().apply("insert")
    pressure.vector().apply("insert")
    flu.write_xdmf(
        run_dir / "U_total.xdmf",
        velocity,
        "U",
        time_step=time_value,
        append=append,
        write_mesh=not append,
    )
    flu.write_xdmf(
        run_dir / "P_total.xdmf",
        pressure,
        "P",
        time_step=time_value,
        append=append,
        write_mesh=not append,
    )


def output_directory(args, example_dir):
    local_root = example_dir / "data_output" / args.mesh
    run_id = (
        f"{args.split}_seed_{args.seed:03d}_eps_{value_slug(args.epsilon)}"
    )
    if args.smoke:
        return local_root, local_root / "_adiabatic_smoke" / run_id, run_id
    if args.output_root is None:
        return local_root, local_root / "adiabatic" / run_id, run_id
    research_root = Path(args.output_root).expanduser().resolve()
    return local_root, research_root / "adiabatic" / args.mesh / run_id, run_id


def main(args):
    example_dir = Path(__file__).resolve().parent
    local_root, run_dir, run_id = output_directory(args, example_dir)
    if (run_dir / "manifest.json").exists() and not (args.overwrite or args.smoke):
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    omega_max = args.epsilon / args.memory_time
    duration = args.cycles * 2.0 * np.pi / omega_max
    if args.smoke:
        duration = 20.0 * args.dt
        args.actuation_limit = min(args.actuation_limit, 1e-4)
        args.observation_dt = args.dt
        args.field_dt = duration
    steps = int(round(duration / args.dt))
    duration = steps * args.dt
    times = np.arange(steps + 1, dtype=float) * args.dt
    trajectory_epsilon = args.epsilon
    if args.smoke:
        trajectory_epsilon = 2.0 * np.pi * args.memory_time / duration
    trajectory = make_trajectory(
        times,
        trajectory_epsilon,
        args.memory_time,
        args.actuation_limit,
        args.seed,
    )

    observation_stride = int(round(args.observation_dt / args.dt))
    if observation_stride < 1 or not math.isclose(
        observation_stride * args.dt, args.observation_dt, abs_tol=1e-12
    ):
        raise ValueError("--observation-dt must be an integer multiple of --dt")
    observation_steps = stride_steps(steps, observation_stride)
    observation_index = {step: index for index, step in enumerate(observation_steps)}

    field_steps = np.array([], dtype=int)
    if args.field_dt > 0.0:
        field_stride = int(round(args.field_dt / args.dt))
        if field_stride < 1 or not math.isclose(
            field_stride * args.dt, args.field_dt, abs_tol=1e-12
        ):
            raise ValueError("--field-dt must be an integer multiple of --dt")
        field_steps = stride_steps(steps, field_stride)
    field_step_set = set(field_steps.tolist())

    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=run_dir,
        num_steps=steps,
        dt=args.dt,
        save_every=0,
        verbose=0,
    )
    wall_operator, sensor_bounds = _wall_shear_operator(fs)
    fixed_point_root = local_root / "fixed_points"
    critical_actuation, critical_shear = critical_wall_table(
        fs, fixed_point_root, wall_operator
    )

    zero_fixed_point = fixed_point_root / actuation_slug(0.0)
    fs.load_steady_state(
        [zero_fixed_point / "U0.xdmf", zero_fixed_point / "P0.xdmf"]
    )
    fs.initialize_time_stepping(ic=None)
    base_velocity = fs.fields.U0.vector().get_local().copy()
    skin_friction_scale = 2.0 / args.reynolds
    wall_observations = np.empty((len(observation_steps), wall_operator.shape[0]))
    wall_observations[0] = skin_friction_scale * np.asarray(
        wall_operator @ base_velocity
    ).ravel()

    field_append = False
    if 0 in field_step_set:
        write_total_fields(fs, run_dir, 0.0, append=False)
        field_append = True

    progress_stride = max(1, min(steps // 20, 100))
    started = time.perf_counter()
    for step in range(1, steps + 1):
        fs.step(u_ctrl=[float(trajectory["actuation"][step])])
        if step in observation_index:
            total_velocity = base_velocity + fs.fields.u_.vector().get_local()
            wall_observations[observation_index[step]] = (
                skin_friction_scale
                * np.asarray(wall_operator @ total_velocity).ravel()
            )
        if step in field_step_set:
            write_total_fields(fs, run_dir, times[step], append=field_append)
            field_append = True
        if step % progress_stride == 0 or step == steps:
            LOGGER.info(
                "%s: %d/%d, t=%.2f, a=%+.4e",
                run_id,
                step,
                steps,
                times[step],
                trajectory["actuation"][step],
            )
    runtime = time.perf_counter() - started
    fs.write_timeseries()

    np.savez_compressed(
        run_dir / "trajectory.npz",
        time=times,
        actuation=trajectory["actuation"],
        actuation_rate=trajectory["actuation_rate"],
        actuation_acceleration=trajectory["actuation_acceleration"],
    )
    np.savez_compressed(
        run_dir / "observations.npz",
        time=times[observation_steps],
        actuation=trajectory["actuation"][observation_steps],
        actuation_rate=trajectory["actuation_rate"][observation_steps],
        actuation_acceleration=trajectory["actuation_acceleration"][
            observation_steps
        ],
        lower_wall_skin_friction=wall_observations,
        wall_sensor_centers=np.mean(sensor_bounds, axis=1),
        wall_sensor_bounds=sensor_bounds,
        critical_actuation=critical_actuation,
        critical_lower_wall_skin_friction=(
            skin_friction_scale * critical_shear
        ),
    )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "backward_facing_step",
        "campaign": "deterministic_adiabatic",
        "run_id": run_id,
        "split": args.split,
        "seed": args.seed,
        "smoke": args.smoke,
        "mesh_id": args.mesh,
        "reynolds": args.reynolds,
        "dt": args.dt,
        "duration": duration,
        "steps": steps,
        "observation_dt": args.observation_dt,
        "field_dt": args.field_dt,
        "field_convention": "total fields and total wall observations",
        "base_actuation": 0.0,
        "disturbance": "none",
        "memory_time": args.memory_time,
        "recommended_discard_time": 0.0 if args.smoke else args.memory_time,
        "epsilon_a": args.epsilon,
        "trajectory": {
            "type": "paired-phase multisine",
            "cycles_at_omega_max": args.cycles,
            "actuation_limit": args.actuation_limit,
            "actuation_min": float(np.min(trajectory["actuation"])),
            "actuation_max": float(np.max(trajectory["actuation"])),
            "max_abs_rate": float(np.max(np.abs(trajectory["actuation_rate"]))),
            "max_abs_acceleration": float(
                np.max(np.abs(trajectory["actuation_acceleration"]))
            ),
            "omega_max": trajectory["omega_max"],
            "trajectory_epsilon_a": trajectory_epsilon,
            "frequency_ratios": FREQUENCY_RATIOS.tolist(),
            "frequencies": trajectory["frequencies"].tolist(),
            "mode_weights": MODE_WEIGHTS.tolist(),
            "phases": trajectory["phases"].tolist(),
            "normalization": trajectory["scale"],
        },
        "observations": {
            "quantity": "segment-averaged lower-wall skin friction",
            "count": wall_operator.shape[0],
        },
        "runtime_seconds": runtime,
        "git": _git_state(example_dir.parents[2]),
    }
    manifest["files"] = sorted(
        {
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        }
        | {"manifest.json"}
    )
    with (run_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default=os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    )
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--memory-time", type=float, default=106.23401923032505)
    parser.add_argument("--actuation-limit", type=float, default=0.01)
    parser.add_argument("--cycles", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--observation-dt", type=float, default=1.0)
    parser.add_argument("--field-dt", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(
        args.epsilon,
        args.memory_time,
        args.actuation_limit,
        args.cycles,
        args.dt,
        args.observation_dt,
    ) <= 0.0:
        parser.error("trajectory and time parameters must be positive")
    if args.actuation_limit > 0.01:
        parser.error("--actuation-limit must not exceed the fixed branch")
    if args.field_dt < 0.0:
        parser.error("--field-dt must be nonnegative")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for noisy_logger in ("FFC", "UFL", "dijitso", "utils.utils_flowsolver"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
