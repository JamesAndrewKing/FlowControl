"""Check that a saved BFS equilibrium is invariant under time stepping."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np

from examples.bfs.bfsflowsolver import make_bfs_solver


def run(args) -> dict:
    run_dir = Path(args.run_dir).expanduser().resolve()
    with (run_dir / "manifest.json").open() as stream:
        manifest = json.load(stream)
    mesh_name = manifest["mesh_id"]
    reynolds = float(manifest["reynolds"])
    base_actuation = float(manifest["actuation"])

    fs = make_bfs_solver(
        mesh_name=mesh_name,
        reynolds=reynolds,
        output_dir=run_dir / "_restart_scratch",
        save_every=0,
        verbose=0,
    )
    fs.params_time.dt = args.dt
    fs.params_time.num_steps = args.steps
    fs.params_time.Tfinal = args.steps * args.dt
    fs.load_steady_state(
        [run_dir / "steady" / "U0.xdmf", run_dir / "steady" / "P0.xdmf"]
    )
    # The time-stepping unknown is the perturbation about the loaded frozen
    # state. A frozen actuator therefore has zero perturbation input.
    fs.flush_actuators_u_ctrl()
    fs.initialize_time_stepping(ic=None)

    samples = []
    for _ in range(args.steps):
        fs.step(u_ctrl=[0.0])
        perturbation = fs.fields.u_
        samples.append(
            {
                "step": int(fs.iter),
                "time": float(fs.t),
                "velocity_perturbation_l2": float(
                    dolfin.norm(perturbation, norm_type="L2", mesh=fs.mesh)
                ),
                "velocity_perturbation_linf_dof": float(
                    np.max(np.abs(perturbation.vector().get_local()))
                ),
            }
        )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_id": mesh_name,
        "reynolds": reynolds,
        "base_actuation": base_actuation,
        "dt": args.dt,
        "steps": args.steps,
        "samples": samples,
        "maximum_velocity_perturbation_l2": max(
            sample["velocity_perturbation_l2"] for sample in samples
        ),
        "maximum_velocity_perturbation_linf_dof": max(
            sample["velocity_perturbation_linf_dof"] for sample in samples
        ),
        "passed": all(
            sample["velocity_perturbation_l2"] <= args.tolerance
            for sample in samples
        ),
        "tolerance": args.tolerance,
    }
    with (run_dir / "restart_invariance.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    manifest.setdefault("accuracy_checks", {})["restart_invariance"] = {
        "file": "restart_invariance.json",
        "passed": result["passed"],
        "maximum_velocity_perturbation_l2": result[
            "maximum_velocity_perturbation_l2"
        ],
    }
    manifest["files"] = sorted(
        set(manifest.get("files", [])) | {"restart_invariance.json"}
    )
    with (run_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    for noisy_logger in ("FFC", "UFL", "dijitso", "utils.utils_flowsolver"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
