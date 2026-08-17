"""Continue BFS fixed points in actuator amplitude on the production mesh.

Checkpoints are always written locally. Supplying a research-data root also
exports raw fields, wall observations, and manifests.
"""

import argparse
import logging
import os
from pathlib import Path

import dolfin
import numpy as np

import utils.utils_flowsolver as flu
from examples.bfs.bfsdiagnostics import save_ground_truth, solution_diagnostics
from examples.bfs.bfsflowsolver import make_bfs_solver


REYNOLDS = 500.0
MESH_NAME = "bfs_3"


def actuation_slug(value):
    sign = "p" if value >= 0.0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def write_fixed_point(fs, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    flu.write_xdmf(checkpoint_dir / "U0.xdmf", fs.fields.U0, "U0")
    flu.write_xdmf(checkpoint_dir / "P0.xdmf", fs.fields.P0, "P0")


def solve_newton(fs, actuation, initial_guess):
    guess = dolfin.Function(fs.W)
    guess.assign(initial_guess)
    try:
        fs.compute_steady_state(
            method="newton",
            max_iter=25,
            u_ctrl=[actuation],
            initial_guess=guess,
        )
        return "Newton"
    except RuntimeError:
        fs.compute_steady_state(
            method="picard",
            max_iter=20,
            tol=1e-8,
            u_ctrl=[actuation],
            initial_guess=initial_guess,
        )
        fs.compute_steady_state(
            method="newton",
            max_iter=25,
            u_ctrl=[actuation],
            initial_guess=fs.fields.UP0,
        )
        return "Picard+Newton fallback"


def export_fixed_point(fs, args, actuation, history, local_root, research_root):
    case = actuation_slug(actuation)
    write_fixed_point(fs, local_root / "fixed_points" / case)
    if research_root is None:
        diagnostics, _ = solution_diagnostics(fs, actuation)
    else:
        diagnostics = save_ground_truth(
            fs,
            run_dir=research_root / "critical_manifold" / args.mesh / case,
            mesh_name=args.mesh,
            actuation=actuation,
            continuation_history=history,
            campaign="critical_manifold",
            write_steady_checkpoint=False,
        )
    logging.info(
        "a=%+.5f: residual %.3e, mass imbalance %.3e",
        actuation,
        diagnostics["free_dof_residual_rms"],
        diagnostics["relative_mass_imbalance"],
    )


def main(args):
    example_dir = Path(__file__).resolve().parent
    local_root = example_dir / "data_output" / args.mesh
    research_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else None
    )
    reynolds = f"{args.reynolds:g}"

    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=local_root,
        save_every=0,
        verbose=args.verbose,
    )
    steady_dir = local_root / "steady"
    fs.load_steady_state(
        [
            steady_dir / f"U0_Re={reynolds}.xdmf",
            steady_dir / f"P0_Re={reynolds}.xdmf",
        ]
    )
    zero_state = dolfin.Function(fs.W)
    zero_state.assign(fs.fields.UP0)
    initial_history = [
        {
            "parameter": "restart",
            "value": 0.0,
            "method": f"loaded local Re={reynolds} checkpoint",
        }
    ]
    export_fixed_point(
        fs, args, 0.0, initial_history, local_root, research_root
    )

    values = np.linspace(
        -args.actuation_limit,
        args.actuation_limit,
        args.num_actuations,
    )
    branches = (values[values > 0.0], values[values < 0.0][::-1])
    for branch in branches:
        initial_guess = dolfin.Function(fs.W)
        initial_guess.assign(zero_state)
        history = initial_history.copy()
        for actuation in branch:
            actuation = float(actuation)
            method = solve_newton(fs, actuation, initial_guess)
            history = history + [
                {
                    "parameter": "actuation",
                    "value": actuation,
                    "method": method,
                }
            ]
            export_fixed_point(
                fs, args, actuation, history, local_root, research_root
            )
            initial_guess = fs.fields.UP0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    parser.add_argument(
        "--output-root",
        default=default_output,
        help="optional external research-data root",
    )
    parser.add_argument("--mesh", default=MESH_NAME)
    parser.add_argument("--reynolds", type=float, default=REYNOLDS)
    parser.add_argument("--num-actuations", type=int, default=21)
    parser.add_argument("--actuation-limit", type=float, default=0.01)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    if args.num_actuations < 3 or args.num_actuations % 2 == 0:
        parser.error("--num-actuations must be an odd integer of at least 3")
    if args.actuation_limit <= 0.0:
        parser.error("--actuation-limit must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
