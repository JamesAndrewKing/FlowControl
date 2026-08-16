"""Compute the uncontrolled BFS steady state by Reynolds continuation."""

import logging
from pathlib import Path

import dolfin

import utils.utils_flowsolver as flu
from examples.bfs.bfsflowsolver import make_bfs_solver


Re_final = 500
MESH_NAME = "bfs_3"
REYNOLDS_NUMBERS = (50, 100, 200, 300, 400, Re_final)


def main():
    dolfin.set_log_level(dolfin.LogLevel.INFO)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    output_dir = Path(__file__).parent / "data_output" / MESH_NAME
    fs = make_bfs_solver(
        mesh_name=MESH_NAME,
        reynolds=REYNOLDS_NUMBERS[0],
        output_dir=output_dir,
        verbose=10,
    )

    initial_guess = None
    for reynolds in REYNOLDS_NUMBERS:
        logger.info("Computing BFS steady state at Re=%s", reynolds)
        fs.params_flow.Re = reynolds
        if initial_guess is None:
            fs.compute_steady_state(
                method="picard",
                max_iter=15,
                tol=1e-8,
                u_ctrl=[0.0],
            )
            initial_guess = fs.fields.UP0
        fs.compute_steady_state(
            method="newton",
            max_iter=25,
            u_ctrl=[0.0],
            initial_guess=initial_guess,
        )
        initial_guess = fs.fields.UP0

        steady_dir = output_dir / "steady"
        flu.write_xdmf(
            steady_dir / f"U0_Re={reynolds}.xdmf", fs.fields.U0, "U0"
        )
        flu.write_xdmf(
            steady_dir / f"P0_Re={reynolds}.xdmf", fs.fields.P0, "P0"
        )

    (output_dir / "current_Re.txt").write_text(f"{Re_final}\n")


if __name__ == "__main__":
    main()
