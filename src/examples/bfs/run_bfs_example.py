"""Run a short unactuated simulation about the BFS steady state."""

import logging
import time
from pathlib import Path

import dolfin

import utils.utils_flowsolver as flu
from examples.bfs.bfsflowsolver import make_bfs_solver
from examples.bfs.compute_steady_state_increasing_Re import MESH_NAME, Re_final


def main():
    dolfin.set_log_level(dolfin.LogLevel.INFO)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start = time.time()

    output_dir = Path(__file__).parent / "data_output" / MESH_NAME
    fs = make_bfs_solver(
        mesh_name=MESH_NAME,
        reynolds=Re_final,
        output_dir=output_dir,
        num_steps=10,
        save_every=5,
        verbose=5,
    )

    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, output_dir / "subdomains.xdmf"
    )
    steady_dir = output_dir / "steady"
    fs.load_steady_state(
        [
            steady_dir / f"U0_Re={Re_final}.xdmf",
            steady_dir / f"P0_Re={Re_final}.xdmf",
        ]
    )
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    flu.summarize_timings(fs, start)
    logger.info(fs.timeseries)
    fs.write_timeseries()
    logger.info("End with success")


if __name__ == "__main__":
    main()
