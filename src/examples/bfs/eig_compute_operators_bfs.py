"""Export the BFS linearized operator A and descriptor mass matrix E.

Fixed points are produced by ``run_mesh_convergence.py``.  Run this script in
the FEniCS environment, for example with ``--mesh bfs_3 --actuation 0``.
"""

import argparse
import os
from pathlib import Path

import dolfin
import scipy.sparse as sparse

from examples.bfs.bfsflowsolver import make_bfs_solver
from flowcontrol.operatorgetter import OperatorGetter


def actuation_slug(value):
    sign = "p" if value >= 0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def save_matrix(matrix, filename):
    indptr, indices, data = matrix.mat().getValuesCSR()
    matrix_csr = sparse.csr_matrix(
        (data, indices, indptr), shape=(matrix.size(0), matrix.size(1))
    )
    matrix_csr.eliminate_zeros()
    sparse.save_npz(filename, matrix_csr)


def main(args):
    root = Path(args.output_root).expanduser().resolve()
    case = actuation_slug(args.actuation)
    fixed_point = root / "mesh_convergence" / args.mesh / case / "steady"
    output = root / "spectral_validation" / args.mesh / case / "operators"
    output.mkdir(parents=True, exist_ok=True)

    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=output / "_scratch",
        save_every=0,
        verbose=0,
    )
    fs.load_steady_state(
        [fixed_point / "U0.xdmf", fixed_point / "P0.xdmf"]
    )

    operators = OperatorGetter(fs)
    A = operators.get_A(
        UP0=fs.fields.UP0, autodiff=True, u_ctrl=[args.actuation]
    )
    E = operators.get_mass_matrix()

    # Constrained perturbation DOFs have identity rows in A.  Their E rows
    # must be zero so they cannot appear as spurious finite eigenmodes.
    for bc in fs.bc.bcu:
        bc.zero(E)

    save_matrix(A, output / "A.npz")
    save_matrix(E, output / "E.npz")
    print(f"Exported {A.size(0)} x {A.size(1)} operators to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    parser.add_argument(
        "--output-root", required=default_root is None, default=default_root
    )
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--actuation", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=500.0)
    return parser.parse_args()


if __name__ == "__main__":
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
