"""Export BFS linearized operators from a local FlowControl fixed point.

Matrices are written locally unless an external research-data root is given.
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
    case = actuation_slug(args.actuation)
    example_dir = Path(__file__).resolve().parent
    fixed_point = (
        args.fixed_point_dir.expanduser().resolve()
        if args.fixed_point_dir is not None
        else example_dir / "data_output" / args.mesh / "fixed_points" / case
    )
    velocity_file = fixed_point / "U0.xdmf"
    pressure_file = fixed_point / "P0.xdmf"
    if not velocity_file.exists() and args.actuation == 0.0:
        reynolds = f"{args.reynolds:g}"
        steady_dir = example_dir / "data_output" / args.mesh / "steady"
        velocity_file = steady_dir / f"U0_Re={reynolds}.xdmf"
        pressure_file = steady_dir / f"P0_Re={reynolds}.xdmf"
    if not velocity_file.exists() or not pressure_file.exists():
        raise FileNotFoundError(
            f"Missing FlowControl fixed point: {velocity_file}, {pressure_file}"
        )
    if args.output_root is None:
        output = (
            example_dir
            / "data_output"
            / args.mesh
            / "spectral_validation"
            / case
            / "operators"
        )
    else:
        root = Path(args.output_root).expanduser().resolve()
        output = root / "spectral_validation" / args.mesh / case / "operators"
    output.mkdir(parents=True, exist_ok=True)

    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=example_dir / "data_output" / args.mesh / "_operator_scratch",
        save_every=0,
        verbose=0,
    )
    fs.load_steady_state([velocity_file, pressure_file])

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
    parser.add_argument("--output-root", default=default_root)
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--actuation", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument(
        "--fixed-point-dir",
        type=Path,
        help="override directory containing U0.xdmf and P0.xdmf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
