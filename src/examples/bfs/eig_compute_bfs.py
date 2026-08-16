"""Compute BFS eigenvalues from exported A and E operators.

Run this script in the complex ``slepc4py`` environment.  Additional complex
shift-invert targets can be passed with ``--targets 0j 0.25j 0.5j``.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from petsc4py import PETSc
from slepc4py import SLEPc

from utils.eig.eig_utils import get_mat_vp_slepc, sparse_to_petscmat


def actuation_slug(value):
    sign = "p" if value >= 0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def main(args):
    root = Path(args.output_root).expanduser().resolve()
    case = actuation_slug(args.actuation)
    output = root / "spectral_validation" / args.mesh / case
    operator_path = output / "operators"

    A = sparse_to_petscmat(sparse.load_npz(operator_path / "A.npz"))
    E = sparse_to_petscmat(sparse.load_npz(operator_path / "E.npz"))

    eigenvalues = []
    eigenvectors = []
    residuals = []
    for target in args.targets:
        values, vectors, solver = get_mat_vp_slepc(
            A=A,
            B=E,
            target=target,
            n=args.nev,
            return_eigensolver=True,
            verbose=True,
            precond_type=PETSc.PC.Type.LU,
            eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
            ksp_type=PETSc.KSP.Type.PREONLY,
            tol=args.tolerance,
        )
        eigenvalues.append(values)
        residuals.append(
            np.array(
                [solver.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                 for i in range(len(values))]
            )
        )
        if args.save_eigenvectors:
            eigenvectors.append(vectors)

    eigenvalues = np.hstack(eigenvalues)
    residuals = np.hstack(residuals)
    np.savez_compressed(
        output / "eigenValues.npz",
        eigenvalues=eigenvalues,
        relative_residuals=residuals,
    )
    np.savetxt(output / "eigenValues.txt", eigenvalues, delimiter=",")
    if args.save_eigenvectors:
        np.savez_compressed(
            output / "eigenVectors.npz", eigenvectors=np.hstack(eigenvectors)
        )

    for value, residual in zip(eigenvalues, residuals):
        print(f"{value.real:+.9f} {value.imag:+.9f}j  residual={residual:.2e}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    parser.add_argument(
        "--output-root", required=default_root is None, default=default_root
    )
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--actuation", type=float, default=0.0)
    parser.add_argument("--targets", nargs="+", type=complex, default=[0j])
    parser.add_argument("--nev", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--save-eigenvectors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
