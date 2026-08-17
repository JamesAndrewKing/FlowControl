"""Compute the first adiabatic response along the BFS steady branch.

For each requested actuation, a three-point quadratic finite difference gives
``x_star,a`` from the existing fixed points.  The script then solves

    A b1 = E x_star,a,

which is the discrete form of the first frozen-flow adiabatic correction for
``E xdot = F(x, a)``.  Fixed points always come from FlowControl.  Operators
and results are local unless an external research-data root is supplied.
"""

import argparse
import csv
import os
from pathlib import Path

import dolfin
import numpy as np
import scipy.sparse as sparse
from petsc4py import PETSc

from examples.bfs.bfsflowsolver import make_bfs_solver


def actuation_slug(value):
    sign = "p" if value >= 0.0 else "m"
    return f"a_{sign}{abs(value):.5f}".replace(".", "p")


def actuation_from_slug(name):
    token = name.removeprefix("a_")
    sign = 1.0 if token[0] == "p" else -1.0
    return sign * float(token[1:].replace("p", "."))


def fixed_point_branch(fixed_point_root):
    branch = {}
    for path in fixed_point_root.glob("a_[mp]*"):
        if (path / "U0.xdmf").exists() and (path / "P0.xdmf").exists():
            branch[actuation_from_slug(path.name)] = path
    if len(branch) < 3:
        raise FileNotFoundError(
            f"Need at least three fixed points under {fixed_point_root}"
        )
    return dict(sorted(branch.items()))


def derivative_stencil(branch_values, target, stride=1):
    values = np.asarray(branch_values)
    matches = np.flatnonzero(np.isclose(values, target, atol=5e-10, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"Actuation {target:+.5f} is not on the fixed-point branch")
    index = int(matches[0])
    if index - stride >= 0 and index + stride < len(values):
        indices = [index - stride, index, index + stride]
    elif index + 2 * stride < len(values):
        indices = [index, index + stride, index + 2 * stride]
    elif index - 2 * stride >= 0:
        indices = [index - 2 * stride, index - stride, index]
    else:
        raise ValueError(f"Branch is too short for stride-{stride} differentiation")
    points = values[indices]
    offsets = points - target
    weights = np.linalg.solve(
        np.vstack((np.ones(3), offsets, offsets**2)),
        np.array([0.0, 1.0, 0.0]),
    )
    return points, weights


def load_state(fs, fixed_point_dir):
    fs.load_steady_state(
        [fixed_point_dir / "U0.xdmf", fixed_point_dir / "P0.xdmf"]
    )
    return fs.fields.UP0.vector().get_local().copy()


def velocity_mass_norm(fs, state_vector):
    state = dolfin.Function(fs.W)
    state.vector().set_local(state_vector)
    state.vector().apply("insert")
    velocity, _ = state.split(deepcopy=True)
    return float(dolfin.norm(velocity, norm_type="L2", mesh=fs.mesh))


def scipy_to_petsc(matrix):
    matrix = sparse.csr_matrix(matrix)
    matrix.sort_indices()
    petsc_matrix = PETSc.Mat().createAIJ(
        size=matrix.shape,
        csr=(matrix.indptr, matrix.indices, matrix.data),
        comm=PETSc.COMM_SELF,
    )
    petsc_matrix.assemble()
    return petsc_matrix


def solve_linear_system(matrix_path, rhs, factor_solver):
    matrix = scipy_to_petsc(sparse.load_npz(matrix_path))
    vector_rhs = matrix.createVecLeft()
    vector_rhs.getArray()[:] = rhs
    vector_rhs.assemble()
    response = matrix.createVecRight()

    solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    solver.setOperators(matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType(factor_solver)
    solver.setFromOptions()
    solver.solve(vector_rhs, response)
    reason = int(solver.getConvergedReason())
    if reason < 0:
        raise RuntimeError(f"PETSc solve diverged with reason {reason}")

    residual = matrix.createVecLeft()
    matrix.mult(response, residual)
    residual.axpy(-1.0, vector_rhs)
    relative_residual = residual.norm() / max(vector_rhs.norm(), np.finfo(float).tiny)
    values = response.getArray(readonly=True).copy()

    residual.destroy()
    response.destroy()
    vector_rhs.destroy()
    solver.destroy()
    matrix.destroy()
    return values, float(relative_residual)


def main(args):
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError("Run this diagnostic in serial")

    example_dir = Path(__file__).resolve().parent
    local_root = example_dir / "data_output" / args.mesh
    branch = fixed_point_branch(local_root / "fixed_points")

    if args.output_root is None:
        operator_root = local_root / "spectral_validation"
        output_dir = local_root / "conditioning"
    else:
        research_root = Path(args.output_root).expanduser().resolve()
        operator_root = research_root / "spectral_validation" / args.mesh
        output_dir = research_root / "critical_manifold" / args.mesh / "conditioning"
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = make_bfs_solver(
        mesh_name=args.mesh,
        reynolds=args.reynolds,
        output_dir=local_root / "_conditioning_scratch",
        save_every=0,
        verbose=0,
    )

    branch_values = list(branch)
    state_cache = {}
    derivatives = []
    responses = []
    rows = []
    stencils = []
    stencil_weights = []
    derivative_changes = []

    for actuation in args.actuations:
        points, weights = derivative_stencil(branch_values, actuation)
        wide_points, wide_weights = derivative_stencil(
            branch_values, actuation, stride=2
        )
        states = []
        for point in points:
            key = float(point)
            if key not in state_cache:
                state_cache[key] = load_state(fs, branch[key])
            states.append(state_cache[key])
        derivative = np.tensordot(weights, np.asarray(states), axes=(0, 0))
        wide_states = []
        for point in wide_points:
            key = float(point)
            if key not in state_cache:
                state_cache[key] = load_state(fs, branch[key])
            wide_states.append(state_cache[key])
        wide_derivative = np.tensordot(
            wide_weights, np.asarray(wide_states), axes=(0, 0)
        )

        case = actuation_slug(actuation)
        operator_dir = operator_root / case / "operators"
        matrix_a = operator_dir / "A.npz"
        matrix_e = operator_dir / "E.npz"
        if not matrix_a.exists() or not matrix_e.exists():
            raise FileNotFoundError(
                f"Missing operators in {operator_dir}. Run "
                f"eig_compute_operators_bfs.py --actuation {actuation:g} first."
            )
        mass = sparse.load_npz(matrix_e)
        if mass.shape[0] != derivative.size:
            raise ValueError(
                f"Operator/state size mismatch: {mass.shape[0]} != {derivative.size}"
            )
        rhs = np.asarray(mass @ derivative).ravel()
        response, relative_residual = solve_linear_system(
            matrix_a, rhs, args.factor_solver
        )

        derivative_norm = velocity_mass_norm(fs, derivative)
        derivative_change = (
            velocity_mass_norm(fs, derivative - wide_derivative) / derivative_norm
        )
        response_norm = velocity_mass_norm(fs, response)
        gain = response_norm / derivative_norm
        rows.append(
            {
                "actuation": float(actuation),
                "x_star_a_mass_norm": derivative_norm,
                "derivative_relative_change": derivative_change,
                "b1_mass_norm": response_norm,
                "response_ratio": gain,
                "linear_relative_residual": relative_residual,
            }
        )
        derivatives.append(derivative)
        responses.append(response)
        stencils.append(points)
        stencil_weights.append(weights)
        derivative_changes.append(derivative_change)
        print(
            f"a={actuation:+.5f}: ||x_star,a||_M={derivative_norm:.6e}, "
            f"||b1||_M={response_norm:.6e}, ratio={gain:.6e}, "
            f"FD change={derivative_change:.3e}, residual={relative_residual:.3e}"
        )

    with (output_dir / "adiabatic_conditioning.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "adiabatic_conditioning.npz",
        actuation=np.asarray(args.actuations),
        x_star_a=np.asarray(derivatives),
        b1=np.asarray(responses),
        stencil_actuation=np.asarray(stencils),
        stencil_weights=np.asarray(stencil_weights),
        x_star_a_mass_norm=np.asarray([row["x_star_a_mass_norm"] for row in rows]),
        derivative_relative_change=np.asarray(derivative_changes),
        b1_mass_norm=np.asarray([row["b1_mass_norm"] for row in rows]),
        response_ratio=np.asarray([row["response_ratio"] for row in rows]),
        linear_relative_residual=np.asarray(
            [row["linear_relative_residual"] for row in rows]
        ),
    )
    print(f"Saved conditioning data to {output_dir}")


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
        "--actuations",
        nargs="+",
        type=float,
        default=[-0.01, 0.0, 0.01],
    )
    parser.add_argument("--factor-solver", default="mumps")
    return parser.parse_args()


if __name__ == "__main__":
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    main(parse_args())
