"""FlowSolver specialization for the backward-facing-step benchmark."""

from __future__ import annotations

from pathlib import Path

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import flowcontrol.flowsolverparameters as flowsolverparameters
from flowcontrol.actuator import ActuatorBCGaussianV
from flowcontrol.flowfield import BoundaryConditions


GEOMETRY = {
    "x_in": -10.0,
    "x_out": 50.0,
    "y_bottom": 0.0,
    "y_step": 1.0,
    "y_top": 2.0,
    "actuator_x": -1.0,
    "actuator_sigma": 0.1,
    "actuator_half_widths": 4.0,
}


class BFSFlowSolver(flowsolver.FlowSolver):
    """Expansion-ratio-two, two-dimensional backward-facing-step flow."""

    def _make_boundaries(self) -> pandas.DataFrame:
        geometry = self.params_mesh.user_data
        tol = 100.0 * dolfin.DOLFIN_EPS
        common = "on_boundary && "

        inlet = dolfin.CompiledSubDomain(
            common + "near(x[0], x_in, tol)",
            x_in=geometry["x_in"],
            tol=tol,
        )
        outlet = dolfin.CompiledSubDomain(
            common + "near(x[0], x_out, tol)",
            x_out=geometry["x_out"],
            tol=tol,
        )
        upper_wall = dolfin.CompiledSubDomain(
            common
            + "near(x[1], y_top, tol) && "
            + "(x[0] <= actuator_left + tol || x[0] >= actuator_right - tol)",
            y_top=geometry["y_top"],
            actuator_left=geometry["actuator_left"],
            actuator_right=geometry["actuator_right"],
            tol=tol,
        )
        actuator = dolfin.CompiledSubDomain(
            common
            + "near(x[1], y_top, tol) && "
            + "x[0] >= actuator_left - tol && x[0] <= actuator_right + tol",
            y_top=geometry["y_top"],
            actuator_left=geometry["actuator_left"],
            actuator_right=geometry["actuator_right"],
            tol=tol,
        )
        upstream_lower_wall = dolfin.CompiledSubDomain(
            common
            + "near(x[1], y_step, tol) && x[0] <= tol && x[0] >= x_in - tol",
            y_step=geometry["y_step"],
            x_in=geometry["x_in"],
            tol=tol,
        )
        step_wall = dolfin.CompiledSubDomain(
            common
            + "near(x[0], 0.0, tol) && "
            + "x[1] >= y_bottom - tol && x[1] <= y_step + tol",
            y_bottom=geometry["y_bottom"],
            y_step=geometry["y_step"],
            tol=tol,
        )
        downstream_lower_wall = dolfin.CompiledSubDomain(
            common
            + "near(x[1], y_bottom, tol) && x[0] >= -tol && x[0] <= x_out + tol",
            y_bottom=geometry["y_bottom"],
            x_out=geometry["x_out"],
            tol=tol,
        )

        names = [
            "inlet",
            "outlet",
            "upper_wall",
            "actuator",
            "upstream_lower_wall",
            "step_wall",
            "downstream_lower_wall",
        ]
        subdomains = [
            inlet,
            outlet,
            upper_wall,
            actuator,
            upstream_lower_wall,
            step_wall,
            downstream_lower_wall,
        ]
        return pandas.DataFrame(index=names, data={"subdomain": subdomains})

    def _make_bcs(self) -> BoundaryConditions:
        """Homogeneous perturbation BCs plus the boundary actuator."""

        zero = dolfin.Constant((0.0, 0.0))
        bcu = [
            dolfin.DirichletBC(self.W.sub(0), zero, self.get_subdomain("inlet")),
            dolfin.DirichletBC(
                self.W.sub(0), zero, self.get_subdomain("upper_wall")
            ),
            dolfin.DirichletBC(
                self.W.sub(0),
                self.params_control.actuator_list[0].expression,
                self.get_subdomain("actuator"),
            ),
            dolfin.DirichletBC(
                self.W.sub(0), zero, self.get_subdomain("upstream_lower_wall")
            ),
            dolfin.DirichletBC(
                self.W.sub(0), zero, self.get_subdomain("step_wall")
            ),
            dolfin.DirichletBC(
                self.W.sub(0), zero, self.get_subdomain("downstream_lower_wall")
            ),
        ]
        self.params_control.actuator_list[0].boundary = self.get_subdomain("actuator")
        return BoundaryConditions(bcu=bcu, bcp=[])

    def _make_BCs(self) -> BoundaryConditions:
        """Full-field BCs with the benchmark's parabolic inlet profile."""

        geometry = self.params_mesh.user_data
        inlet_profile = dolfin.Expression(
            (
                "4.0*uinf*(x[1]-y_step)*(y_top-x[1])"
                "/pow(y_top-y_step, 2)",
                "0.0",
            ),
            element=self.V.ufl_element(),
            uinf=self.params_flow.uinf,
            y_step=geometry["y_step"],
            y_top=geometry["y_top"],
        )
        bcs = self._make_bcs()
        inlet = dolfin.DirichletBC(
            self.W.sub(0), inlet_profile, self.get_subdomain("inlet")
        )
        return BoundaryConditions(bcu=[inlet] + bcs.bcu[1:], bcp=[])

    def _default_steady_state_initial_guess(self) -> dolfin.UserExpression:
        """Flux-matched developed profiles upstream and downstream."""

        geometry = self.params_mesh.user_data
        uinf = self.params_flow.uinf

        class DevelopedProfile(dolfin.UserExpression):
            def eval(self, value, x):
                if x[0] < 0.0:
                    eta = (x[1] - geometry["y_step"]) / (
                        geometry["y_top"] - geometry["y_step"]
                    )
                    value[0] = 4.0 * uinf * eta * (1.0 - eta)
                else:
                    eta = (x[1] - geometry["y_bottom"]) / (
                        geometry["y_top"] - geometry["y_bottom"]
                    )
                    value[0] = 2.0 * uinf * eta * (1.0 - eta)
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        return DevelopedProfile(degree=2)


def make_bfs_solver(
    mesh_name: str = "bfs_2",
    reynolds: float = 500.0,
    output_dir: Path | str | None = None,
    save_every: int = 0,
    verbose: int = 1,
) -> BFSFlowSolver:
    """Construct the standard BFS case with no hidden absolute paths."""

    case_dir = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = case_dir / "data_output" / mesh_name
    output_dir = Path(output_dir).expanduser().resolve()
    (output_dir / "steady").mkdir(parents=True, exist_ok=True)

    params_flow = flowsolverparameters.ParamFlow(Re=reynolds, uinf=1.0)
    params_flow.user_data["D"] = 1.0
    params_time = flowsolverparameters.ParamTime(num_steps=1, dt=0.005, Tstart=0.0)
    params_save = flowsolverparameters.ParamSave(
        save_every=save_every, path_out=output_dir
    )
    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )
    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=case_dir / "data_input" / f"{mesh_name}.xdmf"
    )
    params_mesh.user_data.update(GEOMETRY)
    params_mesh.user_data["actuator_left"] = (
        GEOMETRY["actuator_x"]
        - GEOMETRY["actuator_half_widths"] * GEOMETRY["actuator_sigma"]
    )
    params_mesh.user_data["actuator_right"] = (
        GEOMETRY["actuator_x"]
        + GEOMETRY["actuator_half_widths"] * GEOMETRY["actuator_sigma"]
    )
    params_restart = flowsolverparameters.ParamRestart()
    actuator = ActuatorBCGaussianV(
        position_x=GEOMETRY["actuator_x"],
        sigma=GEOMETRY["actuator_sigma"],
        half_widths=GEOMETRY["actuator_half_widths"],
        inward_sign=-1.0,
    )
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[], actuator_list=[actuator]
    )
    params_ic = flowsolverparameters.ParamIC(amplitude=0.0)

    return BFSFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=verbose,
    )
