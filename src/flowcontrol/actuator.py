from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Optional

import dolfin
import numpy as np
from numpy.typing import NDArray


class ACTUATOR_TYPE(IntEnum):
    """Enumeration of actuator types.

    Args:
        BC: boundary condition actuation
        FORCE: volumic force in momentum equation
    """

    BC = 1
    FORCE = 2


class CYLINDER_ACTUATION_MODE(IntEnum):
    """Modes for actuation on cylinders. This IntEnum is intended to be used
    by the user, it is never called by the original FlowSolver.

    Args:
        SUCTION: blowing & suction devices at the poles (ex. ActuatorBCParabolicV)
        ROTATION: rotation of whole cylinder (= ActuatorBCRotation)
    """

    SUCTION = 1
    ROTATION = 2


@dataclass(kw_only=True)
class Actuator(ABC):
    """Actuator abstract base class

    Args:
        actuator_type (ACTUATOR_TYPE): boundary condition or force actuation
        expression (dolfin.Expression): mathematical expression of the actuator profile
    """

    actuator_type: ACTUATOR_TYPE
    expression: Optional[dolfin.Expression] = None
    # only on (u,v) not p if type is force, for BC on syntax of DirichletBC

    @abstractmethod
    def load_expression(self, flowsolver) -> dolfin.Expression:
        """Load actuator expression projected on flowsolver function spaces. The reason
        behind this method is to be able to instantiate an Actuator independently from a
        FlowSolver object; then to be able to attach the first to the second and load the
        dolfin.Expression (that needs FlowSolver to be evaluated).

        Args:
            flowsolver (FlowSolver): FlowSolver that is using the actuator
        """
        pass


@dataclass(kw_only=True)
class ActuatorBC(Actuator):
    """Boundary condition actuator, inherits from abstract base class Actuator
    This actuators features an attribute _boundary_ that links the boundary to the
    actuator automatically
    --> this attribute is only required for operator B computation, we can probably do better
    See https://github.com/williamjussiau/FlowControl/issues/7

    All BC actuators are closely linked to the definition of boundaries
    (i.e. FlowSolver._make_boundaries() and FlowSolver._make_bcs())

    Args:
        boundary (dolfin.Expression): boundary on which the boundary condition is enforced
    """

    boundary: Optional[dolfin.Expression] = None


@dataclass(kw_only=True)
class ActuatorBCParabolicV(ActuatorBC):
    """Cylinder actuator: parabolic profile depending on first spatial
    coordinate only. Usually located at the poles of a cylinder.
    The width of the actuator can be computed with the static method, given
    the radius of a cylinder and an angular size in degrees.
    """

    width: float = 0.0
    position_x: float = 0.0
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def load_expression(self, flowsolver):
        expression = dolfin.Expression(
            [
                "0",
                "(x[0]-x0>=L || x[0]-x0<=-L) ? 0 : u_ctrl * -1*(x[0]-x0+L)*(x[0]-x0-L) / (L*L)",
            ],
            element=flowsolver.V.ufl_element(),
            L=self.width,
            x0=self.position_x,
            u_ctrl=0.0,
        )

        self.expression = expression

    @staticmethod
    def angular_size_deg_to_width(angular_size_deg, cylinder_radius):
        return cylinder_radius * np.sin(1 / 2 * angular_size_deg * dolfin.pi / 180)


@dataclass(kw_only=True)
class ActuatorBCRotation(ActuatorBC):
    """Cylinder actuator: tangential velocity at a radius D around a center (x0, y0).
    Typically used to model a rotating cylinder.
    """

    position_x: float = 0.0
    position_y: float = 0.0
    diameter: float = 1.0
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def load_expression(self, flowsolver):
        expression = dolfin.Expression(
            [
                "-sin(atan2(x[1]-y0,x[0]-x0))*u_ctrl*d/2",
                "cos(atan2(x[1]-y0,x[0]-x0))*u_ctrl*d/2",
            ],
            element=flowsolver.V.ufl_element(),
            y0=self.position_y,
            x0=self.position_x,
            u_ctrl=0.0,
            d=self.diameter,
        )
        self.expression = expression


@dataclass(kw_only=True)
class ActuatorBCUniformU(ActuatorBC):
    """Lid-driven cavity actuator: uniform profile on u,
    located on top boundary.
    This Actuator has type ACTUATOR_TYPE.BC, which means it is closely linked
    to the definition of boundaries (i.e. FlowSolver._make_boundaries() and
    FlowSolver._make_bcs())"""

    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def load_expression(self, flowsolver):
        expression = dolfin.Expression(
            [
                "u_ctrl",
                "0",
            ],
            element=flowsolver.V.ufl_element(),
            u_ctrl=0.0,
        )

        self.expression = expression


@dataclass(kw_only=True)
class ActuatorBCGaussianV(ActuatorBC):
    """Finite-support, flux-normalized Gaussian vertical boundary actuator.

    ``u_ctrl`` is the volume flux per unit span. ``inward_sign`` selects the
    sign of the vertical velocity that points into the domain. Subtracting the
    endpoint value makes the expression meet a neighboring no-slip boundary
    continuously at the ends of its finite support.
    """

    position_x: float = 0.0
    sigma: float = 0.1
    half_widths: float = 4.0
    inward_sign: float = 1.0
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    @property
    def half_width(self) -> float:
        return self.half_widths * self.sigma

    @property
    def normalization(self) -> float:
        endpoint = math.exp(-(self.half_widths**2))
        integral = self.sigma * (
            math.sqrt(math.pi) * math.erf(self.half_widths)
            - 2.0 * self.half_widths * endpoint
        )
        return 1.0 / integral

    def load_expression(self, flowsolver):
        if self.sigma <= 0.0:
            raise ValueError("Actuator sigma must be positive")
        if self.half_widths <= 0.0:
            raise ValueError("Actuator half_widths must be positive")
        if self.inward_sign not in (-1.0, 1.0):
            raise ValueError("Actuator inward_sign must be -1 or +1")

        self.expression = dolfin.Expression(
            [
                "0.0",
                (
                    "fabs(x[0]-xc) > half_width ? 0.0 : "
                    "inward_sign*u_ctrl*normalization*"
                    "(exp(-pow((x[0]-xc)/sigma, 2))-endpoint)"
                ),
            ],
            element=flowsolver.V.ufl_element(),
            u_ctrl=0.0,
            xc=self.position_x,
            sigma=self.sigma,
            half_width=self.half_width,
            inward_sign=self.inward_sign,
            normalization=self.normalization,
            endpoint=math.exp(-(self.half_widths**2)),
        )


@dataclass(kw_only=True)
class ActuatorForceGaussianV(Actuator):
    """Cavity actuator: volumic force with Gaussian profile acting on
    the second component of the velocity, centered at custom position.
    This Actuator has type ACTUATOR_TYPE.FORCE, so it is taken into account
    automatically when building equations (in FlowSolver._make_varfs())."""

    sigma: float
    position: NDArray[np.float64]
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.FORCE

    def load_expression(self, flowsolver):
        expression = dolfin.Expression(
            [
                "0",
                "u_ctrl * eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))",
            ],
            element=flowsolver.V.ufl_element(),
            eta=1,
            sig=self.sigma,
            x10=self.position[0],
            x20=self.position[1],
            u_ctrl=1.0,
        )

        BtB = dolfin.norm(expression, mesh=flowsolver.mesh)
        expression.eta = 1 / BtB
        expression.u_ctrl = 0.0
        self.expression = expression

@dataclass(kw_only=True)
class ActuatorForceGaussianAngled(Actuator):
    sigma: float
    A: float
    r: float
    theta: float
    center: np.ndarray  # [x_c, y_c]
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.FORCE

    def load_expression(self, flowsolver):
        expr = dolfin.Expression(
            [
                # x-component
                "cos_theta * (A/(2*pi*sig*sig)) * ("
                "exp(-0.5*(pow(x[0]-x1,2) + pow(x[1]-y1,2))/(sig*sig))"
                " - "
                "exp(-0.5*(pow(x[0]-x2,2) + pow(x[1]-y2,2))/(sig*sig))"
                ")",
                # y-component
                "sin_theta * (A/(2*pi*sig*sig)) * ("
                "exp(-0.5*(pow(x[0]-x1,2) + pow(x[1]-y1,2))/(sig*sig))"
                " + "
                "exp(-0.5*(pow(x[0]-x2,2) + pow(x[1]-y2,2))/(sig*sig))"
                ")"
            ],
            element=flowsolver.V.ufl_element(),
            A=self.A,
            sig=self.sigma,
            cos_theta=np.cos(self.theta),
            sin_theta=np.sin(self.theta),
            x1=self.center[0] + self.r * np.cos(self.theta),
            y1=self.center[1] + self.r * np.sin(self.theta),
            x2=self.center[0] + self.r * np.cos(-self.theta),
            y2=self.center[1] + self.r * np.sin(-self.theta),
        )
        self.expression = expr


if __name__ == "__main__":
    print("-" * 10)
    try:
        actuator = Actuator(ACTUATOR_TYPE=ACTUATOR_TYPE.BC)  # type: ignore
    except TypeError as e:
        print(e)

    actuator_cylinder = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg=10, cylinder_radius=3
        ),
        position_x=1.3,
    )
    actuator_cavity = ActuatorForceGaussianV(
        sigma=0.00849, position=np.array([-0.1, 0.02])
    )

    print("-" * 10)
    print(actuator_cylinder)
    print("-" * 10)
    print(actuator_cavity)
    print("-" * 10)
    print("-" * 10)
