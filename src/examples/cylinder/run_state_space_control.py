"""Run the nonlinear cylinder with a state-space feedback controller.

The boundary-resolvent model and nonlinear actuator use opposite input-sign
conventions. Accordingly, the controller receives ``+y`` by default. Velocity
and pressure XDMF files are written for inspection in ParaView.
"""

import argparse
import json
import logging
from pathlib import Path

import dolfin
import numpy as np
import pandas as pd

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint


HERE = Path(__file__).resolve().parent
DEFAULT_CONTROLLER = (
    HERE / "data_input" / "K_thesis_structured_order5_fmincon.mat"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "data_output" / "state_space_control",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--initial-amplitude", type=float, default=0.01)
    parser.add_argument(
        "--feedback-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help=(
            "Multiplier from measured y to controller input. The boundary-FRF "
            "model has the opposite input sign from the nonlinear actuator."
        ),
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Symmetric actuator limit; use a non-positive value to disable.",
    )
    return parser.parse_args()


def make_flow_solver(args):
    flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    flow.user_data["D"] = 1.0
    time = flowsolverparameters.ParamTime(
        num_steps=args.steps, dt=args.dt, Tstart=0.0
    )
    save = flowsolverparameters.ParamSave(
        save_every=args.save_every, path_out=args.output
    )
    solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )
    mesh = flowsolverparameters.ParamMesh(
        meshpath=HERE / "data_input" / "O1.xdmf"
    )
    mesh.user_data.update({"xinf": 20, "xinfa": -10, "yinf": 10})

    width = ActuatorBCParabolicV.angular_size_deg_to_width(10, 0.5)
    actuators = [
        ActuatorBCParabolicV(width=width, position_x=0.0),
        ActuatorBCParabolicV(width=width, position_x=0.0),
    ]
    sensors = [
        SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0.0]))
    ]
    control = flowsolverparameters.ParamControl(
        sensor_list=sensors, actuator_list=actuators
    )
    initial = flowsolverparameters.ParamIC(
        xloc=2.0,
        yloc=0.0,
        radius=0.5,
        amplitude=args.initial_amplitude,
    )
    return CylinderFlowSolver(
        params_flow=flow,
        params_time=time,
        params_save=save,
        params_solver=solver,
        params_mesh=mesh,
        params_restart=flowsolverparameters.ParamRestart(),
        params_control=control,
        params_ic=initial,
        verbose=max(1, args.save_every),
    )


def main():
    args = parse_args()
    if not args.controller.is_file():
        raise FileNotFoundError(args.controller)
    if args.steps < 1 or args.dt <= 0 or args.save_every < 1:
        raise ValueError("steps, dt, and save-every must be positive")

    dolfin.set_log_level(dolfin.LogLevel.INFO)
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)

    fs = make_flow_solver(args)
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, args.output / "subdomains.xdmf"
    )
    fs.load_steady_state(
        path_u_p=[
            HERE / "data_output" / "steady" / "U0.xdmf",
            HERE / "data_output" / "steady" / "P0.xdmf",
        ]
    )
    fs.initialize_time_stepping(ic=None)
    controller = Controller.from_file(file=args.controller)

    metadata = {
        "controller": str(args.controller.resolve()),
        "feedback_sign": args.feedback_sign,
        "feedback_law": "u = feedback_sign*K*y",
        "reynolds_number": 100,
        "steps": args.steps,
        "dt": args.dt,
        "save_every": args.save_every,
        "initial_amplitude": args.initial_amplitude,
        "saturation": args.saturation,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    rows = []
    for index in range(args.steps):
        measurement = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        feedback = np.array([args.feedback_sign * measurement[0]])
        command_raw = float(controller.step(y=feedback, dt=args.dt)[0])
        command = command_raw
        if args.saturation > 0:
            command = float(np.clip(command, -args.saturation, args.saturation))
        fs.step(u_ctrl=np.array([command, command]))
        rows.append(
            {
                "time": fs.t,
                "measurement": measurement[0],
                "command_raw": command_raw,
                "command_applied": command,
                "controller_state_norm": np.linalg.norm(controller.x),
            }
        )
        if (index + 1) % args.save_every == 0:
            energy = fs.compute_energy()
            print(
                f"step={index + 1}, t={fs.t:.4f}, y={fs.y_meas[0]:.6e}, "
                f"u={command:.3e}, dE={energy:.6e}"
            )

    fs.write_timeseries()
    pd.DataFrame(rows).to_csv(
        args.output / "controller_timeseries.csv", index=False
    )
    print(f"Finished. Open {fs.paths['U']} and {fs.paths['P']} in ParaView.")


if __name__ == "__main__":
    main()
