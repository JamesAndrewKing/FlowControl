from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.signal import chirp
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_continuous_are
import dolfin
import logging
import time
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
import flowcontrol.flowsolverparameters as flowsolverparameters
from flowcontrol.actuator import ActuatorBCParabolicV, ActuatorForceGaussianV
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint
import utils.utils_flowsolver as flu
from examples.cylinder.compute_steady_state import Re
from flowcontrol.controller import Controller
from examples.cylinder.batch_run_cylinder import save_data

def find_sensor_dof_index(V, sensor_position):
    """
    V: FunctionSpace (e.g., velocity space)
    sensor_position: np.array([x, y])
    Returns: index of the closest DoF to the sensor position
    """
    # Get the coordinates of all DoFs in the function space
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
    # Compute distances to the sensor position
    distances = np.linalg.norm(dof_coords - sensor_position, axis=1)
    # Find the index of the closest DoF
    dof_index = np.argmin(distances)
    return dof_index


def find_actuator_dof_indices(V, actuator_position, sigma, threshold=1e-2):
    """
    V: FunctionSpace (velocity space)
    actuator_position: np.array([x, y])
    sigma: width of Gaussian
    threshold: minimum value of Gaussian to consider as 'active'
    Returns: list of DoF indices influenced by the actuator
    """
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
    distances = np.linalg.norm(dof_coords - actuator_position, axis=1)
    gaussian_values = np.exp(-0.5 * (distances / sigma)**2)
    dof_indices = np.where(gaussian_values > threshold)[0]
    return dof_indices


def run_forced_simulation(Re, save_dir, num_steps, forcing_amplitude, forcing_frequency):
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=100, path_out=save_dir
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "O1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart(
    )

    # Angled body force actuators:

    actuator_force_1 = ActuatorForceGaussianV(
        sigma=0.1, position=np.array([0.0, 0.5])
    )
    actuator_force_2 = ActuatorForceGaussianV(
        sigma=0.1, position=np.array([0.0, -0.5])
    )

    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_force_1, actuator_force_2],
    )
    # params_control = flowsolverparameters.ParamControl(
    #     sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
    #     actuator_list=[actuator_bc_1, actuator_bc_2],
    # )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=5,
    )


    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, save_dir / "subdomains.xdmf"
    )

    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0.xdmf",
            cwd / "data_output" / "steady" / f"P0.xdmf",
        ]
    )

   
    fs.initialize_time_stepping(ic=None)

    Kss = Controller.from_file(file=cwd / "K_first_try.mat", x0=0)

    for i in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        t_now = fs.t
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)
        u_ctrl = np.clip(u_ctrl, -1, 1)
        energy = fs.compute_energy()
        print(f"Step {i}, Energy: {fs.compute_energy():.6f}, Control: {u_ctrl[0]:.4f}")
        fs.step(u_ctrl=[u_ctrl[0], -u_ctrl[0]])

    #################################################################

    fs.write_timeseries()

    logger.info(fs.timeseries)
    print(f"Forced simulation finished and saved in {save_dir}")
    save_data(fs, save_dir, cwd, logger)

if __name__ == "__main__":
    base_dir = Path("/Users/jaking/Desktop/PhD/cylinder")
    base_dir.mkdir(parents=True, exist_ok=True)

    num_steps_forced = 20000
    forcing_amplitude = 0.3
    forcing_frequency = 1.0

    forced_dir = base_dir / f"Re{Re}_volume_body_force_h_inf" / "run1"
    forced_dir.mkdir(parents=True, exist_ok=True)

    run_forced_simulation(Re, forced_dir, num_steps_forced, forcing_amplitude, forcing_frequency)