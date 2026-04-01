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
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('/Users/jaking/Desktop/PhD/cylinder', nargout=0)
eng.eval('clear all; clc;', nargout=0)
eng.load('mpc_controller_workspace_forced_delay.mat', nargout=0)
eng.eval('rng(1);', nargout=0)
np.random.seed(0)

# eng.cd('/Users/jaking/Desktop/PhD/Cylinder', nargout=0)
# eng.load('lqr_controller_workspace_body_force.mat', nargout=0)
# eng.cd('/Users/jaking/Desktop/PhD/Cylinder/signal code', nargout=0)
# eng.load('ssm_controller_workspace_body_force.mat', nargout=0)

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

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    angular_size_deg = 10
    actuator_bc_1 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    actuator_bc_2 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    # params_control = flowsolverparameters.ParamControl(
    #     sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
    #     actuator_list=[actuator_force_1, actuator_force_2],
    # )
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=2.0
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

    # Find sensor and actuator DoF indices
    V = fs.V  # or whatever your velocity FunctionSpace is
    sensor_position = np.array([3.0, 0.0])
    dof_index = find_sensor_dof_index(V, sensor_position)
    print("Sensor DoF index:", dof_index)


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

    # --- Define MPC parameters at the top ---
    N = 30
    skip_steps = 30
    save_every_train = 30
    steps_per_pred = skip_steps // save_every_train
    alpha = 2
    beta = 10
    gamma = 0
    u_bounds = [-1.5, 1.5]
    delta_u = 0.8
    E_max = 7.0
    eta_max = 30.0

    # Recreate Energy map from workspace
    eng.eval("energy_map = @(eta) create_energy_map(eta, c_opt, exponents);", nargout=0)
    eng.workspace['SSMDim'] = float(3)

    control_window_length = (int(np.ceil(2 * eng.workspace['SSMDim'] + 1)) + int(eng.workspace['overEmbedControl']) - 1) * int(eng.workspace['ShiftSteps']) + 1
    signal_buffer_length = (int(np.ceil(2 * eng.workspace['SSMDim'] + 1)) + int(eng.workspace['overEmbed']) - 1) * int(eng.workspace['ShiftSteps']) + 1

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)
    warmup_steps = 5000
    Kss = Controller.from_file(file=cwd / "data_input" / "Kopt_reduced13.mat", x0=0)

    y_history = [0.0] * signal_buffer_length
    u_history = [0.0] * control_window_length
    u0 = np.zeros((N, 1))
    u_ctrl_block = 0.0
    switch = False

    for i in range(fs.params_time.num_steps):
        energy = fs.compute_energy()
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        y_history.append(float(y_meas[0]))
        if len(y_history) > signal_buffer_length:
            y_history.pop(0)

        u_history.append(float(u_ctrl_block))
        if len(u_history) > control_window_length:
            u_history.pop(0)
        # Update u_history at every ROM step (not just MPC step)
        # if i % save_every_train == 0:
        #     u_history.append(float(u_ctrl_block))
        #     if len(u_history) > control_window_length:
        #         u_history.pop(0)

        # --- MPC control phase ---
        if len(y_history) == signal_buffer_length and i % skip_steps == 0 and i > warmup_steps and not switch:
            print(f"Step {i}, Energy: {energy:.6f}, Control: {u_ctrl_block:.4f}")

            u_opt, eta_pred, energy_pred = eng.mpc_controller_delay_from_history(
                matlab.double(y_history),
                matlab.double(np.array(u_history).reshape(-1, 1)),
                eng.workspace['V'],
                eng.workspace['reduced_dynamics'],
                eng.workspace['B_const'],
                eng.workspace['energy_map'],
                float(N),
                matlab.double([u_bounds]),
                float(alpha),
                float(beta),
                float(gamma),
                float(delta_u),
                float(E_max),
                float(eta_max),
                float(eng.workspace['SSMDim']),
                float(eng.workspace['overEmbed']),
                float(eng.workspace['ShiftSteps']),
                float(steps_per_pred),
                float(save_every_train),
                float(eng.workspace['overEmbedControl']),
                matlab.double(u0.tolist()),
                nargout=3
            )
            u_ctrl_block = float(u_opt[0][0])
            # Shift previous solution and pad with zero for warm start
            u0 = np.vstack([np.array(u_opt[1:]), np.zeros((1, 1))])

            # Switch logic
            # if energy < 2.0:
            #     switch = True

        # --- Switch to steady-state controller if triggered ---
        if switch and i > warmup_steps:
            print(f"Step {i}, Energy: {energy:.6f}, Control: {u_ctrl_block:.4f}")
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            u_ctrl_block = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)[0]
            u_ctrl_block = np.clip(u_ctrl_block, -1, 1)

        fs.step(u_ctrl=np.repeat(u_ctrl_block, repeats=2, axis=0))
    # # Terminal cost (if needed)
    # P = eng.feval('get_terminal_cost', eng.workspace['reduced_dynamics'], eng.workspace['B_const'], eng.workspace['energy_map'], float(alpha))
    # P_scale = 0.0  # Set to 0 for no terminal cost, 1.0 for full
    # P_scaled = np.array(P) * P_scale
    # eng.workspace['P'] = matlab.double(P_scaled.tolist())

    # # --- Initialize control history for warm start ---
    # u0 = np.zeros((N, 1))  # Initial guess for optimizer
    # u_ctrl_block = 0.0     # Last applied control
    # switch = False

    # logger.info("Init time-stepping")
    # fs.initialize_time_stepping(ic=None)
    # warmup_steps = 5000
    # Kss = Controller.from_file(file=cwd / "data_input" / "Kopt_reduced13.mat", x0=0)

    # for i in range(fs.params_time.num_steps):
    #     energy = fs.compute_energy()
    #     if i % skip_steps == 0 and i > warmup_steps and not switch:
    #         print(f"Step {i}, Energy: {fs.compute_energy():.6f}, Control: {u_ctrl_block:.4f}")

    #         energy = fs.compute_energy()
    #         if energy < 2.0:
    #             switch = True

    #         u_current = fs.fields.u_.vector().get_local()  # full velocity field

    #         # Project to reduced coordinates (POD modes)
    #         V = np.array(eng.workspace['V'])  # [full_dim x SSMDim]
    #         eta_current = V.T @ u_current     # [SSMDim,]
    #         eta_current_matlab = eta_current.reshape(-1, 1).tolist()

    #         # Set variables in MATLAB workspace
    #         eng.workspace['eta_current'] = matlab.double(eta_current_matlab)
    #         eng.workspace['N'] = float(N)
    #         eng.workspace['u_bounds'] = matlab.double([[u_bounds[0]], [u_bounds[1]]])
    #         eng.workspace['alpha'] = float(alpha)
    #         eng.workspace['beta'] = float(beta)
    #         eng.workspace['gamma'] = float(gamma)
    #         eng.workspace['delta_u'] = float(delta_u)
    #         eng.workspace['E_max'] = float(E_max)
    #         eng.workspace['eta_max'] = float(eta_max)
    #         eng.workspace['u_prev_applied'] = float(u_ctrl_block)
    #         eng.workspace['u0'] = matlab.double(u0.tolist())
    #         eng.workspace['steps_per_pred'] = float(steps_per_pred)

    #         # Call MPC controller using workspace variables
    #         u_opt, eta_pred, energy_pred = eng.eval(
    #             "mpc_controller(eta_current, reduced_dynamics, B_const, energy_map, N, u_bounds, alpha, beta, gamma, P, delta_u, E_max, eta_max, u_prev_applied, u0, steps_per_pred)",
    #             nargout=3
    #         )
    #         u_ctrl_block = float(u_opt[0][0])  # Use first control input from MPC

    #         # Shift sequence and pad with zero (encourage return to zero control)
    #         u0 = np.vstack([np.array(u_opt[1:]), np.zeros((1, 1))])

    #     # Apply the same control for skip_steps steps
    #     if switch and i > warmup_steps:
    #         energy = fs.compute_energy()
    #         print(f"Step {i}, Energy: {fs.compute_energy():.6f}, Control: {u_ctrl_block:.4f}")
    #         y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
    #         u_ctrl_block = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)[0]
    #         u_ctrl_block = np.clip(u_ctrl_block, -1, 1)
        
    #     fs.step(u_ctrl=np.repeat(u_ctrl_block, repeats=2, axis=0))

    ###################### Delay embedded control ######################
    # buffer_length = (int(np.ceil(2 * eng.workspace['SSMDim'] + 1)) + int(eng.workspace['overEmbed']) - 1) * int(eng.workspace['ShiftSteps']) + 1
    # u_scalar_history = []
    # t_history = []

    # for i in range(fs.params_time.num_steps):
    #     y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
    #     t_now = fs.t
    #     # u_current = fs.fields.u_.vector().get_local()
    #     # print("Norm of current velocity field:", np.linalg.norm(u_current))
    #     print("Current perturbation Energy:", fs.compute_energy())

    #     u_scalar_history.append(float(y_meas[0]))
    #     t_history.append(float(t_now))
    #     if len(u_scalar_history) > buffer_length:
    #         u_scalar_history.pop(0)
    #         t_history.pop(0)

    #     if len(u_scalar_history) == buffer_length:
    #         u_ctrl_matlab = eng.ssm_feedback_controller(
    #             matlab.double(u_scalar_history),
    #             matlab.double(t_history),
    #             eng.workspace['IMInfo'],
    #             eng.workspace['RDInfo'],
    #             eng.workspace['d_forcing'],
    #             eng.workspace['K'],
    #             eng.workspace['SSMDim'],
    #             eng.workspace['overEmbed'],
    #             eng.workspace['ShiftSteps']
    #         )
    #         try:
    #             u_ctrl = float(u_ctrl_matlab)
    #         except (TypeError, ValueError):
    #             u_ctrl = 0.0
    #     else:
    #         u_ctrl = 0.0

    #     fs.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

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

    forced_dir = base_dir / f"Re{Re}_boundary_force_mpc_laptop_delay" / "run1"
    forced_dir.mkdir(parents=True, exist_ok=True)

    run_forced_simulation(Re, forced_dir, num_steps_forced, forcing_amplitude, forcing_frequency)