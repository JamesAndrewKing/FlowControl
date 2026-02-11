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
eng.cd('/Users/james/Desktop/PhD/cylinder', nargout=0)
eng.eval('clear all; clc;', nargout=0)
eng.load('mpc_controller_workspace_forced_poly_yalmip.mat', nargout=0)
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

    # actuator_force_1 = ActuatorForceGaussianV(
    #     sigma=0.6, position=np.array([2.5, 0.0])
    # )
    # actuator_force_2 = ActuatorForceGaussianV(
    #     sigma=0.6, position=np.array([1.5, 0.0])
    # )
    # actuator_force_1 = ActuatorForceGaussianV(
    #     sigma=0.5, position=np.array([0.0, 0.5])
    # )
    # actuator_force_2 = ActuatorForceGaussianV(
    #     sigma=0.5, position=np.array([1.0, -0.5])
    # )

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

    # # dof_indices_1 = find_actuator_dof_indices(fs.V, np.array([0.0, 0.5]), 0.1)
    # # dof_indices_2 = find_actuator_dof_indices(fs.V, np.array([0.0, -0.5]), 0.1)
    # dof_indices_1 = find_actuator_dof_indices(fs.V, np.array([0.0, 0.5]), 0.5)
    # dof_indices_2 = find_actuator_dof_indices(fs.V, np.array([0.0, -0.5]), 0.5)

    # print("Actuator 1 DoFs:", dof_indices_1)
    # print("Actuator 2 DoFs:", dof_indices_2)

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

    ###################### Minimal Controller ######################
    # mat = loadmat(str(cwd / 'data_output' / 'eig_data.mat'))
    # eig_data = mat['eig_data']
    # eigvals = np.array([eig_data[0, i]['lambda'][0, 0] for i in range(eig_data.shape[1])])
    # eigvecs = np.array([eig_data[0, i]['vec'].flatten() for i in range(eig_data.shape[1])]).T
    # unstable_idx = np.argmax(eigvals.real)  # or use a threshold: eigvals.real > 0
    # unstable_eigvec = eigvecs[:, unstable_idx]
    # # Get velocity DOF indices in the mixed space
    # vel_dofs_in_mixed = np.array(fs.W.sub(0).dofmap().dofs())

    # # Suppose unstable_eigvec is your mixed-space eigenvector
    # unstable_eigvec_U = unstable_eigvec[vel_dofs_in_mixed]

    # logger.info("Init time-stepping")
    # fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)
    # K = np.array([0.01, 0.01])

    # for i in range(fs.params_time.num_steps):
    #     print("Current perturbation Energy:", fs.compute_energy())
    #     u_current = fs.fields.u_.vector().get_local()
    #     phi_r = np.real(unstable_eigvec_U)
    #     phi_i = np.imag(unstable_eigvec_U)
    #     V = np.column_stack((phi_r, phi_i))  # shape (ndofs, 2)
    #     z = np.linalg.lstsq(V, u_current, rcond=None)[0]
    #     u_ctrl = - K @ z
    #     fs.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    # Load eigenvalues and eigenvectors
    # mat = loadmat(str(cwd / 'data_output' / 'eig_data.mat'))
    # eig_data = mat['eig_data']
    # eigvals = np.array([eig_data[0, i]['lambda'][0, 0] for i in range(eig_data.shape[1])])
    # eigvecs = np.array([eig_data[0, i]['vec'].flatten() for i in range(eig_data.shape[1])]).T
    # eigvecs_left = np.array([eig_data[0, i]['lvec'].flatten() for i in range(eig_data.shape[1])]).T
    # mat_E = loadmat(str(cwd / 'data_output' / 'operators' / 'E_sparse.mat'))
    # E_data = mat_E['E_data'].flatten()
    # E_indices = mat_E['E_indices'].flatten()
    # E_indptr = mat_E['E_indptr'].flatten()
    # E_shape = tuple(mat_E['E_shape'].flatten())
    # E = csr_matrix((E_data, E_indices, E_indptr), shape=E_shape)
    # mat_A = loadmat(str(cwd / 'data_output' / 'operators' / 'A_sparse.mat'))
    # A_data = mat_A['A_data'].flatten()
    # A_indices = mat_A['A_indices'].flatten()
    # A_indptr = mat_A['A_indptr'].flatten()
    # A_shape = tuple(mat_A['A_shape'].flatten())
    # A = csr_matrix((A_data, A_indices, A_indptr), shape=A_shape)

    # UP_lift_data = np.load(cwd / "data_output" / "UP_lin_field_data_unit_control.npy")
    # C = UP_lift_data

    # logger.info("Init time-stepping")
    # fs.initialize_time_stepping(ic=None) 

    # # Find the most unstable mode
    # unstable_idx = np.argmax(eigvals.real)
    # unstable_eigvec = eigvecs[:, unstable_idx]
    # unstable_lefteigvec = eigvecs_left[:, unstable_idx]

    # print("Right eigenvalue:", eigvals[unstable_idx])
    # print("Biorthogonality:", unstable_lefteigvec.conj().T @ E @ unstable_eigvec)

    # # For right eigenvector
    # residual = A @ unstable_eigvec - eigvals[unstable_idx] * (E @ unstable_eigvec)
    # print("Right eigenvector residual norm:", np.linalg.norm(residual))

    # # For left eigenvector
    # residual_left = A.T @ unstable_lefteigvec - np.conj(eigvals[unstable_idx]) * (E.T @ unstable_lefteigvec)
    # print("Left eigenvector residual norm:", np.linalg.norm(residual_left))

    # # # Get velocity DOF indices in the mixed space
    # # V_to_W_vel_mapping = np.load(str(cwd / 'data_output' / 'V_to_W_vel_mapping.npy'))
    # # unstable_eigvec_U = unstable_eigvec[V_to_W_vel_mapping]
    # # unstable_lefteigvec_U = unstable_lefteigvec[V_to_W_vel_mapping]
    # # E_vel = E[V_to_W_vel_mapping, :][:, V_to_W_vel_mapping]

    # # --- Assume eigvals, eigvecs, eigvecs_left, A, E, C are loaded as before ---

    # # Find the most unstable mode
    # unstable_idx = np.argmax(eigvals.real)
    # phi_raw = eigvecs[:, unstable_idx]           # Right eigenvector (complex)
    # psi_raw = eigvecs_left[:, unstable_idx]      # Left eigenvector (complex)

    # # --- E-norm normalization and biorthogonal scaling ---s
    # phi = phi_raw / np.sqrt(np.dot(phi_raw.conj(), E.dot(phi_raw)))  # ||phi||_E = 1
    # psi = psi_raw / np.dot(psi_raw.conj(), E.dot(phi))   

    # # Split into real and imaginary parts
    # phi_r = np.real(phi)
    # phi_i = np.imag(phi)
    # psi_r = np.real(psi)
    # psi_i = np.imag(psi)

    # # Build real-valued basis for reduced system
    # V = np.column_stack([phi_r, phi_i])    # shape (n, 2)
    # W = np.column_stack([psi_r, psi_i])    # shape (n, 2)

    # # Biorthogonal normalization: W^H E V = I
    # norm_mat = W.conj().T @ E @ V
    # W = W @ np.linalg.inv(norm_mat)  # Now W^H E V = I

    # # Reduced system matrices
    # A_hat = W.conj().T @ A @ V       # shape (2, 2)
    # C_hat = W.conj().T @ E @ C       # shape (2,)

    # print("A_hat (reduced system matrix):\n", A_hat)
    # print("C_hat (reduced control matrix):\n", C_hat)

    # # eigval = eigvals[unstable_idx]
    # # err = A @ V - E @ V @ np.array([[eigval.real, eigval.imag], [-eigval.imag,eigval.real]])

    # # Feedback gains (tune as needed)
    # k_r = -0.05 / C_hat[0]
    # k_i = -0.05 / C_hat[1]
    # print(f"Feedback gains: k_r = {k_r}, k_i = {k_i}")

    # u_ctrl = 0.0
    # u_ctrl_dot_prev = 0.0

    # logger.info("Init time-stepping")
    # fs.initialize_time_stepping(ic=None)

    # B_norm = np.dot(psi.conj(), E@C/np.linalg.norm(E@C))  # psi^H b
    # print("Normalized Actuator coupling B:", B_norm)

    # for i in range(fs.params_time.num_steps):
    #     up_current = fs.fields.up_.vector().get_local()
    #     x = up_current - u_ctrl * C

    #     # Project onto real and imaginary parts of the unstable mode
    #     z_vec = W.conj().T @ E @ x  # shape (2,)
    #     z_r, z_i = z_vec[0], z_vec[1]

    #     # Proportional feedback on both components
    #     c = k_r * z_r + k_i * z_i
    #     u_ctrl_dot = -c  # For lifting convention

    #     # Integrate control signal
    #     if i == 0:
    #         u_ctrl_new = u_ctrl + params_time.dt * u_ctrl_dot
    #     else:
    #         u_ctrl_new = u_ctrl + 0.5 * params_time.dt * (u_ctrl_dot + u_ctrl_dot_prev)

    #     fs.step(u_ctrl=[float(u_ctrl_new)] * 2)
    #     print(f"Step {i}, z_r = {z_r}, z_i = {z_i}, control = {u_ctrl_new}, energy = {fs.compute_energy()}")

    #     u_ctrl = u_ctrl_new
    #     u_ctrl_dot_prev = u_ctrl_dot

    ###################### Full Phase Space Control ######################
    # knots = np.linspace(fs.params_time.Tstart, fs.params_time.Tfinal, 10)
    # values = np.random.uniform(-forcing_amplitude, forcing_amplitude, len(knots))
    # cs = CubicSpline(knots, values)
    # logger.info("Init time-stepping")
    # fs.initialize_time_stepping(ic=None)
    # for i in range(fs.params_time.num_steps):
    #     # y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
    #     print("Current perturbation Energy:", fs.compute_energy())
    #     u_current = fs.fields.u_.vector().get_local() # this is the full velocity field, not the control
    #     u_ctrl = eng.lqr_controller_matlab_body_force(
    #         matlab.double(u_current.tolist()),
    #         eng.workspace['IMInfo'],
    #         eng.workspace['RDInfo'],
    #         eng.workspace['Q'],
    #         eng.workspace['R'],
    #         eng.workspace['d_forcing'],
    #     )
    #     u_ctrl = float(u_ctrl)  # Convert from matlab.double to float
    #     # u_ctrl = forcing_amplitude * np.sin(forcing_frequency * fs.t)
    #     # u_ctrl = cs(fs.t)
    #     fs.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    # --- Define MPC parameters at the top ---
    N = 30
    skip_steps = 40
    save_every_train = 10
    steps_per_pred = skip_steps // save_every_train
    alpha = 0.01
    beta = 0.01
    gamma = 0
    u_bounds = [-1.5, 1.5]
    delta_u = 0.8
    E_max = 7.0
    eta_max = 30.0

    # Recreate Energy map from workspace
    eng.eval("energy_map = @(eta) create_energy_map(eta, c_opt, exponents);", nargout=0)

    # Terminal cost (if needed)
    P = eng.feval('get_terminal_cost', eng.workspace['reduced_dynamics'], eng.workspace['B_const'], eng.workspace['energy_map'], float(alpha))
    P_scale = 0.0  # Set to 0 for no terminal cost, 1.0 for full
    P_scaled = np.array(P) * P_scale
    eng.workspace['P'] = matlab.double(P_scaled.tolist())

    # --- Initialize control history for warm start ---
    u0 = np.zeros((N, 1))  # Initial guess for optimizer
    u_ctrl_block = 0.0     # Last applied control

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)
    warmup_steps = 3000

    for i in range(fs.params_time.num_steps):
        if i % skip_steps == 0 and i > warmup_steps:
            print(f"Step {i}, Energy: {fs.compute_energy():.6f}, Control: {u_ctrl_block:.4f}")
            u_current = fs.fields.u_.vector().get_local()  # full velocity field

            # Project to reduced coordinates (POD modes)
            V = np.array(eng.workspace['V'])  # [full_dim x SSMDim]
            eta_current = V.T @ u_current     # [SSMDim,]
            eta_current_matlab = eta_current.reshape(-1, 1).tolist()

            # Set variables in MATLAB workspace
            eng.workspace['eta_current'] = matlab.double(eta_current_matlab)
            eng.workspace['N'] = float(N)
            eng.workspace['u_bounds'] = matlab.double([[u_bounds[0]], [u_bounds[1]]])
            eng.workspace['alpha'] = float(alpha)
            eng.workspace['beta'] = float(beta)
            eng.workspace['gamma'] = float(gamma)
            eng.workspace['delta_u'] = float(delta_u)
            eng.workspace['E_max'] = float(E_max)
            eng.workspace['eta_max'] = float(eta_max)
            eng.workspace['u_prev_applied'] = float(u_ctrl_block)
            eng.workspace['u0'] = matlab.double(u0.tolist())
            eng.workspace['steps_per_pred'] = float(steps_per_pred)

            # Call MPC controller using workspace variables
            u_opt, eta_pred, energy_pred = eng.eval(
                "mpc_controller(eta_current, reduced_dynamics, B_const, energy_map, N, u_bounds, alpha, beta, gamma, P, delta_u, E_max, eta_max, u_prev_applied, u0, steps_per_pred)",
                nargout=3
            )
            u_ctrl_block = float(u_opt[0][0])  # Use first control input from MPC

            # Shift sequence and pad with zero (encourage return to zero control)
            u0 = np.vstack([np.array(u_opt[1:]), np.zeros((1, 1))])

        # Apply the same control for skip_steps steps
        fs.step(u_ctrl=np.repeat(u_ctrl_block, repeats=2, axis=0))

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
    base_dir = Path("/Users/james/Desktop/PhD/cylinder")
    base_dir.mkdir(parents=True, exist_ok=True)

    num_steps_forced = 40000
    forcing_amplitude = 0.3
    forcing_frequency = 1.0

    forced_dir = base_dir / f"Re{Re}_boundary_force_mpc_laptop_poly_long" / "run1"
    forced_dir.mkdir(parents=True, exist_ok=True)

    run_forced_simulation(Re, forced_dir, num_steps_forced, forcing_amplitude, forcing_frequency)