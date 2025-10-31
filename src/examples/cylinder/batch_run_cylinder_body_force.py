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
# eng = matlab.engine.start_matlab()
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

    actuator_force_1 = ActuatorForceGaussianV(
        sigma=0.6, position=np.array([2.5, 0.0])
    )
    actuator_force_2 = ActuatorForceGaussianV(
        sigma=0.6, position=np.array([1.5, 0.0])
    )
    # actuator_force_1 = ActuatorForceGaussianV(
    #     sigma=0.5, position=np.array([0.0, 0.5])
    # )
    # actuator_force_2 = ActuatorForceGaussianV(
    #     sigma=0.5, position=np.array([1.0, -0.5])
    # )

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    # angular_size_deg = 10
    # actuator_bc_1 = ActuatorBCParabolicV(
    #     width=ActuatorBCParabolicV.angular_size_deg_to_width(
    #         angular_size_deg, params_flow.user_data["D"] / 2
    #     ),
    #     position_x=0.0,
    # )
    # actuator_bc_2 = ActuatorBCParabolicV(
    #     width=ActuatorBCParabolicV.angular_size_deg_to_width(ß
    #         angular_size_deg, params_flow.user_data["D"] / 2
    #     ),
    #     position_x=0.0,
    # )
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
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

    # Find sensor and actuator DoF indices
    V = fs.V  # or whatever your velocity FunctionSpace is
    sensor_position = np.array([3.0, 0.0])
    dof_index = find_sensor_dof_index(V, sensor_position)
    print("Sensor DoF index:", dof_index)

    # dof_indices_1 = find_actuator_dof_indices(fs.V, np.array([0.0, 0.5]), 0.1)
    # dof_indices_2 = find_actuator_dof_indices(fs.V, np.array([0.0, -0.5]), 0.1)
    dof_indices_1 = find_actuator_dof_indices(fs.V, np.array([0.0, 0.5]), 0.5)
    dof_indices_2 = find_actuator_dof_indices(fs.V, np.array([0.0, -0.5]), 0.5)

    print("Actuator 1 DoFs:", dof_indices_1)
    print("Actuator 2 DoFs:", dof_indices_2)

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
    mat = loadmat(str(cwd / 'data_output' / 'eig_data.mat'))
    eig_data = mat['eig_data']
    eigvals = np.array([eig_data[0, i]['lambda'][0, 0] for i in range(eig_data.shape[1])])
    eigvecs = np.array([eig_data[0, i]['vec'].flatten() for i in range(eig_data.shape[1])]).T
    eigvecs_left = np.array([eig_data[0, i]['lvec'].flatten() for i in range(eig_data.shape[1])]).T
    mat_E = loadmat(str(cwd / 'data_output' / 'operators' / 'E_sparse.mat'))
    E_data = mat_E['E_data'].flatten()
    E_indices = mat_E['E_indices'].flatten()
    E_indptr = mat_E['E_indptr'].flatten()
    E_shape = tuple(mat_E['E_shape'].flatten())
    E = csr_matrix((E_data, E_indices, E_indptr), shape=E_shape)
    mat_A = loadmat(str(cwd / 'data_output' / 'operators' / 'A_sparse.mat'))
    A_data = mat_A['A_data'].flatten()
    A_indices = mat_A['A_indices'].flatten()
    A_indptr = mat_A['A_indptr'].flatten()
    A_shape = tuple(mat_A['A_shape'].flatten())
    A = csr_matrix((A_data, A_indices, A_indptr), shape=A_shape)

    # --- Find the most unstable mode ---
    unstable_idx = np.argmax(eigvals.real)
    phi_raw = eigvecs[:, unstable_idx]       # right eigenvector
    psi_raw = eigvecs_left[:, unstable_idx]  # left eigenvector
    lam = eigvals[unstable_idx]

    # --- E-norm normalization and biorthogonal scaling ---
    phi = phi_raw / np.sqrt(np.dot(phi_raw.conj(), E.dot(phi_raw)))  # ||phi||_E = 1
    psi = psi_raw / np.dot(psi_raw.conj(), E.dot(phi))               # psi^H E phi = 1

    # --- Velocity DOF mapping ---
    V_to_W_vel_mapping = np.load(str(cwd / 'data_output' / 'V_to_W_vel_mapping.npy'))

    # --- Assemble actuator forcing vector in mixed space ---
    fs.initialize_time_stepping(ic=None)
    actuator_force_1.expression.u_ctrl = 1.0
    actuator_func_1 = dolfin.interpolate(actuator_force_1.expression, fs.V)

    v = dolfin.TestFunction(fs.V)
    forcing_vec_1 = dolfin.assemble(dolfin.inner(actuator_func_1, v) * dolfin.dx)
    actuator_profile_vels = forcing_vec_1.get_local()

    actuator_profile = np.zeros(A_shape[0])
    actuator_profile[V_to_W_vel_mapping] = actuator_profile_vels

    # --- Compute coupling scalar B correctly ---
    B = np.dot(psi.conj(), actuator_profile)  # psi^H b
    print("Actuator coupling B:", B)

    # --- Feedback gain: place closed-loop eigenvalue ---
    sigma_target = 0.05  # desired decay
    if abs(B.real) > 1e-12:
        K = -(lam.real + sigma_target) / B.real
    else:
        K = -np.real((lam + sigma_target) / B)  # fallback
    print("Feedback gain K:", K)

    # --- Time-stepping loop ---
    for i in range(fs.params_time.num_steps):
        u_current = fs.fields.up_.vector().get_local()
        z = np.dot(psi.conj(), E.dot(u_current))  # modal amplitude
        u_ctrl = np.real(K * z)                     # control input
        fs.step(u_ctrl=np.array([u_ctrl, 0]))
        print(f"Step {i}, z = {z}, control = {u_ctrl}, energy = {fs.compute_energy()}")

    ###################### Full Phase Space Control ######################
    # knots = np.linspace(fs.params_time.Tstart, fs.params_time.Tfinal, 10)
    # values = np.random.uniform(-forcing_amplitude, forcing_amplitude, len(knots))
    # cs = CubicSpline(knots, values)
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

    num_steps_forced = 20000
    forcing_amplitude = 0.3
    forcing_frequency = 1.0

    forced_dir = base_dir / f"Re{Re}_body_force_minimal_nonlin_one_act_no_E" / "run1"
    forced_dir.mkdir(parents=True, exist_ok=True)

    run_forced_simulation(Re, forced_dir, num_steps_forced, forcing_amplitude, forcing_frequency)