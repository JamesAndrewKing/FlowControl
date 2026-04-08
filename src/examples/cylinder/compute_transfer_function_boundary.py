import logging
import time
from pathlib import Path

import dolfin
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as spla
import scipy.io
from dolfin import as_backend_type
import matplotlib.pyplot as plt

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV, ActuatorForceGaussianV, ActuatorForceGaussianAngled
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint
from examples.cylinder.compute_steady_state import Re
from flowcontrol.operatorgetter import OperatorGetter
from examples.cylinder.compute_steady_state import Re

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

def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
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

    params_restart = flowsolverparameters.ParamRestart()

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


    # sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

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

    ##########################################################
    # Extract Actuator Boundary DOFs
    ##########################################################
    print("Extracting actuator boundary DOFs...")
    
    # Get actuator boundary conditions in mixed space (W)
    bcu_actuation_up = dolfin.DirichletBC(
        fs.W.sub(0), dolfin.Constant((0, 0)), fs.get_subdomain("actuator_up"))
    bcu_actuation_lo = dolfin.DirichletBC(
        fs.W.sub(0), dolfin.Constant((0, 0)), fs.get_subdomain("actuator_lo"))
    
    actuator_up_dofs_W = list(bcu_actuation_up.get_boundary_values().keys())
    actuator_lo_dofs_W = list(bcu_actuation_lo.get_boundary_values().keys())
    actuator_dofs_W = np.array(actuator_up_dofs_W + actuator_lo_dofs_W)
    
    # Get actuator boundary conditions in velocity space (V)
    bcu_actuation_up_V = dolfin.DirichletBC(
        fs.V, dolfin.Constant((0, 0)), fs.get_subdomain("actuator_up"))
    bcu_actuation_lo_V = dolfin.DirichletBC(
        fs.V, dolfin.Constant((0, 0)), fs.get_subdomain("actuator_lo"))

    actuator_up_dofs_V = list(bcu_actuation_up_V.get_boundary_values().keys())
    actuator_lo_dofs_V = list(bcu_actuation_lo_V.get_boundary_values().keys())
    actuator_dofs_V = np.array(actuator_up_dofs_V + actuator_lo_dofs_V)

    print(f"Actuator up DOFs W: {len(actuator_up_dofs_W)}")
    print(f"Actuator lo DOFs W: {len(actuator_lo_dofs_W)}")
    print(f"Total actuator DOFs in W space: {len(actuator_dofs_W)}")
    print(f"Actuator up DOFs V: {len(actuator_up_dofs_V)}")
    print(f"Actuator lo DOFs V: {len(actuator_lo_dofs_V)}")
    print(f"Total actuator DOFs in V space: {len(actuator_dofs_V)}")

    V_to_W_vel_mapping = np.load(str(cwd / 'data_output' / 'V_to_W_vel_mapping.npy'))

    # logger.info("Compute steady state...")
    # # U00 = dolfin.Function(fs.V)
    # # P00 = dolfin.Function(fs.P)
    # # steady_state_filename_U0 = params_save.path_out / "steady" / f"U0_Re={Re}.xdmf"
    # # steady_state_filename_P0 = params_save.path_out / "steady" / f"P0_Re={Re}.xdmf"
    # # flu.read_xdmf(steady_state_filename_U0, U00, "U0")
    # # flu.read_xdmf(steady_state_filename_P0, P00, "P0")
    # # initial_guess = fs.merge(U00, P00)
    # uctrl0 = [0.0]
    # fs.compute_steady_state(method="picard", max_iter=20, tol=3e-7, u_ctrl=uctrl0, initial_guess=None)
    # fs.compute_steady_state(
    #     method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    # )
    # or load
    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0.xdmf",
            cwd / "data_output" / "steady" / f"P0.xdmf",
        ]
    )

    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(fs)
    A0 = opget.get_A(UP0=fs.fields.UP0, autodiff=True)
    dim = A0.size(0)  

    # Compute operator: E
    E = opget.get_mass_matrix()

    # Assemble B (sum if multiple actuators with same input)
    # --- Velocity DOF mapping ---
    # V_to_W_vel_mapping = np.load(str(cwd / 'data_output' / 'V_to_W_vel_mapping.npy'))
    # v = dolfin.TestFunction(fs.V)
    # actuator_force_1.expression.u_ctrl = 1.0
    # actuator_func_1 = dolfin.interpolate(actuator_force_1.expression, fs.V)
    # forcing_vec_1 = dolfin.assemble(dolfin.inner(actuator_func_1, v) * dolfin.dx)
    # actuator_profile_vels_1 = forcing_vec_1.get_local()

    # actuator_force_2.expression.u_ctrl = -1.0
    # actuator_func_2 = dolfin.interpolate(actuator_force_2.expression, fs.V)
    # forcing_vec_2 = dolfin.assemble(dolfin.inner(actuator_func_2, v) * dolfin.dx)
    # actuator_profile_vels_2 = forcing_vec_2.get_local()

    # B = np.zeros(dim)
    # B[V_to_W_vel_mapping] = actuator_profile_vels_1 + actuator_profile_vels_2
    # B[V_to_W_vel_mapping] = actuator_profile_vels_1
    # B = actuator_dofs_W
    


    # Find sensor DoF and assemble C
    # sensor_position = np.array([3.0, 0.0])
    # dof_index = find_sensor_dof_index(fs.V, sensor_position)
    # C = np.zeros(dim)
    # C[V_to_W_vel_mapping[dof_index]] = 1.0

    # Partition DOFs
    all_dofs = np.arange(dim)
    interior_dofs = np.setdiff1d(all_dofs, actuator_dofs_W)

    def fenics_to_csr(A):
        """Convert FEniCS PETScMatrix to SciPy CSR matrix."""
        A_petsc = dolfin.as_backend_type(A).mat()
        ai, aj, av = A_petsc.getValuesCSR()
        shape = (A.size(0), A.size(1))
        return spr.csr_matrix((av, aj, ai), shape=shape)

    # Convert before calling transfer_function
    A_csr = fenics_to_csr(A0)
    E_csr = fenics_to_csr(E)

    # Partition matrices
    A_ii = A_csr[interior_dofs[:, None], interior_dofs]
    A_ib = A_csr[interior_dofs[:, None], actuator_dofs_W]
    E_ii = E_csr[interior_dofs[:, None], interior_dofs]
    E_ib = E_csr[interior_dofs[:, None], actuator_dofs_W]

    # B for boundary actuation
    # B_boundary = np.ones(len(actuator_dofs_W))  # TODO: specify true profile!!!
    # Get the full B matrix
    B = opget.get_B()

    # Extract the nonzero entries of B corresponding to the boundary DoFs
    B_boundary = np.sum(B[actuator_dofs_W], axis=1)

    # Output vector
    C = opget.get_C()[0,:]

    C_i = C[interior_dofs]

    # Transfer function
    def boundary_transfer_function(A_ii, E_ii, A_ib, E_ib, B_boundary, C_i, s):
        # rhs = (A_ib - s * E_ib) @ B_boundary
        rhs = A_ib @ B_boundary
        x_i = spla.spsolve(s * E_ii - A_ii, rhs)
        return C_i @ x_i

    # Example usage
    omega = 1.0
    s = 1j * omega
    G = boundary_transfer_function(A_ii, E_ii, A_ib, E_ib, B_boundary, C_i, s)
    print("Boundary transfer function at s = i*omega:", G)

    # Frequency range (log or linear, as appropriate)
    omega_range = np.logspace(-2, 2, 100)  # e.g., from 0.01 to 100

    G_vals = []
    for omega in omega_range:
        s = 1j * omega
        G = boundary_transfer_function(A_ii, E_ii, A_ib, E_ib, B_boundary, C_i, s)
        G_vals.append(G)
        print(f"Frequency (omega): {omega:.4e}, Transfer Function (G): {G.real:.4e} + {G.imag:.4e}j")

    G_vals = np.array(G_vals)

    path_out = cwd / "data_output" / "nyquist"
    path_out.mkdir(parents=True, exist_ok=True)
    np.save(path_out / "G_vals.npy", G_vals)
    np.save(path_out / "omega_range.npy", omega_range)
    scipy.io.savemat(path_out / 'G_vals.mat', {'G_vals': G_vals})
    scipy.io.savemat(path_out / 'omega_range.mat', {'omega_range': omega_range})

    plt.figure(figsize=(6,6))
    plt.plot(G_vals.real, G_vals.imag, label="Nyquist curve")
    plt.plot(G_vals.real, -G_vals.imag, '--', color='gray', label="Negative freq (conj)")
    plt.scatter([-1], [0], color='red', label='-1 (critical point)')
    plt.xlabel('Re(G(iω))')
    plt.ylabel('Im(G(iω))')
    plt.title('Nyquist Plot')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    K = 1.0  # Try different values!
    KG_vals = K * G_vals

    plt.figure(figsize=(6,6))
    plt.plot(KG_vals.real, KG_vals.imag, label=f"Nyquist curve (K={K})")
    plt.scatter([-1], [0], color='red', label='-1 (critical point)')
    plt.xlabel('Re(KG(iω))')
    plt.ylabel('Im(KG(iω))')
    plt.title(f'Nyquist Plot (K={K})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    

    export = False
    if export:
        path_out = cwd / "data_output" / "operators"
        path_out.mkdir(parents=True, exist_ok=True)
        for Mat, Matname in zip([A0, E], ["A", "E"]):
            # Export as png only
            flu.export_sparse_matrix(Mat, path_out / f"{Matname}.png")

            # Export as npz
            Matc, Mats, Matr = Mat.mat().getValuesCSR()
            Acsr = spr.csr_matrix((Matr, Mats, Matc))
            spr.save_npz(path_out / f"{Matname}.npz", Acsr)

            # Export as mat
            scipy.io.savemat(
                path_out / f"{Matname}_sparse.mat",
                {
                    f"{Matname}_data": Acsr.data,
                    f"{Matname}_indices": Acsr.indices,
                    f"{Matname}_indptr": Acsr.indptr,
                    f"{Matname}_shape": Acsr.shape,
                },
            )
        # Export B as vector
        np.save(path_out / "B.npy", B)
        scipy.io.savemat(path_out / "B.mat", {"B": B})
        np.save(path_out / "C.npy", C)
        scipy.io.savemat(path_out / "C.mat",{"C": C})
        # scipy.io.savemat(path_out / 'nyquist/G_vals.mat', {'G_vals': G_vals})
        # scipy.io.savemat(path_out / 'nyquist/omega_range.mat', {'omega_range': omega_range})

    logger.info("Lidcavity -- Finished properly.")
    logger.info("*" * 50)


if __name__ == "__main__":
    main()
