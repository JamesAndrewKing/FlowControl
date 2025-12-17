from examples.lidcavity.compute_steady_state_increasing_Re import Re_final as Re
from examples.lidcavity.batch_run_lidcavity_eigvecs import run_lidcavity_with_eigenvector_ic
from pathlib import Path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def run_single_simulation(args):
    """Wrapper function for a single simulation run, designed for parallel processing."""
    eigenvector_coefficients, forcing_frequency, forcing_amplitude, Re, save_dir, num_steps = args
    
    print(f"Starting simulation in process {mp.current_process().name}: freq={forcing_frequency}, f_amp={forcing_amplitude:.4f}, ic_norm={np.linalg.norm(eigenvector_coefficients):.4f}")
    try:
        run_lidcavity_with_eigenvector_ic(Re, eigenvector_coefficients, forcing_frequency, forcing_amplitude, save_dir, num_steps)
        print(f"✓ Completed simulation in {save_dir.name}")
        return True
    except Exception as e:
        print(f"✗ Error in simulation {save_dir.name}: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path("/Users/jaking/Desktop/PhD/lid_driven_cavity")
    parent_dir = base_dir / f"Re{Re}_test_8800_new"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # --- Define Parameters for Iteration ---
    num_steps = 60000

    # 1. Forcing parameters
    # forcing_frequencies = np.linspace(1, 5, 9)
    # forcing_amplitudes = [0.01]
    forcing_frequencies = [0.0]
    forcing_amplitudes = [0.0]

    # 2. Initial condition parameters
    eigenvector_amplitude = 0.001 # Post hopf
    # eigenvector_amplitude = 0.05 # Pre hopf

    coefficient_directions = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    # coefficient_directions = [
    #     [1, 1, 0, 0],
    # ]

    # --- Pre-calculate all normalized coefficient sets ---
    normalized_coefficient_sets = []
    for d in coefficient_directions:
        direction_vec = np.array(d, dtype=float)
        norm = np.linalg.norm(direction_vec)
        if norm > 1e-10:
            normalized_coeffs = (eigenvector_amplitude * direction_vec / norm).tolist()
        else:
            normalized_coeffs = d
        normalized_coefficient_sets.append(normalized_coeffs)

    # --- Build the list of simulation arguments ---
    simulation_args = []
    run_count = 1
    for freq in forcing_frequencies:
        for f_amp in forcing_amplitudes:
            for coeffs in normalized_coefficient_sets:
                save_dir = parent_dir / f"run{run_count}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                log_path = save_dir / "run_parameters.txt"
                with open(log_path, "w") as f:
                    f.write(f"Run directory: {save_dir}\n")
                    f.write(f"Reynolds number: {Re}\n")
                    f.write(f"Eigenvector coefficients: {coeffs}\n")
                    f.write(f"Coefficient norm: {np.linalg.norm(coeffs)}\n")
                    f.write(f"Forcing frequency: {freq}\n")
                    f.write(f"Forcing amplitude: {f_amp}\n")
                    f.write(f"Number of steps: {num_steps}\n")
                    f.write(f"Run index: {run_count}\n")

                simulation_args.append((coeffs, freq, f_amp, Re, save_dir, num_steps))
                run_count += 1

    # --- Run simulations in parallel ---
    n_processes = min(mp.cpu_count() - 2, len(simulation_args))
    print(f"Running {len(simulation_args)} simulations using {n_processes} processes")

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, simulation_args),
            total=len(simulation_args),
            desc="Running simulations"
        ))
    
    # --- Report results ---
    successful = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Parallel execution completed!")
    print(f"Successful simulations: {successful}/{total}")
    print(f"Failed simulations: {total - successful}")
    
    if total - successful > 0:
        print(f"\nFailed simulation details:")
        for i, (success, args) in enumerate(zip(results, simulation_args)):
            if not success:
                coeffs, freq, f_amp, _, _, _ = args
                print(f"  - Run {i+1}: freq={freq}, f_amp={f_amp}, coeffs={np.array(coeffs).round(4).tolist()}")
    
    print(f"{'='*60}")