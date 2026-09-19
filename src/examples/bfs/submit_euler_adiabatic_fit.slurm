#!/usr/bin/env bash
# Fit M0/M1/M2 after every DNS array task has completed successfully.

#SBATCH --job-name=bfs_adiabatic_fit
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

: "${SCRATCH:?SCRATCH is not defined}"

flowcontrol_root="${FLOWCONTROL_ROOT:-${SCRATCH}/FlowControl}"
environment_prefix="${FLOWCONTROL_ENV:-${SCRATCH}/flowcontrol-fenics}"
results_root="${BFS_ADIABATIC_RESULTS_ROOT:-${SCRATCH}/AdiabaticFlowControl/data/bfs}"

source "${environment_prefix}/bin/activate"
export PYTHONPATH="${flowcontrol_root}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${flowcontrol_root}"
python src/examples/bfs/fit_adiabatic_models.py \
    --input-root "${results_root}/adiabatic/bfs_3"
