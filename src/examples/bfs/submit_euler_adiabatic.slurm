#!/usr/bin/env bash
# Euler array for the BFS deterministic-adiabatic campaign.
#
# Full campaign (array 0-4):
#   sbatch src/examples/bfs/submit_euler_adiabatic.slurm
#
# Short time-step pilot (array 0-3, dt = 0.005, 0.01, 0.02, 0.04):
#   sbatch --time=48:00:00 --array=0-3 \
#     --export=ALL,BFS_ADIABATIC_MODE=timestep \
#     src/examples/bfs/submit_euler_adiabatic.slurm

#SBATCH --job-name=bfs_adiabatic
#SBATCH --array=0-4
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=2G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

: "${SCRATCH:?SCRATCH is not defined}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as a job array}"

flowcontrol_root="${FLOWCONTROL_ROOT:-${SCRATCH}/FlowControl}"
environment_prefix="${FLOWCONTROL_ENV:-${SCRATCH}/flowcontrol-fenics}"
results_root="${BFS_ADIABATIC_RESULTS_ROOT:-${SCRATCH}/AdiabaticFlowControl/data/bfs}"
mode="${BFS_ADIABATIC_MODE:-full}"

if [[ ! -x "${environment_prefix}/bin/python" ]]; then
    echo "Missing environment ${environment_prefix}; run setup_euler.sh" >&2
    exit 2
fi
if [[ ! -d "${flowcontrol_root}/.git" ]]; then
    echo "Missing FlowControl clone at ${flowcontrol_root}" >&2
    exit 2
fi

fixed_points="${flowcontrol_root}/src/examples/bfs/data_output/bfs_3/fixed_points"
if [[ ! -d "${fixed_points}" ]]; then
    echo "Missing fixed-point directory ${fixed_points}" >&2
    exit 2
fi
fixed_point_count="$(find "${fixed_points}" -mindepth 1 -maxdepth 1 \
    -type d -name 'a_*' | wc -l | tr -d ' ')"
if [[ "${fixed_point_count}" != "21" ]]; then
    echo "Expected 21 fixed points in ${fixed_points}; found ${fixed_point_count}" >&2
    exit 2
fi

source "${environment_prefix}/bin/activate"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PYTHONPATH="${flowcontrol_root}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIJITSO_CACHE_DIR="${TMPDIR}/dijitso-cache"

cd "${flowcontrol_root}"
echo "FlowControl commit: $(git rev-parse HEAD)"
echo "Mode: ${mode}; array index: ${SLURM_ARRAY_TASK_ID}"

common_args=(
    --mesh bfs_3
    --reynolds 500
    --observation-dt 1.0
)

if [[ "${mode}" == "timestep" ]]; then
    case "${SLURM_ARRAY_TASK_ID}" in
        0) dt="0.005" ;;
        1) dt="0.01" ;;
        2) dt="0.02" ;;
        3) dt="0.04" ;;
        *) echo "Time-step mode requires array index 0-3" >&2; exit 2 ;;
    esac
    pilot_root="${results_root}/time_step_pilot/dt_${dt/./p}"
    python src/examples/bfs/run_deterministic_adiabatic.py \
        "${common_args[@]}" \
        --output-root "${pilot_root}" \
        --split test --seed 999 --epsilon 0.2 \
        --cycles 0.1 --dt "${dt}" --field-dt 0
elif [[ "${mode}" == "full" ]]; then
    dt="${BFS_ADIABATIC_DT:-0.01}"
    field_dt="${BFS_ADIABATIC_FIELD_DT:-10.0}"
    case "${SLURM_ARRAY_TASK_ID}" in
        0) split="train"; seed="1"; epsilon="0.1" ;;
        1) split="train"; seed="4"; epsilon="0.1" ;;
        2) split="test"; seed="101"; epsilon="0.2" ;;
        3) split="test"; seed="101"; epsilon="0.1" ;;
        4) split="test"; seed="101"; epsilon="0.05" ;;
        *) echo "Full mode requires array index 0-4" >&2; exit 2 ;;
    esac
    python src/examples/bfs/run_deterministic_adiabatic.py \
        "${common_args[@]}" \
        --output-root "${results_root}" \
        --split "${split}" --seed "${seed}" --epsilon "${epsilon}" \
        --cycles 0.5 --dt "${dt}" --field-dt "${field_dt}"
else
    echo "Unknown BFS_ADIABATIC_MODE=${mode}" >&2
    exit 2
fi
