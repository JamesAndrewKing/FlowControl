#!/usr/bin/env bash
# Create the Linux FEniCS environment used by the BFS Euler jobs.

set -euo pipefail

: "${SCRATCH:?SCRATCH is not defined; run this script on Euler}"

repo_root="$(git rev-parse --show-toplevel)"
environment_prefix="${FLOWCONTROL_ENV:-${SCRATCH}/flowcontrol-fenics}"

if [[ "$(git -C "${repo_root}" branch --show-current)" != "feature/bfs" ]]; then
    echo "Expected the feature/bfs branch in ${repo_root}" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available. Load or install Miniforge/Miniconda first." >&2
    exit 2
fi

if [[ ! -x "${environment_prefix}/bin/python" ]]; then
    conda env create \
        --prefix "${environment_prefix}" \
        --file "${repo_root}/environment.yml"
else
    echo "Using existing environment ${environment_prefix}"
fi

source "${environment_prefix}/bin/activate"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
python -c "import dolfin, petsc4py, slepc4py; print(dolfin.__version__)"

mkdir -p "${repo_root}/slurm_logs"
echo "Euler setup is ready. Stage the fixed_points directory before submitting."
