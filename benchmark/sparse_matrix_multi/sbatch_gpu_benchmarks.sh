#!/usr/bin/env bash
# Submit sparse-matrix GPU benchmarks to Slurm.
# Run from the repo root so paths resolve. Logs: slurm-pdst-mm-<jobid>.{out,err} in cwd.
#
#   cd ~/Parallel-Dynamic-Sparse-Training
#   sbatch benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh
#
# Defaults below match a typical main + Lovelace GPU job (partition main, 1× lovelace, dmb, research).
# If your account cannot use qos=research or account=dmb, comment those lines out or override:
#   sbatch --partition=dmb --gres=gpu:1 benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh
#
# Override resources or methods (default: test_sparse_ut test_sputnik):
#   METHODS="test_sparse_cupy test_jax" sbatch benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh
#   sbatch --partition=main-gpu --time=04:00:00 benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh
#
# Run on an interactive allocation (no Slurm submit):
#   bash benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh
#
#SBATCH --job-name=pdst-mm-bench
#SBATCH --partition=main
#SBATCH --account=dmb
#SBATCH --qos=research
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=6G
#SBATCH --time=02:00:00
# Logs in current working directory (must exist). Do not use benchmark/logs/ here unless
# that directory already exists before Slurm starts — otherwise the job can fail immediately.
#SBATCH --output=slurm-pdst-mm-%j.out
#SBATCH --error=slurm-pdst-mm-%j.err

set -euo pipefail

# Slurm copies this script under /tmp/slurmd/... so BASH_SOURCE is useless for finding the repo.
# SLURM_SUBMIT_DIR is the directory you were in when you ran sbatch (use repo root).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
  REPO_ROOT="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  cd "$REPO_ROOT"
fi

exec bash "${REPO_ROOT}/benchmark/sparse_matrix_multi/run_benchmarks_gpu.sh"
