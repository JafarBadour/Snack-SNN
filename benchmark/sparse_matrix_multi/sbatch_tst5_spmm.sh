#!/usr/bin/env bash
# Submit tst5 (SpMM correctness in test.py) to Slurm from the head node.
# Logs: slurm-pdst-tst5-<jobid>.{out,err} in the directory where you ran sbatch.
#
#   cd ~/Parallel-Dynamic-Sparse-Training
#   sbatch benchmark/sparse_matrix_multi/sbatch_tst5_spmm.sh
#
# Defaults match sbatch_gpu_benchmarks.sh (main, lovelace, dmb, research). Override if needed.
#
# Run locally on a GPU box without Slurm (same as bash run_tst5_spmm.sh):
#   bash benchmark/sparse_matrix_multi/sbatch_tst5_spmm.sh
#
#SBATCH --job-name=pdst-tst5-spmm
#SBATCH --partition=main
#SBATCH --account=dmb
#SBATCH --qos=research
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-pdst-tst5-%j.out
#SBATCH --error=slurm-pdst-tst5-%j.err

set -euo pipefail

# Slurm copies this script to a temp path; use submit cwd as repo root (same as sbatch_gpu_benchmarks.sh).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
  REPO_ROOT="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  cd "$REPO_ROOT"
fi

exec bash "${REPO_ROOT}/benchmark/sparse_matrix_multi/run_tst5_spmm.sh"
