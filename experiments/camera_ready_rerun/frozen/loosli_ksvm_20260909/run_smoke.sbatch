#!/usr/bin/env bash
#SBATCH --job-name=wl-loosli-smoke
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909/logs/%x_%j.out
#SBATCH --error=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909/logs/%x_%j.err

set -euo pipefail

module load miniconda3
source activate edm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909
PYTHON_BIN=/home/zwggh/.conda/envs/concept-graphs/bin/python
"${PYTHON_BIN}" "${RUN_ROOT}/code/smoke_test.py"
