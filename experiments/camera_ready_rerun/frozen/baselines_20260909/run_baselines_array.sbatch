#!/usr/bin/env bash
#SBATCH --job-name=wl-baselines
#SBATCH --partition=requeue
#SBATCH --array=0-6%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/baselines_20260909/logs/%x_%A_%a.out
#SBATCH --error=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/baselines_20260909/logs/%x_%A_%a.err

set -euo pipefail

module load miniconda3
source activate edm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/baselines_20260909
PYTHON_BIN=/home/zwggh/.conda/envs/concept-graphs/bin/python
export PYTHONPATH="${RUN_ROOT}/code/wwl_src:${RUN_ROOT}/code:${RUN_ROOT}/python_deps_py312${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS=(MUTAG PROTEINS IMDB-BINARY IMDB-MULTI PTC_FM PTC_MR COX2)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_NAME=$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')
EXTRACT_ROOT=${RUN_ROOT}/scratch/${OUTPUT_NAME}

mkdir -p "${EXTRACT_ROOT}" "${RUN_ROOT}/outputs/${OUTPUT_NAME}" "${RUN_ROOT}/logs"
unzip -oq "${RUN_ROOT}/data/${DATASET}.zip" -d "${EXTRACT_ROOT}"

"${PYTHON_BIN}" "${RUN_ROOT}/code/run_baselines.py" \
  --dataset-dir "${EXTRACT_ROOT}/${DATASET}" \
  --dataset-name "${DATASET}" \
  --output-dir "${RUN_ROOT}/outputs/${OUTPUT_NAME}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --seed 20260909
