#!/usr/bin/env bash
#SBATCH --job-name=wl-wllb-f1
#SBATCH --partition=requeue
#SBATCH --array=0-27%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_wllb_f1_20260909/logs/%x_%A_%a.out
#SBATCH --error=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_wllb_f1_20260909/logs/%x_%A_%a.err

set -euo pipefail

module load miniconda3
source activate edm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_wllb_f1_20260909
PYTHON_BIN=/home/zwggh/.conda/envs/concept-graphs/bin/python
export PYTHONPATH="${RUN_ROOT}/python_deps_py312${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS=(MUTAG PROTEINS IMDB-BINARY IMDB-MULTI PTC_FM PTC_MR COX2)
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
K=$((SLURM_ARRAY_TASK_ID % 4 + 1))
DATASET=${DATASETS[$DATASET_INDEX]}
OUTPUT_NAME=$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')

EXTRACT_ROOT=${RUN_ROOT}/scratch/${OUTPUT_NAME}_k${K}
mkdir -p "${EXTRACT_ROOT}" "${RUN_ROOT}/outputs/${OUTPUT_NAME}_k${K}" "${RUN_ROOT}/logs"
unzip -oq "${RUN_ROOT}/data/${DATASET}.zip" -d "${EXTRACT_ROOT}"

"${PYTHON_BIN}" "${RUN_ROOT}/code/diagnose_wllb_ksvm.py" \
  --dataset-dir "${EXTRACT_ROOT}/${DATASET}" \
  --dataset-name "${DATASET}" \
  --output-dir "${RUN_ROOT}/outputs/${OUTPUT_NAME}_k${K}" \
  --label-mode f1 \
  --k "${K}" \
  --q 0.6 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --outer-splits 10 \
  --raw-inner-splits 10 \
  --wtk-inner-splits 5 \
  --seed 20260909
