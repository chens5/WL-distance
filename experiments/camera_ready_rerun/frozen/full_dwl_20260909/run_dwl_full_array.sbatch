#!/usr/bin/env bash
#SBATCH --job-name=wl-dwl-full
#SBATCH --partition=requeue
#SBATCH --array=0-13%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_dwl_20260909/logs/%x_%A_%a.out
#SBATCH --error=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_dwl_20260909/logs/%x_%A_%a.err

set -euo pipefail

module load miniconda3
source activate edm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_dwl_20260909
PYTHON_BIN=/home/zwggh/.conda/envs/concept-graphs/bin/python
export PYTHONPATH="${RUN_ROOT}/python_deps_py312${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS=(MUTAG PROTEINS IMDB-BINARY IMDB-MULTI PTC_FM PTC_MR COX2)
LABELS=(f1 f2)
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
LABEL_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
DATASET=${DATASETS[$DATASET_INDEX]}
LABEL_MODE=${LABELS[$LABEL_INDEX]}
OUTPUT_NAME=$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')
EXTRACT_ROOT=${RUN_ROOT}/scratch/${OUTPUT_NAME}_${LABEL_MODE}

mkdir -p "${EXTRACT_ROOT}" "${RUN_ROOT}/outputs/${OUTPUT_NAME}_${LABEL_MODE}" "${RUN_ROOT}/logs"
unzip -oq "${RUN_ROOT}/data/${DATASET}.zip" -d "${EXTRACT_ROOT}"

"${PYTHON_BIN}" "${RUN_ROOT}/code/diagnose_dwl.py" \
  --dataset-dir "${EXTRACT_ROOT}/${DATASET}" \
  --dataset-name "${DATASET}" \
  --output-dir "${RUN_ROOT}/outputs/${OUTPUT_NAME}_${LABEL_MODE}" \
  --label-mode "${LABEL_MODE}" \
  --max-k 4 \
  --q 0.6 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --checkpoint-every 1000 \
  --outer-splits 10 \
  --raw-inner-splits 10 \
  --wtk-inner-splits 5 \
  --seed 20260909
