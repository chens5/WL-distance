#!/usr/bin/env bash
#SBATCH --job-name=wl-loosli-dwl
#SBATCH --partition=requeue
#SBATCH --array=0-55%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909/logs/%x_%A_%a.out
#SBATCH --error=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909/logs/%x_%A_%a.err

set -euo pipefail

module load miniconda3
source activate edm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/loosli_ksvm_20260909
SOURCE_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_dwl_20260909
DATA_ROOT=/cluster/pixstor/zwggh-lab/zhengchao/wl_distance_camera_ready/full_wllb_f1_20260909/data
PYTHON_BIN=/home/zwggh/.conda/envs/concept-graphs/bin/python

DATASETS=(MUTAG PROTEINS PTC_FM PTC_MR IMDB-BINARY IMDB-MULTI COX2)
LABELS=(f1 f2)
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 8))
LABEL_INDEX=$((REMAINDER / 4))
K=$((REMAINDER % 4 + 1))
DATASET=${DATASETS[$DATASET_INDEX]}
LABEL_MODE=${LABELS[$LABEL_INDEX]}
OUTPUT_NAME=$(printf '%s' "${DATASET}" | tr '[:upper:]_' '[:lower:]-')
SOURCE_DIR=${SOURCE_ROOT}/outputs/${OUTPUT_NAME}_${LABEL_MODE}
OUTPUT_DIR=${RUN_ROOT}/outputs/dwl/${OUTPUT_NAME}_${LABEL_MODE}_k${K}

mkdir -p "${OUTPUT_DIR}" "${RUN_ROOT}/logs"
"${PYTHON_BIN}" "${RUN_ROOT}/code/run_ksvm_matrix.py" \
  --distance-path "${SOURCE_DIR}/${DATASET}_dwl_${LABEL_MODE}_k${K}.npy" \
  --source-summary "${SOURCE_DIR}/summary_k${K}.json" \
  --data-archive "${DATA_ROOT}/${DATASET}.zip" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --distance-kind dwl \
  --label-mode "${LABEL_MODE}" \
  --depth "${K}" \
  --q 0.6 \
  --outer-splits 10 \
  --inner-splits 10 \
  --seed 20260909 \
  --workers "${SLURM_CPUS_PER_TASK}"
