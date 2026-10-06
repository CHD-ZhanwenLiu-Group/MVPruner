#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
PRUNING_RATIO=${2:-0.9}
OUTPUT_NAME=${3:-mvpruner_90}
MODEL_PATH=${MODEL_PATH:-ckpt/DriveLMMo1-MVPruner}
DATA_JSONL=${DATA_JSONL:-DriveLMMo1_TEST.jsonl}
META_JSON=${META_JSON:-DriveLMMo1_TEST.json}
IMAGE_ROOT=${IMAGE_ROOT:-data/nuscenes}
OUTPUT_DIR=${OUTPUT_DIR:-results}

CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/inference.py   --model-path "${MODEL_PATH}"   --data-jsonl "${DATA_JSONL}"   --meta-json "${META_JSON}"   --image-root "${IMAGE_ROOT}"   --output-dir "${OUTPUT_DIR}"   --output-name "${OUTPUT_NAME}"   --pruning-ratio "${PRUNING_RATIO}"
