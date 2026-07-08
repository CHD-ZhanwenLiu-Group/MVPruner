#!/bin/bash

set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_inference.sh GPU_ID PRUNING_RATIO OUTPUT_NAME"
  echo "Example: bash run_inference.sh 0 0.9 drivelm_mvpruner_90"
  echo "Example: bash run_inference.sh 0 0.75 drivelm_mvpruner_75"
  exit 1
fi

CUDA_DEVICES=$1
PRUNING_RATIO=$2
OUTPUT_NAME=$3

timestamp=$(TZ=Asia/Shanghai date +"%Y%m%d_%H%M%S")
output_dir="output/${OUTPUT_NAME}/${timestamp}"
mkdir -p "$output_dir"

exec > >(tee -i "$output_dir/run_all.log") 2>&1
cp inference.py "$output_dir"
cp "$0" "$output_dir/run_all.sh"

echo "Created folder: $output_dir"
echo "Will use GPUs: ${CUDA_DEVICES}"
echo "MVPruner pruning ratio: ${PRUNING_RATIO}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
python inference.py \
  --model-path ../ckpt/DriveMM \
  --data ./test_llama_eval.json \
  --pruning-ratio "${PRUNING_RATIO}" \
  --output "$output_dir/drivemm_original.json"

echo "Processing json2array..."
python json2array.py \
  --input "$output_dir/drivemm_original.json" \
  --output "$output_dir/drivemm_original_list.json"

echo "Preparing submission..."
python prepare_submission.py \
  --input "$output_dir/drivemm_original_list.json" \
  --output "$output_dir/submission.json"

echo "All steps finished!"
