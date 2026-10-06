# MVPruner on DriveLMM-o1

This release contains the inference code for reproducing MVPruner on the DriveLMM-o1 benchmark.

## Setup

Run the following commands from the `drivelmm-o1/` directory:

```bash
conda create -n mvpruner-drivelmm python=3.10 -y
conda activate mvpruner-drivelmm
pip install -r requirements.txt
```

## Checkpoint

Download the DriveLMM-o1 checkpoint, then install the MVPruner remote-code files into that local checkpoint directory:

```bash
huggingface-cli download ayeshaishaq/DriveLMMo1 --local-dir ckpt/DriveLMMo1-MVPruner
python scripts/prepare_model_code.py --model-dir ckpt/DriveLMMo1-MVPruner
```

The prepare script copies the files in `modeling/` into the checkpoint directory so that `trust_remote_code=True` loads MVPruner instead of the original DriveLMM-o1 model code.

## Data

Prepare the DriveLMM-o1 test annotations:

```text
DriveLMMo1_TEST.jsonl
DriveLMMo1_TEST.json
```

`DriveLMMo1_TEST.jsonl` stores the question/answer conversation. `DriveLMMo1_TEST.json` stores the six nuScenes camera image paths for each sample. The image root should point to the nuScenes directory that contains `samples/CAM_FRONT`, `samples/CAM_FRONT_LEFT`, and the other camera folders.

## Inference

90% pruning:

```bash
IMAGE_ROOT=/path/to/nuscenes bash scripts/run_inference.sh 0 0.9 mvpruner_90
```

75% pruning:

```bash
IMAGE_ROOT=/path/to/nuscenes bash scripts/run_inference.sh 0 0.75 mvpruner_75
```

The output is written to:

```text
results/DriveLMMo1_mvpruner_90.json
results/DriveLMMo1_mvpruner_75.json
```


## Evaluation

The GPT-as-judge evaluation script computes the DriveLMM-o1 reasoning metrics and MCQ accuracy from an inference output file. Provide the API key through an environment variable instead of editing the script:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.openai.com/v1

python evaluation/evaluation_script.py \
  --prediction results/DriveLMMo1_mvpruner_90.json \
  --dataset DriveLMMo1_TEST.json \
  --output-json results/DriveLMMo1_mvpruner_90_eval.json \
  --output-csv results/DriveLMMo1_mvpruner_90_eval.csv \
  --summary-json results/DriveLMMo1_mvpruner_90_summary.json \
  --model gpt-4o-mini \
  --num-workers 4
```

For a quick format check, add `--max-samples 1`.

## MVPruner Configurations

The two released configurations share one implementation in `modeling/modeling_internlm2_mvpruner.py`.

| pruning ratio | stage keep ratio | stage 1 layer | attention layer | stage 2 layer | tokens per view |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.167 | 0 | 15 | 16 | 256 |
| 0.75 | 0.3660254 | 0 | 15 | 16 | 256 |

## Acknowledgements

This code builds on DriveLMM-o1 and InternVL.
