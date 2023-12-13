#!/bin/bash

dataset_name=$1
export EXP_NAME="test_reflexion_${dataset_name}_gpt4"
python main.py \
  --run_name  $EXP_NAME \
  --root_dir "root" \
  --dataset_path /ML-A100/home/gujiasheng/reflexion/programming_runs/benchmarks/real_${dataset_name}_eval_v3_api_5.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-4" \
  --pass_at_k "1" \
  --max_iters "2" \
  --verbose \
  > $EXP_NAME.log 2>&1 &

