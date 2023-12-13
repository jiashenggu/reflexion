#!/bin/bash

dataset_names=("numpy" "pandas" "torchdata")
for dataset_name in "${dataset_names[@]}"
do
    # 对每个 dataset_name 执行操作
    echo "正在处理数据集: $dataset_name"
    # 在这里添加你的命令，例如处理或分析数据集
    export EXP_NAME="test_reflexion_${dataset_name}_with_api_1_full_update_embedding"
    python main.py \
      --run_name  $EXP_NAME \
      --root_dir "root" \
      --dataset_path /ML-A100/home/gujiasheng/reflexion/programming_runs/benchmarks/real_${dataset_name}_eval_v3_api_5.jsonl \
      --strategy "reflexion" \
      --language "py" \
      --model "gpt-3.5-turbo" \
      --pass_at_k "1" \
      --max_iters "2" \
      --verbose \
      > $EXP_NAME.log 2>&1 &
done
dataset_name=$1


