export EXP_NAME="test_simple_torchdata_with_api_1_full_update"
python main.py \
  --run_name  $EXP_NAME \
  --root_dir "root" \
  --dataset_path /ML-A100/home/gujiasheng/reflexion/programming_runs/benchmarks/real_torchdata_eval_v3_api_5.jsonl \
  --strategy "simple" \
  --language "py" \
  --model "gpt-3.5-turbo" \
  --pass_at_k "1" \
  --max_iters "1" \
  --verbose > $EXP_NAME.log 2>&1 &
