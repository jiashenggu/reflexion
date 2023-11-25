python main.py \
  --run_name "test_reflexion_pandas_with_api_update_1" \
  --root_dir "root" \
  --dataset_path /home/gujiasheng/reflexion/programming_runs/benchmarks/real_pandas_eval_v3_api_5.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-3.5-turbo" \
  --pass_at_k "1" \
  --max_iters "2" \
  --verbose
