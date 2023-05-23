python main.py \
  --run_name "reflexion_ucs_plays_with_the_ferris_crab_2" \
  --root_dir "root" \
  --dataset_path ./benchmarks/human_eval_rs.jsonl \
  --strategy "reflexion-ucs" \
  --language "rs" \
  --model "gpt-4" \
  --pass_at_k 1 \
  --max_iters 5 \
  --expansion_factor 3 \
  --verbose