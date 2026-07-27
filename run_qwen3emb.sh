#!/usr/bin/env bash
set -e
export TORCH_NUM_THREADS=4

LOG_DIR="outputs/final_qwen3emb"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "=== Qwen3-Embedding Experiment started at $(date) ==="
echo "=== Log file: $LOG_FILE ==="

python -u -m src.run_final_experiment \
  --mode all \
  --dialogues "data/dialogues_all.jsonl" \
  --counterfactuals "data/counterfactuals_all.jsonl" \
  --sft "data/sft_all.jsonl" \
  --output_dir outputs/final_qwen3emb \
  --base_model Qwen/Qwen3-4B \
  --text_encoder qwen3_embedding \
  --embedding_model Qwen/Qwen3-Embedding-0.6B \
  --pctr_epochs 5 \
  --judge_epochs 5 \
  --stop_epochs 30 \
  --patience 1 \
  --lr 1e-3 \
  --batch_size 256 \
  --discount 0.95 \
  --lambda_gamma 0.25 \
  --llm_max_samples 0 \
  --llm_epochs 1 \
  --llm_max_length 8192 \
  --llm_batch_size 4 \
  --llm_grad_accum 32 \
  --llm_lr 2e-5 \
  --llm_max_new_tokens 128 \
  --max_eval_sessions 0 \
  --eval_batch_size 64 \
  --train_rounds 2 \
  --eval_rounds 2 \
  --use_vllm \
  --include_ablation \
  2>&1 | tee "$LOG_FILE"

echo "=== Finished at $(date) ==="
echo "=== Log: $LOG_FILE ==="
