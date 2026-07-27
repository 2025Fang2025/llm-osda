#!/usr/bin/env bash
set -e
export TORCH_NUM_THREADS=4

# Multi-SEED experiment for statistically meaningful mean±std.
# Each seed gets its OWN output_dir so data_splits.json is NOT reused across
# seeds (different seed -> different train/test split -> genuine repetition).
# Each run is single-round (train_rounds 1) — no iterative refinement, so the
# variance reflects seed sensitivity, not round-to-round progression.
# Parameters mirror run_qwen3emb.sh (the paper's main configuration).

BASE_DIR="outputs/multi_seed"
mkdir -p "$BASE_DIR"

SEEDS="42 123 456"

echo "=== Multi-seed experiment (seeds: ${SEEDS}) for mean±std ==="

for SEED in ${SEEDS}; do
  OUT_DIR="${BASE_DIR}/seed_${SEED}"
  mkdir -p "$OUT_DIR"
  LOG_FILE="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
  echo ""
  echo "=============================="
  echo "  seed = ${SEED}  ->  ${OUT_DIR}"
  echo "=============================="

  python -u -m src.run_final_experiment \
    --mode all \
    --dialogues "data/dialogues_all.jsonl" \
    --counterfactuals "data/counterfactuals_all.jsonl" \
    --sft "data/sft_all.jsonl" \
    --output_dir "${OUT_DIR}" \
    --base_model Qwen/Qwen3-4B \
    --text_encoder qwen3_embedding \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --seed ${SEED} \
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
    --train_rounds 1 \
    --eval_rounds 1 \
    --use_vllm \
    --include_ablation \
    2>&1 | tee "$LOG_FILE"
done

echo ""
echo "=== Aggregating mean±std across seeds ==="
python3 - <<'PYEOF'
import glob, pandas as pd
files = sorted(glob.glob("outputs/multi_seed/seed_*/final_evaluation_summary.csv"))
print(f"  found {len(files)} seed summaries: {files}")
if not files:
    raise SystemExit("  No summaries found — did the runs finish?")
dfs = []
for f in files:
    df = pd.read_csv(f)
    df["seed_file"] = f
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
# Aggregate only numeric columns by policy; n_sessions etc. averaged too.
num_cols = combined.select_dtypes("number").columns.drop("round", errors="ignore")
agg = combined.groupby("policy")[list(num_cols)].agg(["mean", "std"])
agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
agg = agg.reset_index()
out = "outputs/multi_seed/aggregated_mean_std.csv"
agg.to_csv(out, index=False)
print(f"  wrote {out}")
# Print the headline reward comparison
cols = [c for c in ["policy", "avg_reward_mean", "avg_reward_std",
                    "avg_revenue_mean", "avg_revenue_std"] if c in agg.columns]
print(agg[cols].to_string(index=False))
PYEOF

echo ""
echo "=== Done. Per-seed: ${BASE_DIR}/seed_*/  |  Aggregated: ${BASE_DIR}/aggregated_mean_std.csv ==="
