#!/usr/bin/env bash
set -euo pipefail
export TORCH_NUM_THREADS=4

# ============================================================================
# Major-5 adaptive baseline: myopic threshold stopping (greedy, no training).
# Reuses the already-trained Round-2 models (G_eta / renderer / judge).
# Runs --mode eval only -> NO retraining. Sweeps a few thresholds; pick the
# one with the highest Reward on the eval set as the reported adaptive baseline.
#
# The myopic policy stops at the first turn whose winner expected click value
# b_i * q_i >= MYOPIC_THRESHOLD (read from env by closed_loop_eval.py).
# ============================================================================

# --- EDIT THESE to match your trained run ---
MODEL_DIR="outputs/final_qwen3emb/round_2"      # trained Round-2 models
SPLITS_DIR="outputs/final_qwen3emb"             # data_splits.json location
BASE_OUT="outputs/myopic_sweep"
# --------------------------------------------

mkdir -p "$BASE_OUT"

# b_i * q_i thresholds to sweep (bid up to ~4, pCTR in [0,1] -> product spans ~0..3)
THRESHOLDS="0.2 0.3 0.4 0.5 0.6"

echo "=== Myopic threshold baseline sweep (eval only, no retraining) ==="
for THR in ${THRESHOLDS}; do
  OUT_DIR="${BASE_OUT}/thr_${THR}"
  mkdir -p "$OUT_DIR"
  # reuse trained splits + models: copy split file so eval finds it
  cp "${SPLITS_DIR}/data_splits.json" "${OUT_DIR}/" 2>/dev/null || true
  echo ""
  echo "=============================="
  echo "  MYOPIC_THRESHOLD = ${THR}  ->  ${OUT_DIR}"
  echo "=============================="

  MYOPIC_THRESHOLD=${THR} python -u -m src.run_final_experiment \
    --mode eval \
    --model_dir "${MODEL_DIR}" \
    --dialogues "data/dialogues_all.jsonl" \
    --counterfactuals "data/counterfactuals_all.jsonl" \
    --sft "data/sft_all.jsonl" \
    --output_dir "${OUT_DIR}" \
    --base_model Qwen/Qwen3-4B \
    --text_encoder qwen3_embedding \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --seed 42 \
    --discount 0.95 \
    --lambda_gamma 0.25 \
    --max_eval_sessions 0 \
    --eval_batch_size 64 \
    --eval_rounds 1 \
    --use_vllm \
    2>&1 | tee "${OUT_DIR}/eval.log"
done

echo ""
echo "=== Sweep done. Compare myopic_threshold Reward across thresholds: ==="
python3 - <<'PYEOF'
import glob, pandas as pd, os
rows = []
for f in sorted(glob.glob("outputs/myopic_sweep/thr_*/**/final_evaluation_summary.csv", recursive=True)):
    thr = f.split("thr_")[1].split("/")[0]
    df = pd.read_csv(f)
    m = df[df["policy"] == "myopic_threshold"]
    if len(m):
        r = m.iloc[0]
        rows.append({"threshold": thr,
                     "reward": r.get("avg_reward", float("nan")),
                     "net_rev": r.get("avg_revenue", float("nan")),
                     "trigger": r.get("avg_trigger_turn", r.get("trigger", float("nan")))})
if rows:
    out = pd.DataFrame(rows).sort_values("threshold")
    print(out.to_string(index=False))
    best = out.loc[out["reward"].idxmax()]
    print(f"\n>>> Best threshold by Reward: {best['threshold']} (reward={best['reward']:.4f})")
else:
    print("No myopic_threshold rows found — check eval logs / summary paths.")
PYEOF

echo ""
echo "=== Use the best-threshold run's myopic_threshold row as the adaptive baseline. ==="
