#!/usr/bin/env bash
# Route B+ Envelope Payment — 全量运行脚本
# 在 GPU 上对全部 2,983 test sessions 计算 envelope CPC
#
# Usage:
#   bash run_envelope_revision.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export TORCH_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "  Route B+ Envelope Payment — Full Run"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Start: $(date)"
echo ""

python -m src.eval.run_envelope_experiment \
    --model_dir outputs/final_qwen3emb/round_2 \
    --splits_dir outputs/final_qwen3emb \
    --dialogues data/dialogues_all.jsonl \
    --output_dir outputs/envelope_revision \
    --outcome_sessions outputs/final_qwen3emb/round_2/final_evaluation_sessions.csv \
    --n_grid 64 \
    --max_sessions 0 \
    --policies learned_stop first_turn last_turn random_turn \
    --device auto \
    --seed 42

echo ""
echo "=============================================="
echo "  Generating Figures"
echo "=============================================="

python -m src.eval.plot_envelope \
    --output_dir outputs/envelope_revision

python -m src.eval.export_envelope_paper_results \
    --outcome_sessions outputs/final_qwen3emb/round_2/final_evaluation_sessions.csv \
    --envelope_results outputs/envelope_revision/envelope_results.csv \
    --output_dir outputs/envelope_revision

echo ""
echo "=============================================="
echo "  Done: $(date)"
echo "  Results: outputs/envelope_revision/"
echo "=============================================="
