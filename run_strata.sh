#!/usr/bin/env bash
# Stratified revision pipeline.  (NEW FILE: run_strata.sh -- run_revision.sh is untouched)
#
# WHY TRAINING AND EVALUATION ARE SPLIT
# -------------------------------------
#   TRAIN  must MIX T=1/2/3.  Restricting StopNet to the T=3 slice keeps only
#          2,324 of 9,527 clean training sessions (75.6% thrown away) and is not
#          comparable with the original run's 10,444.  Mixing is sound: the
#          realized T shapes the Bellman target, it is never an input feature
#          (stop_features_v2 is 12-dim, carries no n_turns).
#   EVAL   must stay inside ONE stratum.  The forced last-turn auction asks
#          "is this the last turn?".  On a mixed corpus that is either
#          `t == HORIZON` (never fires for short sessions -> the guarantee
#          disappears) or `t == len(turns)` (that IS the label).
#   A single `--mode all` cannot satisfy both.  Running `--mode all` three times
#          would retrain StopNet/pCTR/judge per stratum, so the three strata
#          would evaluate three DIFFERENT models and the weighted recombination
#          would be meaningless.  Hence: train once, evaluate three times, pool.
#
# THE ORIGINAL SPLIT IS NOT RE-DRAWN.  tools/make_strata_subset.py filters
# $OLD_RUN/data_splits.json; train/val keep the full clean mixed membership in
# every stratum file and only test_sessions is partitioned by realized T.
#
# Stages (mapping to the old run_revision.sh):
#   D      <- S1   build the clean corpus + per-stratum splits
#   TEST   <- S2   pytest (unchanged stage, kept so the guards still run)
#   AB     <- S3   dedup off/on paired A/B on frozen per-session outcomes
#   TRAIN  <- S4a  mixed-horizon StopNet, frozen pCTR/judge/renderer
#   EVAL   <- S4b  closed-loop evaluation, one process per stratum
#   ENV    <- S5   envelope diagnostics + per-stratum headline numbers
#   POOL   <- S6   recombine the strata into the 2,741-session report
# Default: D TEST TRAIN AB EVAL ENV POOL
#   (AB runs after TRAIN so it can reuse that StopNet instead of fitting a
#    throwaway one on the 9,527-session mixed pool.)
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
[[ ! -f run_revision.local.sh ]] || source run_revision.local.sh
# Dedicated overrides survive assignments in run_revision.local.sh.
OUT=${STRATA_RUN_OUT:-${OUT:-outputs/strata}}
OLD_MODEL_DIR=${STRATA_FROZEN_MODEL_DIR:-${OLD_MODEL_DIR:-}}
TRAIN_SEED=${STRATA_TRAIN_SEED:-${TRAIN_SEED:-42}}
EVAL_SEED=${STRATA_EVAL_SEED:-${EVAL_SEED:-42}}
PYTHON=${PYTHON:-python3}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data}
OLD_RUN=${OLD_RUN:-$PROJECT_ROOT/outputs/final_tfidf}
OUT=${OUT:-outputs/strata}
[[ "$OUT" = /* ]] || OUT="$PROJECT_ROOT/$OUT"
ENCODER=${STRATA_ENCODER:-${ENCODER:-tfidf}}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
PROFILES=${PROFILES:-$DATA_DIR/profiles_.jsonl}
STRATA=${STRATA_RUN_HORIZONS:-${STRATA:-1 2 3}}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Checkpoints move into round_N as soon as total_rounds > 1; resolve the highest
# round that actually carries user_model.pt instead of guessing round_2.
if [[ -z "${OLD_MODEL_DIR:-}" ]]; then
  OLD_MODEL_DIR="$OLD_RUN"; _best=-1
  shopt -s nullglob
  for _d in "$OLD_RUN"/round_*; do
    _n=${_d##*/round_}
    if [[ "$_n" =~ ^[0-9]+$ && -f "$_d/user_model.pt" ]] && (( _n > _best )); then _best=$_n; OLD_MODEL_DIR="$_d"; fi
  done
  shopt -u nullglob
fi

CORPUS="$OUT/dialogues_clean.jsonl"
TRAIN_DIR="$OUT/main"
DATA_ARGS=(--dialogues "$CORPUS" --counterfactuals "$DATA_DIR/counterfactuals_all.jsonl" --sft "$DATA_DIR/sft_all.jsonl" --profiles "$PROFILES")
mkdir -p "$OUT"

echo "== stratified run =="
echo "   OUT=$OUT"
echo "   reuse source=$OLD_RUN  (selected $OLD_MODEL_DIR)"
echo "   strata=$STRATA  encoder=$ENCODER  train_seed=$TRAIN_SEED  eval_seed=$EVAL_SEED"

STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(D TEST TRAIN AB EVAL ENV POOL)
for stage in "${STAGES[@]}"; do
  case "$stage" in D|TEST|AB|TRAIN|EVAL|ENV|POOL) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
done
want() { for s in "${STAGES[@]}"; do [[ "$s" != "$1" ]] || return 0; done; return 1; }

# ---------------------------------------------------------------- D: corpus --
if want D; then
  "$PYTHON" tools/make_strata_subset.py \
    --dialogues "$DATA_DIR/dialogues_all.jsonl" --out "$CORPUS" \
    --splits "$OLD_RUN/data_splits.json" \
    --out-train-splits "$TRAIN_DIR/data_splits.json" \
    --out-eval-splits-dir "$OUT" --max-horizon 3 --drop-null-auction-turn \
    --report "$OUT/strata_report.json" 2>&1 | tee "$OUT/strata_data.log"
fi

# ------------------------------------------------------------- TEST: pytest --
if want TEST; then
  # Same stage as run_revision.sh S2.  conftest.py pins LLM_OSDA_HORIZON=3 for
  # the suite, so this is independent of whichever stratum runs next.
  "$PYTHON" -m pytest tests -q 2>&1 | tee "$OUT/tests.log"
fi

# ------------------------------------------------- TRAIN: mixed-horizon StopNet
if want TRAIN; then
  [[ -f "$CORPUS" ]] || { echo "missing $CORPUS; run stage D first" >&2; exit 2; }
  # Freeze the reused pCTR/judge/renderer; only StopNet is retrained.
  "$PYTHON" - "$OLD_RUN" "$OLD_MODEL_DIR" "$TRAIN_DIR" "$ENCODER" <<'PY'
import json, shutil, sys
from pathlib import Path
old_run, old, new = map(Path, sys.argv[1:4]); encoder = sys.argv[4]
a = json.loads((old_run/'data_splits.json').read_text())
b = json.loads((new/'data_splits.json').read_text())
for name in ('train_sessions', 'val_sessions', 'test_sessions'):
    if not set(b[name]).issubset(a[name]):
        raise ValueError(f"filtered splits changed original {name} membership")
if set(b['test_sessions']) & (set(a['train_sessions']) | set(a['val_sessions'])):
    raise ValueError('new test intersects old training/validation')
meta = json.loads((old/'text_encoder_meta.json').read_text())
if meta['type'] != encoder:
    raise ValueError(f"ENCODER={encoder} differs from checkpoint {meta['type']}")
required = ['user_model.pt', 'text_encoder_meta.json'] + (['text_encoder_tfidf.joblib'] if meta['type'] == 'tfidf' else [])
for f in required:
    if not (old/f).is_file(): raise FileNotFoundError(old/f)
if (new/'stop_policy.pt').exists():
    raise FileExistsError('TRAIN needs a fresh $OUT/main; StopNet is retrained from scratch')
for f in ('user_model.pt', 'text_encoder_meta.json', 'text_encoder_tfidf.joblib'):
    if (old/f).exists(): shutil.copy2(old/f, new/f)
sft = old/'llm_full_sft'
if not (sft/'config.json').is_file():
    raise FileNotFoundError(f"--use_vllm requires a full SFT model at {sft}; merge a LoRA-only dir first")
dest = new/'llm_full_sft'
if not dest.exists():
    shutil.copytree(sft, dest)
(new/'llm_sft_meta.json').write_text(json.dumps({'save_dir': str(dest.resolve())}, indent=2))
(new/'reuse_provenance.json').write_text(json.dumps(
    {'old_run': str(old_run.resolve()), 'old_model_dir': str(old.resolve()), 'encoder': encoder}, indent=2))
print(f"TRAIN: reusing frozen pCTR/judge/renderer from {old}; retraining StopNet only")
PY
  LLM_OSDA_DEDUP=1 LLM_OSDA_HORIZON=3 LLM_OSDA_DEDUP_CACHE="$OUT/dedup_cache_v2.json" \
    "$PYTHON" -u -m src.run_final_experiment \
    --mode train --mixed_horizon_train --train_rounds 1 --seed "$TRAIN_SEED" "${DATA_ARGS[@]}" \
    --output_dir "$TRAIN_DIR" --base_model "$BASE_MODEL" --text_encoder "$ENCODER" \
    --stop_epochs 30 --patience 1 --lr 1e-3 --batch_size 256 --discount 0.95 \
    --lambda_gamma 0.25 2>&1 | tee "$OUT/train.log"
  "$PYTHON" - "$TRAIN_DIR" <<'PY'
import sys, torch, json
from pathlib import Path
d = Path(sys.argv[1])
ck = torch.load(d/'stop_policy.pt', map_location='cpu')
assert ck['input_dim'] == 12, f"StopNet input_dim={ck['input_dim']}, expected 12 (no n_turns feature)"
sp = json.loads((d/'data_splits.json').read_text())
print(f"TRAIN PASS: 12-dim StopNet trained on {len(sp['train_sessions'])} mixed-horizon sessions")
PY
fi

# -------------------------------------------------- AB: dedup off/on A/B (S3) --
if want AB; then
  # Equivalent of run_revision.sh S3, run inside the T=3 stratum: it has the
  # most LLM calls per session, so it is the strictest test that request
  # de-duplication does not change any per-session outcome.
  AB="$OUT/ab"
  AB_SPLITS="$OUT/eval_T3/data_splits.json"
  [[ -f "$AB_SPLITS" ]] || { echo "missing $AB_SPLITS; run stage D first" >&2; exit 2; }
  mkdir -p "$AB"
  cp "$AB_SPLITS" "$AB/data_splits.json"
  # Prefer the model TRAIN already produced.  The A/B only asks whether request
  # de-duplication changes any per-session outcome, so it does not need its own
  # checkpoint -- and fitting one here would re-run the pCTR forward pass over
  # all 9,527 mixed training sessions (the old S3 pool was 2,324).
  if [[ -f "$TRAIN_DIR/stop_policy.pt" ]]; then
    AB_MODEL="$TRAIN_DIR"
    echo "AB: reusing the StopNet from $TRAIN_DIR"
  else
    AB_MODEL="$AB"
    echo "AB: $TRAIN_DIR has no StopNet yet; fitting a throwaway proxy model"
    if [[ ! -f "$AB/stop_policy.pt" ]]; then
      LLM_OSDA_DEDUP=0 LLM_OSDA_HORIZON=3 "$PYTHON" -u -m src.run_final_experiment \
        --mode train --mixed_horizon_train --train_rounds 1 "${DATA_ARGS[@]}" \
        --output_dir "$AB" --text_encoder "$ENCODER" --base_model "$BASE_MODEL" \
        --eval_with_proxy_renderer --pctr_epochs 3 --stop_epochs 10 --patience 1 \
        2>&1 | tee "$OUT/ab_train.log"
    fi
  fi
  # Compare frozen per-session outcomes, not only averages.  Neither arm loads a
  # persistent cache; persistence has its own regression tests.
  for MODE in off on; do
    DEDUP=0; [[ "$MODE" != on ]] || DEDUP=1
    LLM_OSDA_DEDUP="$DEDUP" LLM_OSDA_DEDUP_CACHE='' LLM_OSDA_HORIZON=3 \
      "$PYTHON" -u -m src.run_final_experiment --mode eval --eval_rounds 1 \
      --model_dir "$AB_MODEL" "${DATA_ARGS[@]}" --output_dir "$AB" \
      --text_encoder "$ENCODER" --base_model "$BASE_MODEL" \
      --eval_with_proxy_renderer --include_ablation --max_eval_sessions 50 \
      --eval_batch_size 8 2>&1 | tee "$OUT/ab_$MODE.log"
    cp "$AB/final_evaluation_sessions.csv" "$AB/sessions_$MODE.csv"
  done
  "$PYTHON" - "$AB" <<'PY'
import pathlib, sys, pandas as pd
p = pathlib.Path(sys.argv[1])
a, b = [pd.read_csv(p/f"sessions_{m}.csv").sort_values(["policy", "session_id"]).reset_index(drop=True)
        for m in ("off", "on")]
pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-6, rtol=0)
print("AB PASS: every paired session outcome agrees with de-duplication on and off")
PY
fi

# -------------------------------------------- EVAL: one run per horizon stratum
if want EVAL; then
  [[ -f "$TRAIN_DIR/stop_policy.pt" ]] || { echo "missing $TRAIN_DIR/stop_policy.pt; run TRAIN first" >&2; exit 2; }
  for h in $STRATA; do
    ED="$OUT/eval_T$h"
    [[ -f "$ED/data_splits.json" ]] || { echo "missing $ED/data_splits.json; run stage D first" >&2; exit 2; }
    echo "--- EVAL stratum T=$h ---"
    # HORIZON is read at import time, so each stratum must be its own process.
    LLM_OSDA_DEDUP=1 LLM_OSDA_HORIZON="$h" LLM_OSDA_DEDUP_CACHE="$OUT/dedup_cache_v2.json" \
      "$PYTHON" -u -m src.run_final_experiment \
      --mode eval --eval_rounds 1 --seed "$EVAL_SEED" --model_dir "$TRAIN_DIR" "${DATA_ARGS[@]}" \
      --output_dir "$ED" --base_model "$BASE_MODEL" --text_encoder "$ENCODER" \
      --discount 0.95 --lambda_gamma 0.25 --llm_max_new_tokens 128 \
      --renderer_temperature 0 --max_eval_sessions 0 --eval_batch_size 64 \
      --use_vllm --include_ablation 2>&1 | tee "$OUT/eval_T$h.log"
    "$PYTHON" - "$ED" "$h" "$TRAIN_DIR" <<'PY'
import json, sys, pandas as pd
from pathlib import Path
ed, h, train_dir = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
s = pd.read_csv(ed/'final_evaluation_sessions.csv')
expected = ['first_turn_trained_llm', 'last_turn_trained_llm', 'learned_stop', 'hindsight_bellman']
if h > 2: expected.insert(1, 'second_turn_trained_llm')
for p in expected:
    rows = s[s.policy == p]
    assert len(rows) and (rows.renderer_group == 'trained').all(), f"T={h}: missing/mismatched arm {p}"
if h <= 2:
    assert 'second_turn_trained_llm' not in set(s.policy), f"T={h}: second_turn must not be emitted"
want = set(json.loads((ed/'data_splits.json').read_text())['test_sessions'])
got = set(s.session_id.unique())
assert got == want, f"T={h}: evaluated {len(got)} sessions, split lists {len(want)}"
# One StopNet for all strata: the pooled mean is only interpretable if the three
# runs scored the SAME model.
a = json.loads((ed/'user_id_map.json').read_text()) if (ed/'user_id_map.json').is_file() else {}
b = json.loads((train_dir/'user_id_map.json').read_text()) if (train_dir/'user_id_map.json').is_file() else {}
assert a and a == b, 'user_id_map differs from the training run: judge embeddings would be misaligned'
print(f"EVAL PASS T={h}: {len(got)} sessions, arms={expected}")
PY
  done
fi

# ------------------------------------------- ENV: envelope diagnostics per layer
if want ENV; then
  for h in $STRATA; do
    ED="$OUT/eval_T$h"
    [[ -f "$ED/final_evaluation_sessions.csv" ]] || { echo "missing eval output for T=$h" >&2; exit 2; }
    POLICIES=(hindsight_bellman learned_stop first_turn last_turn random_turn)
    (( h > 2 )) && POLICIES+=(second_turn)
    echo "--- ENV stratum T=$h ---"
    LLM_OSDA_HORIZON="$h" "$PYTHON" -u -m src.eval.run_envelope_experiment \
      --model_dir "$TRAIN_DIR" --splits_dir "$ED" --dialogues "$CORPUS" \
      --output_dir "$ED/envelope" --outcome_sessions "$ED/final_evaluation_sessions.csv" \
      --n_grid 64 --max_sessions 0 --gamma 0.95 --seed 42 --device auto \
      --policies "${POLICIES[@]}" 2>&1 | tee "$OUT/envelope_T$h.log"
    LLM_OSDA_HORIZON="$h" "$PYTHON" -m src.eval.export_envelope_paper_results \
      --outcome_sessions "$ED/final_evaluation_sessions.csv" \
      --envelope_results "$ED/envelope/envelope_results.csv" \
      --output_dir "$ED/envelope" --horizon "$h"
    LLM_OSDA_HORIZON="$h" "$PYTHON" -m src.eval.revision_metrics \
      --outcome_sessions "$ED/final_evaluation_sessions.csv" \
      --output "$ED/paper_headline_numbers.json" --horizon "$h"
  done
fi

# ------------------------------------------------------- POOL: recombine strata
if want POOL; then
  "$PYTHON" tools/pool_strata.py --run_root "$OUT" --out_dir "$OUT/pooled" \
    --strata "$(echo $STRATA | tr ' ' ',')" --n_bootstrap 2000 --seed 42 \
    2>&1 | tee "$OUT/pooled.log"
fi

printf '\nStages done: %s\n' "${STAGES[*]}"
printf 'Pooled table: %s\n' "$OUT/pooled/pooled_policy_table.csv"
printf 'E4 strategic revenue and E5 literature comparisons remain pending.\n'
