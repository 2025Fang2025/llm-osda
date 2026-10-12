#!/usr/bin/env bash
# ============================================================================
# run_matched_myopic.sh   (NEW FILE -- run_strata.sh / run_revision.sh untouched)
#
# EXPERIMENT A (NAACL rebuttal): same-condition refined-Myopic control.
#
# WHY
#   In closed_loop_eval.py the greedy "myopic_threshold" arm was rendered with
#   the UNTRAINED renderer, while "learned_stop" uses the TRAINED (refined)
#   renderer.  The learned-vs-myopic revenue gap therefore mixed "better timing"
#   with "better renderer" and could not isolate the stopping policy.
#   This run adds the arm "myopic_threshold_trained_llm": the SAME greedy timing
#   rendered with the SAME refined renderer, judge, test sessions, candidate ads,
#   bids and discount as learned_stop.  learned_stop - myopic_threshold_trained_llm
#   is then a clean, renderer-matched estimate of the stopping policy's value.
#
# WHAT IT REUSES (NO RETRAINING)
#   - the StopNet / pCTR / judge / renderer already trained by run_strata.sh,
#     taken from  $SRC/main                (default SRC=outputs/strata)
#   - the frozen per-stratum test splits    $SRC/eval_T{1,2,3}/data_splits.json
#   - the cleaned corpus                     $SRC/dialogues_clean.jsonl
#   It only re-renders + re-scores the closed loop, which also reproduces every
#   existing arm bit-for-bit (determinism check) and adds the new matched arm.
#
# OUTPUT GOES TO A FRESH DIR so nothing is overwritten.
#   ROLLBACK = `rm -rf outputs/strata_matched_myopic`  + revert the two-line
#              edit in src/eval/closed_loop_eval.py (git checkout that file).
#
# THRESHOLD POLICY
#   The myopic rule stops at the first turn with winner b_i*q_i >= MYOPIC_THRESHOLD.
#   It is FIXED IN ADVANCE at 0.4 (the value already used by the reported
#   myopic_threshold arm), NOT tuned on the test set.  Keeping it identical means
#   the untrained myopic arm reproduces the existing numbers and the only change
#   is the matched renderer, so the contrast isolates exactly that.  To tune on
#   the validation split instead, set MYOPIC_THRESHOLD and record the rationale.
#
# USAGE (on the server, from the project root)
#   bash run_matched_myopic.sh                 # EVAL POOL DIFF (default)
#   bash run_matched_myopic.sh EVAL            # just re-render the closed loop
#   bash run_matched_myopic.sh POOL DIFF       # recombine + paired contrast only
#   SRC=outputs/strata MYOPIC_THRESHOLD=0.4 bash run_matched_myopic.sh
# ============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
[[ ! -f run_revision.local.sh ]] || source run_revision.local.sh

# ---- reuse source (a COMPLETED run_strata.sh run with trained models) --------
SRC=${SRC:-outputs/strata}
[[ "$SRC" = /* ]] || SRC="$PROJECT_ROOT/$SRC"
OUT=${MATCHED_OUT:-outputs/strata_matched_myopic}
[[ "$OUT" = /* ]] || OUT="$PROJECT_ROOT/$OUT"

TRAIN_DIR="$SRC/main"
CORPUS=${MATCHED_CORPUS:-$SRC/dialogues_clean.jsonl}
PYTHON=${PYTHON:-python3}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
ENCODER=${STRATA_ENCODER:-${ENCODER:-tfidf}}
PROFILES=${PROFILES:-$DATA_DIR/profiles_.jsonl}
STRATA=${STRATA_RUN_HORIZONS:-${STRATA:-1 2 3}}
EVAL_SEED=${STRATA_EVAL_SEED:-${EVAL_SEED:-42}}
export MYOPIC_THRESHOLD=${MYOPIC_THRESHOLD:-0.4}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_ARGS=(--dialogues "$CORPUS" --counterfactuals "$DATA_DIR/counterfactuals_all.jsonl" --sft "$DATA_DIR/sft_all.jsonl" --profiles "$PROFILES")
mkdir -p "$OUT"

echo "== matched-renderer myopic control =="
echo "   reuse model=$TRAIN_DIR"
echo "   corpus=$CORPUS"
echo "   OUT=$OUT   encoder=$ENCODER   eval_seed=$EVAL_SEED   MYOPIC_THRESHOLD=$MYOPIC_THRESHOLD"
[[ -f "$TRAIN_DIR/stop_policy.pt" ]] || { echo "ERROR: missing $TRAIN_DIR/stop_policy.pt -- run run_strata.sh TRAIN first." >&2; exit 2; }
[[ -f "$CORPUS" ]] || { echo "ERROR: missing corpus $CORPUS -- run run_strata.sh D first." >&2; exit 2; }

STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(EVAL POOL DIFF)
for stage in "${STAGES[@]}"; do
  case "$stage" in EVAL|POOL|DIFF) ;; *) echo "Unknown stage: $stage (use EVAL POOL DIFF)" >&2; exit 2 ;; esac
done
want() { for s in "${STAGES[@]}"; do [[ "$s" != "$1" ]] || return 0; done; return 1; }

# -------------------------------------------- EVAL: one process per stratum ---
if want EVAL; then
  for h in $STRATA; do
    ED="$OUT/eval_T$h"
    mkdir -p "$ED"
    SPLIT="$SRC/eval_T$h/data_splits.json"
    [[ -f "$SPLIT" ]] || { echo "missing $SPLIT -- run run_strata.sh D first" >&2; exit 2; }
    cp "$SPLIT" "$ED/data_splits.json"
    echo "--- EVAL stratum T=$h  (reuse $TRAIN_DIR) ---"
    LLM_OSDA_DEDUP=1 LLM_OSDA_HORIZON="$h" LLM_OSDA_DEDUP_CACHE="$OUT/dedup_cache_v2.json" \
      "$PYTHON" -u -m src.run_final_experiment \
      --mode eval --eval_rounds 1 --seed "$EVAL_SEED" --model_dir "$TRAIN_DIR" "${DATA_ARGS[@]}" \
      --output_dir "$ED" --base_model "$BASE_MODEL" --text_encoder "$ENCODER" \
      --discount 0.95 --lambda_gamma 0.25 --llm_max_new_tokens 128 \
      --renderer_temperature 0 --max_eval_sessions 0 --eval_batch_size 64 \
      --use_vllm --include_ablation 2>&1 | tee "$OUT/eval_T$h.log"
    "$PYTHON" - "$ED" "$h" <<'PY'
import sys, pandas as pd
from pathlib import Path
ed, h = Path(sys.argv[1]), int(sys.argv[2])
s = pd.read_csv(ed/'final_evaluation_sessions.csv')
for p in ('learned_stop', 'myopic_threshold', 'myopic_threshold_trained_llm'):
    rows = s[s.policy == p]
    assert len(rows), f"T={h}: arm {p} missing"
m = s[s.policy == 'myopic_threshold_trained_llm']
assert (m.renderer_group == 'trained').all(), f"T={h}: matched myopic must use the trained renderer"
u = s[s.policy == 'myopic_threshold']
assert (u.renderer_group == 'untrained').all(), f"T={h}: plain myopic must stay on the untrained renderer"
# Same greedy timing on both renderer variants (identical turn per session).
j = m[['session_id','trigger_turn']].merge(u[['session_id','trigger_turn']], on='session_id', suffixes=('_tr','_un'))
assert (j.trigger_turn_tr == j.trigger_turn_un).all(), f"T={h}: matched/plain myopic chose different turns"
print(f"EVAL PASS T={h}: {s.session_id.nunique()} sessions; matched myopic arm present and renderer-matched")
PY
  done
fi

# ------------------------------------------------------- POOL: recombine ------
if want POOL; then
  for h in $STRATA; do
    [[ -f "$OUT/eval_T$h/final_evaluation_sessions.csv" ]] || { echo "missing eval output for T=$h; run EVAL first" >&2; exit 2; }
  done
  "$PYTHON" tools/pool_strata.py --run_root "$OUT" --out_dir "$OUT/pooled" \
    --strata "$(echo $STRATA | tr ' ' ',')" --n_bootstrap 2000 --seed 42 \
    2>&1 | tee "$OUT/pooled.log"
fi

# --------------------------- DIFF: paired learned_stop vs matched myopic ------
if want DIFF; then
  [[ -f "$OUT/pooled/pooled_sessions.csv" ]] || { echo "missing $OUT/pooled/pooled_sessions.csv; run POOL first" >&2; exit 2; }
  "$PYTHON" - "$OUT/pooled/pooled_sessions.csv" "$OUT/pooled/matched_myopic_contrast.csv" <<'PY'
import sys, numpy as np, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
A, B = "learned_stop", "myopic_threshold_trained_llm"
metrics = ["revenue", "gross_revenue", "model_welfare", "reward",
           "gamma_score", "expected_click", "clicked_binary", "trigger_turn"]
metrics = [m for m in metrics if m in df.columns]
a = df[df.policy == A].set_index("session_id")
b = df[df.policy == B].set_index("session_id")
common = a.index.intersection(b.index)
a, b = a.loc[common], b.loc[common]
rng = np.random.default_rng(42)
n = len(common)
print(f"paired sessions (pooled T1/T2/T3): {n}")
rows = []
for m in metrics:
    da, db = a[m].to_numpy(float), b[m].to_numpy(float)
    diff = da - db
    boot = np.array([diff[rng.integers(n, size=n)].mean() for _ in range(2000)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    rel = (diff.mean() / db.mean() * 100) if abs(db.mean()) > 1e-12 else float("nan")
    rows.append({"metric": m, f"{A}": da.mean(), f"{B}": db.mean(),
                 "diff(learned-myopic)": diff.mean(), "rel_%": rel,
                 "ci95_lo": lo, "ci95_hi": hi})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out.to_csv(dst, index=False)
print(f"\nwrote {dst}")
PY
fi

printf '\nStages done: %s\n' "${STAGES[*]}"
printf 'Pooled table : %s\n' "$OUT/pooled/pooled_policy_table.csv"
printf 'Paired diff  : %s\n' "$OUT/pooled/matched_myopic_contrast.csv"
