#!/usr/bin/env bash
# Fixed-horizon revision: S1 subset; S2 tests; S3 proxy A/B;
# S4 reuse frozen models/retrain StopNet; S5 envelope diagnostics; S6 welfare report.
# E4 strategic revenue and E5 literature comparisons are not implemented here.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
[[ ! -f run_revision.local.sh ]] || source run_revision.local.sh
PYTHON=${PYTHON:-python3}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data}
OLD_RUN=${OLD_RUN:-$PROJECT_ROOT/outputs/final_tfidf}
# data_splits.json and user_id_map.json are always written at the run root
# (run_final_experiment.py:285,322), while checkpoints move into round_N as soon
# as total_rounds > 1 (:424).  Resolve the highest round that actually carries
# user_model.pt rather than assuming the literal name round_2; a wrong guess
# makes S4 copy nothing and silently retrain everything (:457,:488,:516).
if [[ -z "${OLD_MODEL_DIR:-}" ]]; then
  OLD_MODEL_DIR="$OLD_RUN"
  _best=-1
  shopt -s nullglob
  for _d in "$OLD_RUN"/round_*; do
    _n=${_d##*/round_}
    if [[ "$_n" =~ ^[0-9]+$ && -f "$_d/user_model.pt" ]] && (( _n > _best )); then
      _best=$_n
      OLD_MODEL_DIR="$_d"
    fi
  done
  shopt -u nullglob
fi
echo "== horizon stratum: T=$LLM_OSDA_HORIZON   OUT=$OUT =="
echo "== reuse source: $OLD_RUN =="
shopt -s nullglob
for _d in "$OLD_RUN" "$OLD_RUN"/round_*; do
  _mark=" "
  [[ "$_d" != "$OLD_MODEL_DIR" ]] || _mark="*"
  _have=""
  for _f in user_model.pt stop_policy.pt text_encoder_meta.json text_encoder_tfidf.joblib llm_full_sft/config.json data_splits.json user_id_map.json final_evaluation_summary.csv; do
    [[ ! -e "$_d/$_f" ]] || _have="$_have $_f"
  done
  [[ -n "$_have" ]] || _have=" (none)"
  printf '  %s %s ->%s\n' "$_mark" "$_d" "$_have"
done
shopt -u nullglob
echo "  selected OLD_MODEL_DIR=$OLD_MODEL_DIR"
# Horizon stratum under evaluation.  T=3 keeps the historical default so the
# already-completed outputs/revision run is never overwritten; any other value
# gets its own OUT directory.  A stratum is a set of sessions whose dialogue
# really has LLM_OSDA_HORIZON turns, so is_last == (t == HORIZON) stays a true
# horizon boundary and V_{T+1} = 0 remains legitimate inside that stratum.
export LLM_OSDA_HORIZON=${LLM_OSDA_HORIZON:-3}
case "$LLM_OSDA_HORIZON" in 1|2|3) ;; *) echo "LLM_OSDA_HORIZON must be 1, 2 or 3" >&2; exit 2 ;; esac
if [[ "$LLM_OSDA_HORIZON" == 3 ]]; then
  OUT=${OUT:-outputs/revision}
else
  OUT=${OUT:-outputs/revision_T$LLM_OSDA_HORIZON}
fi
[[ "$OUT" = /* ]] || OUT="$PROJECT_ROOT/$OUT"
ENCODER=${ENCODER:-tfidf}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
PROFILES=${PROFILES:-$DATA_DIR/profiles_.jsonl}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LLM_OSDA_DEDUP_CACHE=${LLM_OSDA_DEDUP_CACHE:-$OUT/dedup_cache_v2.json}
SUBSET="$OUT/dialogues_t$LLM_OSDA_HORIZON.jsonl"
MAIN="$OUT/main"
# second_turn is a distinct arm only when turn 2 is INTERIOR.  At T=2 it is the
# same decision as last_turn; at T=1 turn 2 does not exist.  The arm list must
# match _resolve_auction_for_policy, which applies the identical HORIZON gate.
POLICIES=(hindsight_bellman learned_stop first_turn last_turn random_turn)
if (( LLM_OSDA_HORIZON > 2 )); then POLICIES+=(second_turn); fi
DATA_ARGS=(--dialogues "$SUBSET" --counterfactuals "$DATA_DIR/counterfactuals_all.jsonl" --sft "$DATA_DIR/sft_all.jsonl" --profiles "$PROFILES")
mkdir -p "$OUT"
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(S1 S2 S3 S4 S5 S6)
want() { for stage in "${STAGES[@]}"; do [[ "$stage" != "$1" ]] || return 0; done; return 1; }
for stage in "${STAGES[@]}"; do
  case "$stage" in S1|S2|S3|S4|S4FULL|S5|S6) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
done
if want S4 && want S4FULL; then echo 'Choose S4 or S4FULL, not both' >&2; exit 2; fi

# profiles_.jsonl is required by every stage that trains or evaluates through the
# judge.  run_final_experiment.py:271-285 uses it twice: it injects
# d["user_profile"] into the judge text and it builds user_id_map, which indexes
# the judge's learned nn.Embedding.  If the file is absent the run does not fail;
# it logs "skipping" and sends every session to uid=0 (the padding row), so CTR
# and gamma_score are produced with a zero user vector against a judge that was
# trained on real uids.  Fail loudly instead.
if want S3 || want S4 || want S4FULL; then
  if [[ ! -f "$PROFILES" ]]; then
    echo "Missing profiles file: $PROFILES" >&2
    echo "Set PROFILES=/abs/path/to/profiles_.jsonl (3000 lines, ids h_0000..h_2999)." >&2
    echo "Refusing to run with the silent uid=0 fallback." >&2
    exit 2
  fi
fi

# S5/S6 depend on the directory S4 recorded; without this guard "cat" dies under
# set -e with no explanation.
require_model_dir() {
  if [[ ! -f "$OUT/.model_dir" ]]; then
    echo "Missing $OUT/.model_dir -- run S4 (or S4FULL) first, or write the" >&2
    echo "checkpoint directory into that file by hand." >&2
    return 1
  fi
  MODEL_DIR=$(cat "$OUT/.model_dir")
  if [[ ! -f "$MODEL_DIR/stop_policy.pt" ]]; then
    echo "Recorded model dir has no stop_policy.pt: $MODEL_DIR" >&2
    return 1
  fi
}

if want S1; then
  mkdir -p "$MAIN"
  "$PYTHON" tools/make_t3_subset.py --dialogues "$DATA_DIR/dialogues_all.jsonl" \
    --out "$SUBSET" --splits "$OLD_RUN/data_splits.json" \
    --out-splits "$MAIN/data_splits.json" --horizon "$LLM_OSDA_HORIZON" --drop-null-auction-turn \
    --report "$OUT/subset_report.json"
fi

if want S2; then
  "$PYTHON" -m pytest tests -q
fi

if want S3; then
  AB="$OUT/ab"
  mkdir -p "$AB"
  cp "$MAIN/data_splits.json" "$AB/data_splits.json"
  if [[ ! -f "$AB/stop_policy.pt" ]]; then
    LLM_OSDA_DEDUP=0 "$PYTHON" -u -m src.run_final_experiment \
      --mode train --train_rounds 1 "${DATA_ARGS[@]}" --output_dir "$AB" \
      --text_encoder "$ENCODER" --base_model "$BASE_MODEL" --eval_with_proxy_renderer \
      --pctr_epochs 3 --stop_epochs 10 --patience 1 2>&1 | tee "$OUT/ab_train.log"
  fi
  # Compare frozen per-session outcomes, not only averages. No persistent cache
  # is loaded in either arm; persistence itself has separate regression tests.
  for MODE in off on; do
    DEDUP=0
    [[ "$MODE" != on ]] || DEDUP=1
    LLM_OSDA_DEDUP="$DEDUP" LLM_OSDA_DEDUP_CACHE='' \
      "$PYTHON" -u -m src.run_final_experiment --mode eval --eval_rounds 1 \
      --model_dir "$AB" "${DATA_ARGS[@]}" --output_dir "$AB" \
      --text_encoder "$ENCODER" --base_model "$BASE_MODEL" \
      --eval_with_proxy_renderer --include_ablation --max_eval_sessions 50 \
      --eval_batch_size 8 2>&1 | tee "$OUT/ab_$MODE.log"
    cp "$AB/final_evaluation_sessions.csv" "$AB/sessions_$MODE.csv"
  done
  "$PYTHON" - "$AB" <<'PY'
import pathlib, sys, pandas as pd
p = pathlib.Path(sys.argv[1])
a, b = [pd.read_csv(p/f"sessions_{mode}.csv").sort_values(["policy", "session_id"]).reset_index(drop=True) for mode in ("off", "on")]
pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-6, rtol=0)
print("S3 PASS: every paired session outcome agrees")
PY
fi

if want S4; then
  # Verify model provenance and split isolation again, even when S1 was skipped.
  "$PYTHON" - "$OLD_RUN" "$OLD_MODEL_DIR" "$MAIN" "$ENCODER" <<'PY'
import json, shutil, sys
from pathlib import Path
old_run, old, new = map(Path, sys.argv[1:4])
encoder = sys.argv[4]
a = json.loads((old_run/'data_splits.json').read_text())
b = json.loads((new/'data_splits.json').read_text())
for name in ('train_sessions', 'val_sessions', 'test_sessions'):
    if not set(b[name]).issubset(a[name]):
        raise ValueError(f"Subset changed original {name} membership")
if set(b['test_sessions']) & (set(a['train_sessions']) | set(a['val_sessions'])):
    raise ValueError('New test intersects old training/validation')
meta = json.loads((old/'text_encoder_meta.json').read_text())
if meta['type'] != encoder:
    raise ValueError(f"ENCODER={encoder} differs from checkpoint {meta['type']}")
required = ['user_model.pt', 'text_encoder_meta.json']
# TfidfTextEncoder.load joblib-loads text_encoder_tfidf.joblib (text_mlp.py:73),
# reached from UnifiedUserModel.load -> load_text_encoder (text_mlp.py:1071,222).
# Without it S4 dies at "[train 1/5] ... loading" (run_final_experiment.py:457),
# so fail here instead of after the split checks.
if meta['type'] == 'tfidf':
    required.append('text_encoder_tfidf.joblib')
for f in required:
    if not (old/f).is_file(): raise FileNotFoundError(old/f)
if (new/'stop_policy.pt').exists():
    raise FileExistsError('S4 requires a fresh MAIN directory for StopNet retraining; use eval mode to resume an existing run')
for f in ('user_model.pt', 'text_encoder_meta.json', 'text_encoder_tfidf.joblib'):
    if (old/f).exists(): shutil.copy2(old/f, new/f)
# The default vLLM path requires a standalone full SFT checkpoint.
sft = old/'llm_full_sft'
if not (sft/'config.json').is_file():
    raise FileNotFoundError(f"S4 --use_vllm requires a full SFT model at {sft}; a LoRA-only directory must first be merged")
dest = new/'llm_full_sft'
if dest.exists(): raise FileExistsError(f"Use a fresh output directory: {dest}")
shutil.copytree(sft, dest)
(new/'llm_sft_meta.json').write_text(json.dumps({'save_dir': str(dest.resolve())}, indent=2))
(new/'reuse_provenance.json').write_text(json.dumps({'old_run': str(old_run.resolve()), 'old_model_dir': str(old.resolve()), 'encoder': encoder}, indent=2))
print('S4 model reuse and inherited split verified')
PY
  LLM_OSDA_DEDUP=1 "$PYTHON" -u -m src.run_final_experiment \
    --mode all --train_rounds 1 --eval_rounds 1 "${DATA_ARGS[@]}" --output_dir "$MAIN" \
    --base_model "$BASE_MODEL" --text_encoder "$ENCODER" --stop_epochs 30 \
    --patience 1 --lr 1e-3 --batch_size 256 --discount 0.95 --lambda_gamma 0.25 \
    --llm_max_new_tokens 128 --renderer_temperature 0 --max_eval_sessions 0 \
    --eval_batch_size 64 --use_vllm --include_ablation 2>&1 | tee "$OUT/main.log"
  echo "$MAIN" > "$OUT/.model_dir"
fi

if want S4FULL; then
  if [[ -e "$MAIN/round_1" || -e "$MAIN/user_model.pt" ]]; then
    echo 'S4FULL requires a fresh OUT directory (run S1 there first)' >&2; exit 2
  fi
  LLM_OSDA_DEDUP=1 "$PYTHON" -u -m src.run_final_experiment \
    --mode all --train_rounds 2 --eval_rounds 2 "${DATA_ARGS[@]}" --output_dir "$MAIN" \
    --base_model "$BASE_MODEL" --text_encoder "$ENCODER" --pctr_epochs 5 \
    --judge_epochs 5 --stop_epochs 30 --patience 1 --lr 1e-3 --batch_size 256 \
    --discount 0.95 --lambda_gamma 0.25 --llm_max_samples 0 --llm_epochs 1 \
    --llm_max_length 8192 --llm_batch_size 4 --llm_grad_accum 32 --llm_lr 2e-5 \
    --llm_max_new_tokens 128 --renderer_temperature 0 --max_eval_sessions 0 \
    --eval_batch_size 64 --use_vllm --include_ablation 2>&1 | tee "$OUT/main_full.log"
  echo "$MAIN/round_2" > "$OUT/.model_dir"
fi

if want S4 || want S4FULL; then
  require_model_dir
  "$PYTHON" - "$MODEL_DIR" "$OLD_RUN" "$MAIN" <<'PY'
import json, os, sys, pathlib, torch, pandas as pd
model_dir, old_run, main = map(pathlib.Path, sys.argv[1:4])
horizon = int(os.environ.get('LLM_OSDA_HORIZON', '3'))
assert torch.load(model_dir/'stop_policy.pt', map_location='cpu')['input_dim'] == 12
s = pd.read_csv(model_dir/'final_evaluation_sessions.csv')
expected = ['first_turn_trained_llm', 'last_turn_trained_llm', 'learned_stop', 'hindsight_bellman']
if horizon > 2:
    expected.insert(1, 'second_turn_trained_llm')
for policy in expected:
    rows = s[s.policy == policy]
    assert len(rows) and (rows.renderer_group == 'trained').all(), policy
print(f'S4 PASS: 12-dimensional StopNet and matched renderer controls (T={horizon}, arms={expected})')
# user_id_map is built by enumerating sorted(profile_id), so one missing profile
# shifts every row of the reused judge's user embedding.  Verify it was built at
# all, and that it matches the run whose judge weights are being reused.
new_path = main/'user_id_map.json'
new_map = json.loads(new_path.read_text()) if new_path.is_file() else {}
assert new_map, f'{new_path} missing or empty: profiles were not read, every session fell back to uid=0'
old_path = old_run/'user_id_map.json'
if not old_path.is_file():
    print(f'WARNING: {old_path} absent; cannot verify uid alignment with the reused judge')
else:
    old_map = json.loads(old_path.read_text())
    assert old_map == new_map, (
        f'user_id_map differs from the reused run ({len(old_map)} vs {len(new_map)} entries): '
        'the judge user embedding would be misaligned')
    print(f'S4 PASS: user_id_map identical to {old_run} ({len(new_map)} users)')
PY
fi

if want S5; then
  require_model_dir
  # Allocation-only replay: this stage does not render or judge any responses.
  "$PYTHON" -u -m src.eval.run_envelope_experiment --model_dir "$MODEL_DIR" \
    --splits_dir "$MAIN" --dialogues "$SUBSET" --output_dir "$OUT/envelope" \
    --outcome_sessions "$MODEL_DIR/final_evaluation_sessions.csv" \
    --n_grid 64 --max_sessions 0 --gamma 0.95 --seed 42 --device auto \
    --policies "${POLICIES[@]}" 2>&1 | tee "$OUT/envelope.log"
  "$PYTHON" -m src.eval.export_envelope_paper_results \
    --outcome_sessions "$MODEL_DIR/final_evaluation_sessions.csv" \
    --envelope_results "$OUT/envelope/envelope_results.csv" --output_dir "$OUT/envelope" \
    --horizon "$LLM_OSDA_HORIZON"
fi

if want S6; then
  require_model_dir
  "$PYTHON" -m src.eval.revision_metrics --outcome_sessions "$MODEL_DIR/final_evaluation_sessions.csv" \
    --output "$OUT/paper_headline_numbers.json" --horizon "$LLM_OSDA_HORIZON"
fi
printf '\nCompleted requested stages. E4 strategic revenue and E5 literature comparisons remain pending.\n'
