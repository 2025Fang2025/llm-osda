#!/usr/bin/env bash
# StopNet training stability, conditional on frozen quality/judge/renderer models.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
# Default reference run and training seeds; run with: bash run_multiseed.sh
REFERENCE="${REFERENCE:-outputs/strata}"
MULTISEED_OUT=${MULTISEED_OUT:-outputs/stopnet_multiseed}
SEEDS="${SEEDS:-42 43 44}"
EVAL_SEED=${EVAL_SEED:-42}
STAGES=${STAGES:-TRAIN EVAL POOL}
export REFERENCE MULTISEED_OUT SEEDS EVAL_SEED
# Reuse exact cleaned corpus and all splits. Do not draw new splits.
"$PYTHON" - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
ref=Path(os.environ['REFERENCE']).resolve();out=Path(os.environ['MULTISEED_OUT']).resolve()
seeds=[int(x) for x in os.environ['SEEDS'].split()]
if len(seeds)<2 or len(seeds)!=len(set(seeds)):raise ValueError('Need at least two distinct seeds')
files=['dialogues_clean.jsonl','main/data_splits.json']+[f'eval_T{h}/data_splits.json' for h in (1,2,3)]
for name in files+['main/user_model.pt','main/text_encoder_meta.json','main/llm_full_sft/config.json']:
 if not (ref/name).is_file():raise FileNotFoundError(ref/name)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
# Record hashes for frozen model and all renderer files once, then use the same source for each run.
frozen=[p for p in (ref/'main').rglob('*') if p.is_file() and
 (p.relative_to(ref/'main').parts[0]=='llm_full_sft' or p.name in ['user_model.pt','text_encoder_meta.json','text_encoder_tfidf.joblib'])]
common=dict(reference=str(ref),eval_seed=int(os.environ['EVAL_SEED']),target_bid_seed=42,
 hashes={str(p.relative_to(ref)):sha(p) for p in [ref/n for n in files]+frozen})
for seed in seeds:
 dest=out/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
 manifest=dict(common,train_seed=seed);mp=dest/'multiseed_manifest.json'
 if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError(f'Configuration changed: {dest}')
 for name in files:
  target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and sha(target)!=common['hashes'][name]:raise ValueError(f'Input changed: {target}')
  if not target.exists():shutil.copy2(ref/name,target)
 mp.write_text(json.dumps(manifest,indent=2))
print('Prepared seeds:',seeds,'from',ref)
PY
FROZEN=$("$PYTHON" -c 'import os;from pathlib import Path;print((Path(os.environ["REFERENCE"])/"main").resolve())')
ENCODER_TYPE=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["type"])' "$FROZEN/text_encoder_meta.json")
for seed in $SEEDS; do
  STRATA_RUN_OUT="$MULTISEED_OUT/seed_$seed" \
  STRATA_FROZEN_MODEL_DIR="$FROZEN" \
  STRATA_TRAIN_SEED="$seed" STRATA_EVAL_SEED="$EVAL_SEED" \
  STRATA_ENCODER="$ENCODER_TYPE" STRATA_RUN_HORIZONS="1 2 3" \
  bash run_strata.sh $STAGES
done
"$PYTHON" tools/summarize_multiseed.py --root "$MULTISEED_OUT" --seeds $SEEDS
