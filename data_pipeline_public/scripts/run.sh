#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ../logs

for i in $(seq 0 14); do
  echo "== start shard $i =="
  python3 03_generate_pure_dialogues.py \
    --skip-query-gen \
    --task-shard-size 1000 \
    --task-shard-index "$i" \
    > "../logs/shard_${i}.log" 2>&1 &
done

wait
echo "All 15 shards finished."