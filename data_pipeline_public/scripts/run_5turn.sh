#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TAG="5turn"
MAX_TURNS=5
MIN_TURNS=3
SHARD_SIZE=1000
BATCH_SIZE=10 # 每批并行跑几个 shard

mkdir -p ../logs

# 检查 queries 文件
if [ ! -f "../data/queries_${TAG}.jsonl" ]; then
  echo "[error] queries_${TAG}.jsonl not found. Run:"
  echo "  cp data/queries_5turn_test.jsonl data/queries_5turn.jsonl"
  exit 1
fi

# 计算总 shard 数
TOTAL_QUERIES=$(wc -l < "../data/queries_${TAG}.jsonl")
TOTAL_SHARDS=$(( (TOTAL_QUERIES + SHARD_SIZE - 1) / SHARD_SIZE ))
echo "Total queries: ${TOTAL_QUERIES}, shard size: ${SHARD_SIZE}, total shards: ${TOTAL_SHARDS}"

# 分批跑，每批 BATCH_SIZE 个并行
for (( start=0; start<TOTAL_SHARDS; start+=BATCH_SIZE )); do
  end=$(( start + BATCH_SIZE ))
  if [ $end -gt $TOTAL_SHARDS ]; then
    end=$TOTAL_SHARDS
  fi

  echo "== Batch: shard $start to $((end-1)) =="
  for (( i=start; i<end; i++ )); do
    python3 03_generate_pure_dialogues.py \
      --max-turns "$MAX_TURNS" --min-turns "$MIN_TURNS" --tag "$TAG" \
      --task-shard-size "$SHARD_SIZE" \
      --task-shard-index "$i" \
      > "../logs/${TAG}_shard_${i}.log" 2>&1 &
  done
  wait
  echo "== Batch done (shard $start to $((end-1))) =="
done

echo ""
echo "All ${TOTAL_SHARDS} shards finished."
echo "Output files: ../data/dialogues_${TAG}_shard*.jsonl"
echo ""
echo "To merge:"
echo "  cat ../data/dialogues_${TAG}_shard*.jsonl > ../data/dialogues_${TAG}.jsonl"
