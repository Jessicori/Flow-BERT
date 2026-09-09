#!/usr/bin/env bash
# Flow-missing augmentation finetune for CSTNET (v3-style protocol):
#   pretrained = 100k full-corpus dual checkpoint
#   trainset   = cap2500 base rows + payload-masked ("00") copies
# Usage: run_flowaug_cstnet.sh <GPU>
set -euo pipefail

ROOT=/home/wzw/Flow-BERT
GPU="${1:-0}"

TRAIN="$ROOT/corpora/cl_corpora/CSTNET_finetune_cap2500_train_flowaug.tsv"
PRETRAINED="$ROOT/models/dual_pretrain_CSTNET.bin-100000"
OUT=dual_finetuned_CSTNET_flowaug
RES="$ROOT/results/flowaug_eval/CSTNET"
FT_LOG="$ROOT/ft_flowaug_CSTNET_cap2500.log"
SUP_LOG="$ROOT/flowaug_CSTNET_cap2500_supervisor.log"

{
echo "=== FLOWAUG START: CSTNET cap2500 $(date -u) ==="
echo "train=$(basename "$TRAIN") ($(wc -l < "$TRAIN") lines)"
echo "pretrained=$(basename "$PRETRAINED")"

echo "=== Stage 1: finetune with flow-augmented cap2500 trainset (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
FT_TRAIN_TSV="$TRAIN" FT_PRETRAINED_PATH="$PRETRAINED" \
  bash "$ROOT/runbook/ft_one.sh" CSTNET dual_pretrain_CSTNET "$GPU" "$OUT" \
  > "$FT_LOG" 2>&1
echo "[finetune done] $(date -u '+%H:%M:%S')"

echo "=== Stage 2: evaluate fused/flow/payload (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
bash "$ROOT/runbook/infer_one.sh" CSTNET "$OUT" 120 "$GPU" "$RES"
bash "$ROOT/runbook/metrics_one.sh" CSTNET "$RES"

echo "FLOWAUG DONE: CSTNET $(date -u '+%Y-%m-%d %H:%M:%S')"
} | tee "$SUP_LOG"
