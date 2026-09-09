#!/usr/bin/env bash
# Flow-missing augmentation, heavier masked ratio (2 masked copies per real row):
#   cap8000 protocol with 1 real + 2 payload-masked ("00") copies.
# Usage: run_flowaug_ustc_r2.sh <GPU>
set -euo pipefail

ROOT=/home/wzw/Flow-BERT
GPU="${1:-0}"

TRAIN="$ROOT/corpora/cl_corpora/USTC_finetune_cap8000_train_flowaug_r2.tsv"
PRETRAINED="$ROOT/models/dual_pretrain_USTC.bin-100000"
OUT=dual_finetuned_USTC_flowaug_r2
RES="$ROOT/results/flowaug_eval_r2/USTC"
FT_LOG="$ROOT/ft_flowaug_USTC_cap8000_r2.log"
SUP_LOG="$ROOT/flowaug_USTC_cap8000_r2_supervisor.log"

{
echo "=== FLOWAUG R2 START: USTC cap8000 $(date -u) ==="
echo "train=$(basename "$TRAIN") ($(wc -l < "$TRAIN") lines)"
echo "pretrained=$(basename "$PRETRAINED")"

echo "=== Stage 1: finetune with flow-augmented cap8000 trainset (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
FT_TRAIN_TSV="$TRAIN" FT_PRETRAINED_PATH="$PRETRAINED" \
  bash "$ROOT/runbook/ft_one.sh" USTC dual_pretrain_USTC "$GPU" "$OUT" \
  > "$FT_LOG" 2>&1
echo "[finetune done] $(date -u '+%H:%M:%S')"

echo "=== Stage 2: evaluate fused/flow/payload (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
bash "$ROOT/runbook/infer_one.sh" USTC "$OUT" 20 "$GPU" "$RES"
bash "$ROOT/runbook/metrics_one.sh" USTC "$RES"

echo "FLOWAUG R2 DONE: USTC $(date -u '+%Y-%m-%d %H:%M:%S')"
} | tee "$SUP_LOG"
