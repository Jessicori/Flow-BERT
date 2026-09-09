#!/usr/bin/env bash
# CIC flowaug, corrected init: continue from the CIC-domain June finetuned model
# (dual_finetuned_CIC.bin) instead of the USTC 100k checkpoint, so the encoder
# already carries CIC domain knowledge. Train on the same cap10000 flowaug r2
# set for FT_EPOCHS_NUM epochs (default 2), then evaluate.
# Usage: run_flowaug_cic_juneinit.sh <GPU> [EPOCHS]
set -euo pipefail

ROOT=/home/wzw/Flow-BERT
GPU="${1:-0}"
EPOCHS="${2:-2}"

TRAIN="$ROOT/corpora/cl_corpora/CIC_finetune_cap10000_train_flowaug_r2.tsv"
PRETRAINED="$ROOT/models/dual_finetuned_CIC.bin"
OUT=dual_finetuned_CIC_flowaug_r2_juneinit
RES="$ROOT/results/flowaug_eval_r2/CIC_juneinit"
FT_LOG="$ROOT/ft_flowaug_CIC_juneinit.log"
SUP_LOG="$ROOT/flowaug_CIC_juneinit_supervisor.log"

{
echo "=== FLOWAUG CIC JUNEINIT START $(date -u) ==="
echo "train=$(basename "$TRAIN") ($(wc -l < "$TRAIN") lines)"
echo "resume=$(basename "$PRETRAINED") epochs=${EPOCHS}"

echo "=== Stage 1: finetune (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
FT_TRAIN_TSV="$TRAIN" FT_PRETRAINED_PATH="$PRETRAINED" FT_EPOCHS_NUM="$EPOCHS" \
  bash "$ROOT/runbook/ft_one.sh" CIC dual_pretrain_USTC "$GPU" "$OUT" \
  > "$FT_LOG" 2>&1
echo "[finetune done] $(date -u '+%H:%M:%S')"

echo "=== Stage 2: evaluate fused/flow/payload (gpu${GPU}) $(date -u '+%H:%M:%S') ==="
bash "$ROOT/runbook/infer_one.sh" CIC "$OUT" 10 "$GPU" "$RES"
bash "$ROOT/runbook/metrics_one.sh" CIC "$RES"

echo "FLOWAUG CIC JUNEINIT DONE $(date -u '+%Y-%m-%d %H:%M:%S')"
} | tee "$SUP_LOG"
