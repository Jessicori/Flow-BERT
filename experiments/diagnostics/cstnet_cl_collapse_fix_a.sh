#!/usr/bin/env bash
# Fix attempt A for the CSTNET CL-only collapse: rerun CL-only pretraining with
# 60k steps (other hyperparameters unchanged), then the standard 5-epoch
# default fine-tune and three-modality evaluation.
set -euo pipefail
ROOT=${ROOT:-/home/wzw/Flow-BERT}
LOGS=${LOGS:-$ROOT/experiments/diagnostics/logs}
STEPS=${STEPS:-60000}
GPU=${GPU:-0}
mkdir -p "$LOGS"
cd "$ROOT"

echo "[A pretrain] CSTNET cl_only ${STEPS} steps $(date -u '+%F %T')"
PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" python3 -u pre-training/pretrain.py \
  --dataset_path "$ROOT/dataset_CSTNET_cl_balanced.pt" \
  --vocab_path "$ROOT/assets/encryptd_vocab_flow_tor.txt" \
  --tokenizer space --config_path "$ROOT/assets/bert/base_config.json" \
  --encoder dual_stream_cross --dual_stream_blocks 3 --dual_self_layers_per_block 4 \
  --target bert --embedding word_pos_seg --mask fully_visible \
  --world_size 1 --gpu_ranks "$GPU" \
  --total_steps "$STEPS" --save_checkpoint_steps "$STEPS" --report_steps 200 \
  --batch_size 32 --instances_buffer_size 25600 \
  --cl_loss cl_only --cl_alpha 0.5 --cl_beta 0.5 --cl_margin 1.0 \
  --lambda_mlm 1.0 --lambda_cl 1.0 \
  --output_model_path "$ROOT/models/abl6_CSTNET_cl_only_60k.bin" \
  > "$LOGS/CSTNET_pretrain_cl_only_60k.log" 2>&1

OUT=dual_finetuned_6CSTNET_cl_only_60k
RES="$ROOT/results/abl6_eval/CSTNET/cl_only_60k/pred"
echo "[A ft] CSTNET 5 epochs $(date -u '+%F %T')"
FT_PRETRAINED_PATH="$ROOT/models/abl6_CSTNET_cl_only_60k.bin-${STEPS}" \
  bash "$ROOT/runbook/ft_one.sh" CSTNET dual_pretrain_CSTNET "$GPU" "$OUT" \
  > "$LOGS/CSTNET_ft_cl_only_60k.log" 2>&1
echo "[A infer] $(date -u '+%F %T')"
bash "$ROOT/runbook/infer_one.sh" CSTNET "$OUT" 120 "$GPU" "$RES" \
  > "$LOGS/CSTNET_infer_cl_only_60k.log" 2>&1
bash "$ROOT/runbook/metrics_one.sh" CSTNET "$RES" \
  | tee "$LOGS/CSTNET_metrics_cl_only_60k.log"
echo "CSTNET FIX A DONE $(date -u '+%F %T')"
