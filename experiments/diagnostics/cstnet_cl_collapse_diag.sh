#!/usr/bin/env bash
# CSTNET CL-only failure diagnosis:
#   GPU0: CL-only checkpoint, 20 epochs  (is fine-tuning merely slow?)
#   GPU1: strong MLM checkpoint, 2 epochs (does the same data/protocol learn?)
set -euo pipefail
ROOT=${ROOT:-/home/wzw/Flow-BERT}
LOGS=${LOGS:-$ROOT/experiments/diagnostics/logs}
TRAIN=${TRAIN:-/home/wzw/ET-BERT-main/corpora/ISCX/TSV/CSTNET_finetune_train.tsv}
VALID=${VALID:-/home/wzw/ET-BERT-main/corpora/ISCX/TSV/CSTNET_finetune_valid.tsv}
mkdir -p "$LOGS"
cd "$ROOT"

common=(--encoder dual_stream_cross \
  --flow_seq_length 32 --payload_seq_length 128 \
  --config_path "$ROOT/assets/bert/base_config.json" \
  --vocab_path "$ROOT/assets/encryptd_vocab_flow_tor.txt" --tokenizer space \
  --train_path "$TRAIN" --dev_path "$VALID" \
  --batch_size 32 --pooling first --dual_aux_loss_weight 0.2 --max_seq_length 512)

echo "[diag] CL-only 20 epochs on GPU0 $(date -u '+%F %T')"
"$ROOT/run_finetune_gpu0.sh" "${common[@]}" \
  --pretrained_model_path "$ROOT/models/abl6_CSTNET_cl_only.bin-20000" \
  --output_model_path /tmp/ctrl_cstnet_clonly_20ep.bin --epochs_num 20 \
  > "$LOGS/control_cstnet_clonly_20ep.log" 2>&1 &
P1=$!

echo "[diag] strong MLM 2 epochs on GPU1 $(date -u '+%F %T')"
"$ROOT/run_finetune_gpu1.sh" "${common[@]}" \
  --pretrained_model_path "$ROOT/models/dual_pretrain_CSTNET.bin-100000" \
  --output_model_path /tmp/ctrl_cstnet_mlm_2ep.bin --epochs_num 2 \
  > "$LOGS/control_cstnet_mlm_2ep.log" 2>&1 &
P2=$!

wait "$P1"; wait "$P2"
echo "CSTNET DIAG DONE $(date -u '+%F %T')"
