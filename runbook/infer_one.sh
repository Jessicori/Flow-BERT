#!/usr/bin/env bash
# Usage: infer_one.sh <NAME> <MODEL_PREFIX> <NLABELS> <GPU> [OUT_DIR]
#   Runs fused / flow-only / payload-only inference for one dataset.
set -e
NAME="$1"
PREFIX="$2"
NLABELS="$3"
GPU="${4:-0}"
OUT_DIR="${5:-/home/wzw/Flow-BERT/results/${NAME}}"
if [ "$GPU" = "1" ]; then WRAP=/home/wzw/Flow-BERT/run_infer_gpu1.sh; else WRAP=/home/wzw/Flow-BERT/run_infer_gpu0.sh; fi
TSV="${TEST_TSV_DIR:-/home/wzw/ET-BERT-main/corpora/ISCX/TSV}"
RES="$OUT_DIR"
mkdir -p "$RES"
if [ "$NAME" = "Tor" ]; then
  TEST_JOINT="$TSV/Tor_test.tsv"
  TEST_FLOW="$TSV/Tor_test_flow.tsv"
  TEST_PAY="$TSV/Tor_test_payload.tsv"
else
  TEST_JOINT="$TSV/${NAME}_finetune_test.tsv"
  TEST_FLOW="$TSV/${NAME}_test_flowonly.tsv"
  TEST_PAY="$TSV/${NAME}_test_payloadonly.tsv"
fi
MODEL="/home/wzw/Flow-BERT/models/${PREFIX}.bin"
COMMON=(--encoder dual_stream_cross --flow_seq_length 32 --payload_seq_length 128 \
  --config_path /home/wzw/Flow-BERT/assets/bert/base_config.json \
  --vocab_path /home/wzw/Flow-BERT/assets/encryptd_vocab_flow_tor.txt \
  --tokenizer space --load_model_path "$MODEL" --labels_num "$NLABELS" \
  --batch_size 64 --pooling first --max_seq_length 512)
"$WRAP" "${COMMON[@]}" --test_path "$TEST_JOINT" \
  --prediction_path "$RES/${NAME}_test_pred.tsv"
"$WRAP" "${COMMON[@]}" --test_path "$TEST_FLOW" \
  --prediction_path "$RES/${NAME}_test_pred_flow.tsv" --flow_only_logits
"$WRAP" "${COMMON[@]}" --test_path "$TEST_PAY" \
  --prediction_path "$RES/${NAME}_test_pred_payload.tsv" --payload_only_logits
