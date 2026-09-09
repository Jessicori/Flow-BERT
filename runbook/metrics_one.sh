#!/usr/bin/env bash
# Usage: metrics_one.sh <NAME> [RES_DIR]
#   Computes F1 for fused / flow / payload predictions of one dataset.
set -e
NAME="$1"
RES_DIR="${2:-/home/wzw/Flow-BERT/results/${NAME}}"
TSV="${GOLD_TSV_DIR:-/home/wzw/ET-BERT-main/corpora/ISCX/TSV}"
RES="$RES_DIR"
cd /home/wzw/Flow-BERT
if [ "$NAME" = "Tor" ]; then
  GOLD_JOINT="$TSV/Tor_test.tsv"
  GOLD_FLOW="$TSV/Tor_test_flow.tsv"
  GOLD_PAY="$TSV/Tor_test_payload.tsv"
else
  GOLD_JOINT="$TSV/${NAME}_finetune_test.tsv"
  GOLD_FLOW="$TSV/${NAME}_test_flowonly.tsv"
  GOLD_PAY="$TSV/${NAME}_test_payloadonly.tsv"
fi
echo "===== ${NAME} fused ====="
python3 calculate_f1.py --gold "$GOLD_JOINT" --pred "$RES/${NAME}_test_pred.tsv"
echo "===== ${NAME} flow_only ====="
python3 calculate_f1.py --gold "$GOLD_FLOW" --pred "$RES/${NAME}_test_pred_flow.tsv"
echo "===== ${NAME} payload_only ====="
python3 calculate_f1.py --gold "$GOLD_PAY" --pred "$RES/${NAME}_test_pred_payload.tsv"
