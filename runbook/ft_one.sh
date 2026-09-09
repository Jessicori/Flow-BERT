#!/usr/bin/env bash
# Usage: ft_one.sh <NAME> <MODEL_PREFIX> <GPU> [OUT_PREFIX]
#   NAME: dataset name, e.g. CSTNET / CIC / NonTor / USTC / VPN / Tor
#   MODEL_PREFIX: checkpoint basename without .bin, e.g. dual_pretrain_CSTNET (Tor: dual_pretrain)
#   GPU: 0 or 1
#   OUT_PREFIX: optional output basename without .bin (default dual_finetuned[_NAME])
set -e
NAME="$1"
PREFIX="$2"
GPU="${3:-0}"
OUT_PREFIX="${4:-}"
if [ "$GPU" = "1" ]; then WRAP=/home/wzw/Flow-BERT/run_finetune_gpu1.sh; else WRAP=/home/wzw/Flow-BERT/run_finetune_gpu0.sh; fi
if [ -z "$OUT_PREFIX" ]; then
  if [ "$NAME" = "Tor" ]; then OUT_PREFIX=dual_finetuned; else OUT_PREFIX=dual_finetuned_${NAME}; fi
fi
TSV=/home/wzw/ET-BERT-main/corpora/ISCX/TSV
PRETRAINED_PATH="${FT_PRETRAINED_PATH:-/home/wzw/Flow-BERT/models/${PREFIX}.bin-100000}"
TRAIN_TSV="${FT_TRAIN_TSV:-$TSV/${NAME}_finetune_train.tsv}"
"$WRAP" --encoder dual_stream_cross \
  --flow_seq_length 32 --payload_seq_length 128 \
  --config_path /home/wzw/Flow-BERT/assets/bert/base_config.json \
  --vocab_path /home/wzw/Flow-BERT/assets/encryptd_vocab_flow_tor.txt \
  --tokenizer space \
  --train_path "$TRAIN_TSV" \
  --dev_path "$TSV/${NAME}_finetune_valid.tsv" \
  --pretrained_model_path "$PRETRAINED_PATH" \
  --output_model_path "/home/wzw/Flow-BERT/models/${OUT_PREFIX}.bin" \
  --epochs_num "${FT_EPOCHS_NUM:-5}" --batch_size 32 --pooling first \
  --dual_aux_loss_weight "${FT_AUX_WEIGHT:-0.2}" --max_seq_length 512
