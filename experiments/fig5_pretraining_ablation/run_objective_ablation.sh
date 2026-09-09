#!/usr/bin/env bash
# Fig.5-style pretraining-objective ablation runner.
#
# Usage: run_objective_ablation.sh <cl_only|mlm_only|mlm_cl> [DATASET ...]
# Env: STEPS=20000 GPU=0 FORCE=0 EPOCHS=5 MODELS_DIR RESULTS_DIR LOGS
#
# Per dataset: balanced labeled corpus -> pretraining -> default-train
# fine-tune (dev-best) -> fused/flow/payload inference + metrics.
set -euo pipefail

OBJ="${1:?usage: run_objective_ablation.sh <cl_only|mlm_only|mlm_cl> [DATASET ...]}"
shift || true
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(Tor NonTor VPN CIC USTC CSTNET)
fi

ROOT=${ROOT:-/home/wzw/Flow-BERT}
STEPS=${STEPS:-20000}
GPU=${GPU:-0}
FORCE=${FORCE:-0}
EPOCHS=${EPOCHS:-5}
TSV=${TSV:-/home/wzw/ET-BERT-main/corpora/ISCX/TSV}
TOR_TSV="$ROOT/corpora/ISCX/TSV"
MODELS_DIR=${MODELS_DIR:-$ROOT/models}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/objective_ablation}
LOGS=${LOGS:-$ROOT/experiments/logs}
mkdir -p "$LOGS"

case "$OBJ" in
  cl_only)  CL_ARG=cl_only ;;
  mlm_only) CL_ARG=none ;;
  mlm_cl)   CL_ARG=mlm_cl ;;
  *) echo "unknown objective: $OBJ" >&2; exit 2 ;;
esac

nlabels() {
  case "$1" in
    Tor) echo 35 ;; NonTor) echo 39 ;; VPN) echo 29 ;;
    CIC) echo 10 ;; USTC) echo 20 ;; CSTNET) echo 120 ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

cd "$ROOT"
for NAME in "${DATASETS[@]}"; do
  NL=$(nlabels "$NAME")
  DATA="$ROOT/dataset_${NAME}_cl_balanced.pt"
  CKPT="$MODELS_DIR/objective_${OBJ}_${NAME}.bin-${STEPS}"
  OUT="dual_finetuned_${OBJ}_${NAME}"
  RES="$RESULTS_DIR/${OBJ}/${NAME}/pred"
  [ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 1; }
  if [ "$FORCE" != "1" ] && [ -s "$RES/${NAME}_test_pred.tsv" ]; then
    echo "[skip] $OBJ/$NAME (predictions exist)"
    continue
  fi

  echo "[pretrain] objective=$OBJ dataset=$NAME steps=$STEPS gpu=$GPU $(date -u '+%F %T')"
  PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" python3 -u pre-training/pretrain.py \
    --dataset_path "$DATA" --vocab_path "$ROOT/assets/encryptd_vocab_flow_tor.txt" \
    --tokenizer space --config_path "$ROOT/assets/bert/base_config.json" \
    --encoder dual_stream_cross --dual_stream_blocks 3 --dual_self_layers_per_block 4 \
    --target bert --embedding word_pos_seg --mask fully_visible \
    --world_size 1 --gpu_ranks "$GPU" \
    --total_steps "$STEPS" --save_checkpoint_steps "$STEPS" --report_steps 200 \
    --batch_size 32 --instances_buffer_size 25600 \
    --cl_loss "$CL_ARG" --cl_alpha 0.5 --cl_beta 0.5 --cl_margin 1.0 \
    --lambda_mlm 1.0 --lambda_cl 1.0 \
    --output_model_path "$MODELS_DIR/objective_${OBJ}_${NAME}.bin" \
    > "$LOGS/${OBJ}_${NAME}_pretrain.log" 2>&1

  echo "[finetune] $NAME"
  if [ "$NAME" = "Tor" ]; then
    "$ROOT/run_finetune_gpu${GPU}.sh" --encoder dual_stream_cross \
      --flow_seq_length 32 --payload_seq_length 128 \
      --config_path "$ROOT/assets/bert/base_config.json" \
      --vocab_path "$ROOT/assets/encryptd_vocab_flow_tor.txt" --tokenizer space \
      --train_path "$TOR_TSV/Tor_finetune_train.tsv" \
      --dev_path "$TOR_TSV/Tor_finetune_valid.tsv" \
      --pretrained_model_path "$CKPT" \
      --output_model_path "$MODELS_DIR/${OUT}.bin" \
      --epochs_num "$EPOCHS" --batch_size 32 --pooling first \
      --dual_aux_loss_weight 0.2 --max_seq_length 512 \
      > "$LOGS/${OBJ}_${NAME}_ft.log" 2>&1
  else
    FT_PRETRAINED_PATH="$CKPT" FT_EPOCHS_NUM="$EPOCHS" \
      bash "$ROOT/runbook/ft_one.sh" "$NAME" "dual_pretrain_${NAME}" "$GPU" "$OUT" \
      > "$LOGS/${OBJ}_${NAME}_ft.log" 2>&1
  fi

  echo "[infer] $NAME"
  bash "$ROOT/runbook/infer_one.sh" "$NAME" "$OUT" "$NL" "$GPU" "$RES" \
    > "$LOGS/${OBJ}_${NAME}_infer.log" 2>&1
  bash "$ROOT/runbook/metrics_one.sh" "$NAME" "$RES" \
    | tee "$LOGS/${OBJ}_${NAME}_metrics.log"
  echo "DONE: $OBJ $NAME $(date -u '+%F %T')"
done
