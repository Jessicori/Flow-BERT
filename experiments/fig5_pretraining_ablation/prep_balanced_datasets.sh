#!/usr/bin/env bash
# Build dataset_<NAME>_cl_balanced.pt (class-labeled static instances) for the
# balanced pretraining corpora produced by build_balanced_corpus.py.
set -euo pipefail
ROOT=${ROOT:-/home/wzw/Flow-BERT}
cd "$ROOT"

for NAME in Tor NonTor VPN CIC USTC CSTNET; do
  OUT="$ROOT/dataset_${NAME}_cl_balanced.pt"
  if [ -s "$OUT" ]; then
    echo "[prep] exists: $OUT"
    continue
  fi
  echo "[prep] $NAME -> $OUT"
  python3 "$ROOT/Tor_preprocess_dual.py" \
    --corpus_path "$ROOT/corpora/cl_corpora/${NAME}_finetune_balanced.tsv" \
    --vocab_path "$ROOT/assets/encryptd_vocab_flow_tor.txt" \
    --dataset_path "$OUT" \
    --tokenizer space --processes_num 8 \
    --flow_tokens_num 23 --flow_seq_length 32 --payload_seq_length 128 \
    --with-class-label --seed 7
done
echo "PREP DATASETS DONE"
