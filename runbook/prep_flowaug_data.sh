#!/usr/bin/env bash
# Regenerate every flow-augmentation train TSV used by the flowaug experiments.
#   flowaug (1:1)  : each real row + one payload-masked ("00") copy
#   flowaug r2     : each real row + two payload-masked ("00") copies
#   flowonly       : payload-masked rows only (stage-2 continuation set)
# Masked rows are byte-identical in form to the flow-only test files (text_a="00").
#
# Usage: prep_flowaug_data.sh
# Requires: awk, python3 (for the seeded CIC cap10000 sample).
set -euo pipefail

ROOT=/home/wzw/Flow-BERT
CL="$ROOT/corpora/cl_corpora"
TSV=/home/wzw/ET-BERT-main/corpora/ISCX/TSV

fa11() { # fa11 <base.tsv> <out.tsv> : 1 real + 1 masked copy
  awk 'BEGIN{FS=OFS="\t"} NR==1{print; next} {print; $3="00"; print}' "$1" > "$2"
}
far2() { # far2 <base.tsv> <out.tsv> : 1 real + 2 masked copies
  awk 'BEGIN{FS=OFS="\t"} NR==1{print; next} {print; $3="00"; print; print}' "$1" > "$2"
}
faonly() { # faonly <base.tsv> <out.tsv> : masked rows only
  awk 'BEGIN{FS=OFS="\t"} NR==1{print; next} {$3="00"; print}' "$1" > "$2"
}

echo "=== flowaug USTC default (1:1) ==="
fa11 "$TSV/USTC_finetune_train.tsv" "$CL/USTC_finetune_default_train_flowaug.tsv"

echo "=== flowaug USTC cap8000 (1:1 / r2 / flowonly) ==="
fa11 "$CL/USTC_finetune_cap8000_train.tsv" "$CL/USTC_finetune_cap8000_train_flowaug.tsv"
far2 "$CL/USTC_finetune_cap8000_train.tsv" "$CL/USTC_finetune_cap8000_train_flowaug_r2.tsv"
faonly "$CL/USTC_finetune_cap8000_train.tsv" "$CL/USTC_finetune_cap8000_train_flowonly.tsv"

echo "=== flowaug CSTNET cap2500 (1:1) ==="
fa11 "$CL/CSTNET_finetune_cap2500_train.tsv" "$CL/CSTNET_finetune_cap2500_train_flowaug.tsv"

echo "=== CIC cap10000 sample + flowaug r2 (seeded) ==="
python3 - <<'PYEOF'
import random
random.seed(20260907)
src = "/home/wzw/ET-BERT-main/corpora/ISCX/TSV/CIC_finetune_train.tsv"
cap_path = "/home/wzw/Flow-BERT/corpora/cl_corpora/CIC_finetune_cap10000_train.tsv"
fa_path = "/home/wzw/Flow-BERT/corpora/cl_corpora/CIC_finetune_cap10000_train_flowaug_r2.tsv"
by = {}
with open(src) as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            by.setdefault(parts[0], []).append(line.rstrip("\n"))
sample = []
for lab in sorted(by):
    sample.extend(random.sample(by[lab], min(10000, len(by[lab]))))
random.shuffle(sample)
with open(cap_path, "w") as f:
    f.write(header)
    f.write("\n".join(sample) + "\n")
with open(fa_path, "w") as f:
    f.write(header)
    for line in sample:
        f.write(line + "\n")
        m = line.split("\t")
        m[2] = "00"
        f.write("\t".join(m) + "\n")
        f.write("\t".join(m) + "\n")
PYEOF

echo "=== line counts (data rows incl header) ==="
wc -l \
  "$CL/USTC_finetune_default_train_flowaug.tsv" \
  "$CL/USTC_finetune_cap8000_train_flowaug.tsv" \
  "$CL/USTC_finetune_cap8000_train_flowaug_r2.tsv" \
  "$CL/USTC_finetune_cap8000_train_flowonly.tsv" \
  "$CL/CSTNET_finetune_cap2500_train_flowaug.tsv" \
  "$CL/CIC_finetune_cap10000_train.tsv" \
  "$CL/CIC_finetune_cap10000_train_flowaug_r2.tsv"
