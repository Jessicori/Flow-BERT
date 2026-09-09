#!/usr/bin/env bash
# Recompute every FINAL_RESULTS.md number from the prediction files on disk.
# Prints fused / flow_only / payload_only F1_weighted (+AC) per dataset.
set -euo pipefail

ROOT=/home/wzw/Flow-BERT
cd "$ROOT"

run() { # run NAME RES_DIR
  echo "===== $1 ====="
  bash "$ROOT/runbook/metrics_one.sh" "$1" "$2"
}

echo "== June archive models (Tor/NonTor/VPN/CIC) =="
run Tor   "$ROOT/results/rebuild/Tor"
run NonTor "$ROOT/results/rebuild/NonTor"
run VPN   "$ROOT/results/rebuild/VPN"
run CIC   "$ROOT/results/rebuild/CIC"

echo "== Flow-aug models (USTC / CSTNET / CIC) =="
run USTC   "$ROOT/results/flowaug_eval_r2/USTC"
run CSTNET "$ROOT/results/flowaug_eval/CSTNET"
if [ -d "$ROOT/results/flowaug_eval_r2/CIC_juneinit" ]; then
  run CIC "$ROOT/results/flowaug_eval_r2/CIC_juneinit"
fi
