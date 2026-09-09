#!/usr/bin/env bash
cd /home/wzw/Flow-BERT
export PYTHONPATH=/home/wzw/Flow-BERT${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1
exec python3 pre-training/pretrain.py "$@"
