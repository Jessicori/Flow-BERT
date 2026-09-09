# experiments — paper-aligned experiment code

This directory mirrors the experiment structure of the Flow-BERT paper. It
contains the code for the ablations and diagnostics that are reported in the
manuscript, while `runbook/` keeps the final (published) flow-aug pipeline.

## Organization logic

| Paper part | Directory | Content | Status |
|---|---|---|---|
| Table II/III, final method | `../runbook/` | final flow-aug fine-tune/eval pipeline and final matrix recompute | complete |
| Fig. 4 — modality ablation | `fig4_modality_ablation/` | protocol + pointers to the final flow-aug runs (payload/flow/joint) | complete |
| Fig. 5 — pretraining-objective ablation | `fig5_pretraining_ablation/` | MLM-only / CL-only / MLM+CL runners, balanced labeled corpus builder, USTC 20k/100k units | CL-only six datasets done (CSTNET collapsed); MLM-only/MLM+CL only USTC |
| Failure analysis / probes | `diagnostics/` | CSTNET CL collapse controls, flow-path probe protocol | complete |
| Result tables | `reports/` | metric tables extracted from the runs above | complete |

Everything that belongs to a *single experiment* is placed next to the script
that produced it, and the raw artifacts stay outside the repository:

| Artifact | Location | Tracked |
|---|---|---|
| pretrain checkpoints | `models/` | no (`*.bin`) |
| balanced / flow-aug corpora | `corpora/` | no |
| static datasets | `dataset_*.pt` | no |
| prediction files / metrics | `results/` | no |
| run logs | `experiments/**/logs/` | no (`*.log`) |
| code, protocols, result tables | `experiments/`, `runbook/` | yes |

## Reproducing a pretraining-objective ablation

```bash
# 1. balanced labeled corpora (6000 rows/class, seed 7) + static datasets
python3 experiments/fig5_pretraining_ablation/build_balanced_corpus.py
bash   experiments/fig5_pretraining_ablation/prep_balanced_datasets.sh

# 2. one objective on one or more datasets (20k steps, default fine-tune)
bash experiments/fig5_pretraining_ablation/run_objective_ablation.sh cl_only CSTNET
bash experiments/fig5_pretraining_ablation/run_objective_ablation.sh mlm_only USTC
bash experiments/fig5_pretraining_ablation/run_objective_ablation.sh mlm_cl  USTC
```

Protocol (identical across datasets): balanced class-labeled corpus →
20k pretraining steps (batch 32, `cl_alpha=0.5`, `cl_beta=0.5`,
`cl_margin=1.0`, `lambda_mlm=lambda_cl=1.0`) → default training-set fine-tune
(5 epochs, dev-best) → fused / flow-only / payload-only inference and
F1-weighted / AC / PR / RC metrics.

## Notes and caveats

- The scripts default to the local absolute paths used in this project
  (`/home/wzw/Flow-BERT`, `/home/wzw/ET-BERT-main/...`); override with the
  environment variables shown in each script if you move the data.
- Tor labels in `ET-BERT-main/.../Tor_finetune_train.tsv` are non-contiguous
  (35 classes, max label 41). The `Tor` branch of the ablation runner therefore
  uses the relabeled corpus in `corpora/ISCX/TSV/`, matching the June Tor model.
- The CSTNET CL-only cell is a documented failure: 120 classes with batch 32
  leads to representation collapse (see `reports/cl_only_ablation.md` and
  `diagnostics/`). It is kept as an honest negative result, not silently
  replaced.
