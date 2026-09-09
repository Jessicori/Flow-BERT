# Fig. 5 — pretraining-objective ablation

## Objectives

| Objective | `--cl_loss` | trainer loss |
|---|---|---|
| MLM-only | `none` | `loss_mlm` (`dual_mlm_only=True`) |
| CL-only | `cl_only` | `lambda_cl * loss_cl` |
| MLM+CL | `mlm_cl` | `lambda_mlm * loss_mlm + lambda_cl * loss_cl` |

CL is the distance-based objective of Eq. (9): normalized cross-stream
representations with strong-positive (same flow), same-class positive and
cross-class negative terms (`cl_alpha=0.5`, `cl_beta=0.5`, `cl_margin=1.0`).

## Protocol

- Balanced class-labeled pretraining corpus: 6000 rows/class (rare classes are
  oversampled), built from `<NAME>_finetune.tsv` with seed 7.
- 20k pretraining steps, batch 32, `dual_stream_blocks=3`,
  `dual_self_layers_per_block=4`.
- Downstream: default training set, 5 epochs, dev-best; evaluation on the clean
  joint / flow-only / payload-only test splits.

## Scripts

| Script | Purpose |
|---|---|
| `build_balanced_corpus.py` | build `corpora/cl_corpora/<NAME>_finetune_balanced.tsv` (6000/class) |
| `prep_balanced_datasets.sh` | convert those TSVs into `dataset_<NAME>_cl_balanced.pt` with class labels |
| `run_objective_ablation.sh` | run one objective (`cl_only` / `mlm_only` / `mlm_cl`) on one or more datasets |

## Status of the ablation matrix

| Dataset | MLM-only | CL-only | MLM+CL |
|---|---|---|---|
| USTC (20k) | 0.9718 joint (done) | 0.9682 joint (done) | 0.9694 joint (done) |
| USTC (100k) | 0.9745 (cap8000) / 0.9703 (default) | — | 0.9728 / 0.9721 |
| Tor | pending | 0.9595 joint | pending |
| NonTor | pending | 0.7160 joint | pending |
| VPN | pending | 0.9956 joint | pending |
| CIC | pending | 0.8321 joint | pending |
| CSTNET | pending | 0.0419 joint (20k, collapsed) / 0.0001 (60k) | pending |

The six-dataset MLM-only / MLM+CL cells are **not** available yet; the final
models in Table II/III come from heterogeneous protocols (June archive for
Tor/NonTor/VPN/CIC, MLM-only + flow-aug for USTC/CSTNET). Do not present the
current final models as MLM+CL cells.

## CSTNET CL-only failure

The 120-class CSTNET cell collapses: `loss_cl` ends at 0.375 after 20k steps and
the 5-epoch fine-tune stays at the uniform-prediction loss (dev acc 0.0083,
all 7080 test rows predicted as one class). Increasing pretraining to 60k
steps drives `loss_cl` to 0.000 but the downstream model still outputs a single
class. See `../diagnostics/` and `../reports/cl_only_ablation.md`.
