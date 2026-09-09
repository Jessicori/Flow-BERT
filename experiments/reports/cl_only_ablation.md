# CL-only pretraining ablation — six datasets (final tables)

Protocol: balanced class-labeled corpus, 20k pretraining steps, default
training-set fine-tune (5 epochs, dev-best), clean test split. USTC reuses the
existing 20k unit from the original USTC ablation; the other five datasets are
the new runs. Metrics are F1-weighted unless noted.

## F1-weighted

| Dataset | fused joint | flow-only | payload-only | Note |
|---|---|---|---|---|
| Tor | 0.9595 | 0.6127 | 0.0325 | local relabeled corpus |
| NonTor | 0.7160 | 0.3439 | 0.2009 | — |
| VPN | 0.9956 | 0.5664 | 0.2835 | — |
| CIC | 0.8321 | 0.1595 | 0.7395 | — |
| USTC | 0.9682 | 0.0835 | 0.6797 | existing 20k unit |
| CSTNET | 0.0419 | 0.0154 | 0.0006 | collapsed; see below |

## AC / weighted PR / weighted RC (CL-only)

| Dataset | Modality | AC | PR_w | RC_w |
|---|---|---|---|---|
| Tor | fused / flow / payload | 0.9596 / 0.7050 / 0.0860 | 0.9597 / 0.5705 / 0.0766 | 0.9596 / 0.7050 / 0.0860 |
| NonTor | fused / flow / payload | 0.7264 / 0.4231 / 0.2578 | 0.7256 / 0.4093 / 0.3473 | 0.7264 / 0.4231 / 0.2578 |
| VPN | fused / flow / payload | 0.9956 / 0.6210 / 0.3736 | 0.9959 / 0.6083 / 0.3154 | 0.9956 / 0.6210 / 0.3736 |
| CIC | fused / flow / payload | 0.8324 / 0.2016 / 0.7395 | 0.8343 / 0.4237 / 0.7536 | 0.8324 / 0.2016 / 0.7395 |
| USTC | fused / flow / payload | 0.9682 / 0.1343 / 0.7038 | 0.9686 / 0.1217 / 0.7420 | 0.9682 / 0.1343 / 0.7038 |
| CSTNET | fused / flow / payload | 0.0083 (all) | 0.0001 (all) | 0.0083 (all) |

## CSTNET failure and controls

| CSTNET run | dev acc | fused F1_w | flow F1_w | payload F1_w |
|---|---|---|---|---|
| CL-only 20k + 5-epoch fine-tune | 0.0886 | 0.0419 | 0.0154 | 0.0006 |
| CL-only 20k + 20-epoch fine-tune | 0.7103 | 0.7234 | 0.0332 | 0.2388 |
| strong MLM checkpoint + 2-epoch fine-tune | 0.7315 | 0.7246 | 0.1355 | 0.4445 |
| CL-only 60k + 5-epoch fine-tune | 0.0083 | 0.0001 | 0.0001 | 0.0001 |

The 60k variant predicts a single class for all 7080 test rows in fused and
flow mode; `loss_cl` reaches 0.000 while the downstream loss stays at the
uniform-prediction value `ln(120) * 1.4`, i.e. representation collapse.

## Comparison with the current final models (F1-weighted, fused / flow / payload)

| Dataset | Current final model | CL-only |
|---|---|---|
| Tor | 0.9589 / 0.6627 / 0.0609 | 0.9595 / 0.6127 / 0.0325 |
| NonTor | 0.7217 / 0.4904 / 0.2721 | 0.7160 / 0.3439 / 0.2009 |
| VPN | 0.9927 / 0.6377 / 0.5705 | 0.9956 / 0.5664 / 0.2835 |
| CIC | 0.8542 / 0.4009 / 0.7805 | 0.8321 / 0.1595 / 0.7395 |
| USTC | 0.9754 / 0.7278 / 0.8669 | 0.9682 / 0.0835 / 0.6797 |
| CSTNET | 0.9790 / 0.6460 / 0.7881 | 0.0419 / 0.0154 / 0.0006 |

Note: the current final models are not a controlled MLM+CL baseline
(Tor/NonTor/VPN/CIC are June archive models; USTC/CSTNET use MLM-only
pretraining plus flow-aug), so this table must not be presented as a clean
MLM+CL vs. CL-only comparison. CL-only fused F1 exceeds the current model on
Tor and VPN, which contradicts a blanket "MLM+CL is always best" claim.
