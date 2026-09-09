# Diagnostics and failure analysis

Supporting experiments for the modality-ablation and pretraining-ablation
claims. Raw logs are written to `logs/` (not tracked); predictions and models
stay outside the repository.

| Script | Purpose | Key result |
|---|---|---|
| `cstnet_cl_collapse_diag.sh` | CL-only checkpoint fine-tuned for 20 epochs vs. strong MLM checkpoint for 2 epochs on the same CSTNET data | CL-only needs far longer fine-tuning: 20 epochs reach dev acc 0.710, fused F1 0.723; 5 epochs only 0.089 / 0.042 |
| `cstnet_cl_collapse_fix_a.sh` | Fix attempt A: CL-only pretraining with 60k steps instead of 20k | fails worse: `loss_cl -> 0.000`, fine-tune stays at uniform loss, all 7080 rows predicted as one class |

## Flow-path probe protocol (modality shortcut evidence)

The motivation for flow-aug comes from probing the flow head with three payload
treatments on the same model: original flow-only test rows (payload = `00`),
payload replaced by `[PAD]`, and joint rows (real payload). A flow head that
scores ~0.97 on joint rows but ~0.1–0.27 on flow-only rows is exploiting the
cross-stream shortcut; the probe results are summarized in `FINAL_RESULTS.md`
§3 and `../reports/cl_only_ablation.md`.

## CSTNET CL-only failure summary

1. `loss_cl` at 20k steps: Tor 0.012, VPN 0.046, NonTor 0.060, CIC 0.092,
   CSTNET 0.375. CSTNET is the only dataset where contrastive training does not
   converge to a low loss.
2. 120 classes with batch 32: only ≈23% of anchors have another same-class
   sample in the batch, so most anchors receive only the trivial same-flow
   positive and the class-level signal is weak.
3. 720k balanced rows vs. 20k × 32 = 640k samples seen (≈0.9 epoch), the least
   coverage of all datasets.
4. Fine-tuning cannot recover within the common 5-epoch protocol; with 20
   epochs it reaches a usable joint model (dev 0.710, F1 0.723), but the
   single-modality heads remain far below the MLM-initialized control.
5. Increasing pretraining to 60k steps leads to complete representation
   collapse (`loss_cl = 0.000`, one predicted class for all test rows), so
   step count is not the fix; a corrected contrastive objective (e.g.
   temperature-scaled SupCon with class-aware batches) would be required.
