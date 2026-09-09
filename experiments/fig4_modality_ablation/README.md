# Fig. 4 — modality ablation (joint / payload-only / flow-only)

The modality ablation is produced by the final flow-aug pipeline, which lives
in `runbook/`:

| Step | Script |
|---|---|
| build flow-augmented training sets | `runbook/prep_flowaug_data.sh` |
| USTC final (cap8000, 1 real + 2 masked copies) | `runbook/run_flowaug_ustc_r2.sh` |
| CSTNET final (cap2500, 1 real + 1 masked copy) | `runbook/run_flowaug_cstnet.sh` |
| CIC final (June-init, cap10000, ×2 epochs) | `runbook/run_flowaug_cic_juneinit.sh` |
| generic fine-tune / inference / metrics | `runbook/ft_one.sh`, `infer_one.sh`, `metrics_one.sh` |
| recompute the six-dataset matrix from predictions | `runbook/print_final_matrix.sh` |

Evaluation protocol: for each dataset the fine-tuned model is evaluated three
times — fused joint input, flow-only input (payload replaced by the constant
placeholder used at test time) and payload-only input. Metrics are
F1-weighted, AC, weighted PR/RC (and macro counterparts). The resulting numbers
are summarized in `../reports/cl_only_ablation.md` (CL-only runs) and in the
repository-root `FINAL_RESULTS.md` (final models vs. paper).

The flow-aug method itself is described in `FINAL_RESULTS.md` §3; the negative
controls (flow-only stage-2 continuation, 1:1 → 2:1 saturation) are recorded in
`diagnostics/`.
