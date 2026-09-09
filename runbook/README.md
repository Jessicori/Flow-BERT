# runbook 索引（最终方案）

统一约定：`ft_one.sh` 通过环境变量覆盖默认协议：
`FT_TRAIN_TSV` / `FT_PRETRAINED_PATH` / `FT_AUX_WEIGHT`（默认 0.2）/
`FT_EPOCHS_NUM`（默认 5）；推理 `infer_one.sh <NAME> <MODEL> <NLABELS>
<GPU> [OUT_DIR]`；指标 `metrics_one.sh <NAME> [RES_DIR]`。

## 数据准备

| 脚本 | 作用 |
|---|---|
| `prep_flowaug_data.sh` | 复现全部 flowaug 训练集（USTC/CSTNET/CIC，种子固定，逐字节可复现） |

## 最终实验（微调 + 评测）

| 脚本 | 作用 |
|---|---|
| `ft_one.sh` | 单数据集微调（默认 USTC/CSTNET 100k ckpt） |
| `run_flowaug_ustc_r2.sh` | USTC cap8000 flowaug r2（1 原行 + 2 缺失行） |
| `run_flowaug_cstnet.sh` | CSTNET cap2500 flowaug 1:1 |
| `run_flowaug_cic_juneinit.sh` | CIC cap10000 flowaug r2（June CIC 初始化，×2ep） |

## 推理 / 汇总

| 脚本 | 作用 |
|---|---|
| `infer_one.sh` / `metrics_one.sh` | fused/flow/payload 推理与 F1 |
| `print_final_matrix.sh` | 从磁盘预测文件重算最终矩阵（FINAL_RESULTS.md 数据源） |

## 数据集/标签数速查

| 数据集 | NLABELS | joint 测试 | flow/payload 测试 |
|---|---|---|---|
| Tor | 35 | 34,066 | 34,066 |
| NonTor | 39 | 32,638 | 32,638 |
| VPN | 29 | 17,875 | 2,987 |
| CIC | 10 | 69,620 | 69,620 |
| USTC | 20 | 10,105 | 12,000 |
| CSTNET | 120 | 7,080 | 7,080 |
