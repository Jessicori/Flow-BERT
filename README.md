# Flow-BERT

双流 Transformer 网络流量分类项目：以 **flow 字节序列**与**载荷（payload）字节**为
双输入，通过双流编码 + 交叉注意力做加密流量识别；提出 **流缺失增强（flow-aug）**
训练方法，在单模态（仅有 flow、载荷缺失）评测上显著恢复 flow 支路的分类能力。

支持六类公开流量数据集（Tor / NonTor / VPN / CIC / USTC / CSTNET）的预训练、
微调、三模态（fused / flow-only / payload-only）推理与指标复算。

## 目录结构

```text
Flow-BERT/
├── assets/              # 词表与 BERT 配置（仓库内置，路径已相对化）
├── uer/                 # 模型/编码器/训练器/数据处理（含双流交叉注意力编码器）
├── pre-training/        # 预训练入口（MLM / CL / MLM+CL 目标）
├── fine-tuning/         # 微调分类入口
├── inference/           # 推理入口（fused / flow-only / payload-only）
├── runbook/             # 全套可复现实验脚本（数据准备→预训练→微调→评测）
├── experiments/         # 按论文图表结构组织的消融/诊断实验代码（见 experiments/README.md）
├── calculate_f1.py      # F1 指标计算
├── FINAL_RESULTS.md     # 最终结果矩阵与论文修订对照（推荐先读）
└── requirements.txt
```

大型产物（`models/`、`corpora/`、`results/`、`dataset_*.pt`）不入版本库，本地保留。

## 环境

- Python 3.10+、PyTorch（CUDA 可用）、scikit-learn（RF 探针用）
- 依赖见 `requirements.txt`

## 快速复现

所有实验脚本集中在 `runbook/`，统一约定：

```bash
# 微调（支持环境变量覆盖训练集/预训练权重/aux 权重/epoch 数）
bash runbook/ft_one.sh <DATASET> <CKPT_PREFIX> <GPU> [OUT_PREFIX]

# 三模态推理与指标
bash runbook/infer_one.sh  <DATASET> <MODEL> <NLABELS> <GPU> [OUT_DIR]
bash runbook/metrics_one.sh <DATASET> [RES_DIR]
```

复现 flow-aug 数据与实验：

```bash
# 1. 生成全部流缺失增强训练集（种子固定，逐字节可复现）
bash runbook/prep_flowaug_data.sh

# 2. 复现最终实验（按需任选）
bash runbook/run_flowaug_ustc_r2.sh 0        # USTC：cap8000 + 1原2缺失
bash runbook/run_flowaug_cstnet.sh 0         # CSTNET：cap2500 + 1:1
bash runbook/run_flowaug_cic_juneinit.sh 0 2 # CIC：June 模型续训 ×2ep（最终方案）

# 3. 全矩阵指标复算（从 results/ 下预测文件）
bash runbook/print_final_matrix.sh
```

各数据集标签数与测试规模、以及更细的脚本索引见
[runbook/README.md](runbook/README.md)。

按论文图表结构组织的消融与诊断实验代码（Fig. 4 模态消融、Fig. 5 预训练目标消融、
CSTNET CL-only 失败分析、CL-only 指标汇总）见
[experiments/README.md](experiments/README.md)。

## 方法要点：流缺失增强（flow-aug）

- 动机：flow 头在载荷真实时接近 fused 水平，但单模态（载荷置 `00`）评测骤降；
  只读 flow 令牌的随机森林可达较高上限，说明问题在训练期缺少“载荷无信息”样本。
- 做法：微调训练集每行追加载荷置 `"00"` 的复制行（与 flow-only 测试同形），
  其余协议不变。
- 效果与消融：见 [FINAL_RESULTS.md](FINAL_RESULTS.md)（含负例与比例饱和证据）。

## 最终结果

六数据集 × 三模态的最终数字、模型来源、与论文目标列的对照口径见
[FINAL_RESULTS.md](FINAL_RESULTS.md)。
