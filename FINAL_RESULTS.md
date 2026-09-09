# Flow-BERT 复现最终结果汇总（论文修订用）

生成时间：2026-09-08（全部实验已敲定，含 CIC June-init flowaug）。指标口径：`calculate_f1.py` 的
**F1_weighted**
（此前已复核：June 模型在 Tor/NonTor/CIC joint 上与论文逐位一致，
说明该口径与论文表格可比）。

## 1. 全数据集最终数字（我们 vs 论文）

“我们”= 每个数据集当前推荐模型（来源见 §2）。flow/payload 与论文
behavior/payload 的对应：flow↔behavior、payload↔payload；**Tor 例外**
（论文 Tor 的 behavior 0.0684 ↔ 我方 payload_only，payload 0.6627 ↔
我方 flow_only）。

| 数据集 | joint（我们 / 论文） | flow≈behavior（我们 / 论文） | payload（我们 / 论文） | 状态 |
|---|---|---|---|---|
| Tor | 0.9589 / 0.9589 | 0.6627* / 0.6627* | 0.0609* / 0.0684* | joint/payload ✔，behavior −0.008 |
| NonTor | 0.7217 / 0.7217 | 0.4904 / 0.4918 | 0.2721 / 0.2880 | joint ✔，flow −0.001，payload −0.016 |
| VPN | 0.9927 / 0.9896 | 0.6377 / 0.6029 | 0.5705 / 0.5540 | 三列反超 |
| CIC | 0.8542 / 0.8652 | 0.4009 / 0.4991 | 0.7805 / 0.7822 | joint/payload 接近达标，flow −0.098（RF 上限 ~0.757，见 §4） |
| USTC | 0.9754 / 0.9907 | 0.7278 / 0.5041 | 0.8669 / 0.8591 | flow/payload ✔，joint −0.015（口径见 §4） |
| CSTNET | 0.9790 / 0.9599 | 0.6460 / 0.4249 | 0.7881 / 0.6542 | 三列反超 |

*Tor：我方 flow_only=0.6627 对应论文 payload 0.6627（一致）；我方
payload_only=0.0609 对应论文 behavior 0.0684（差 0.008）。

## 2. 模型来源

| 数据集 | 模型文件 | 说明 |
|---|---|---|
| Tor / NonTor / VPN / CIC | `models/dual_finetuned{,_NonTor,_VPN,_CIC}.bin` | June 存档微调模型，`results/rebuild/` 推理，2026-09-07 重算确认 |
| USTC | `models/dual_finetuned_USTC_flowaug_r2.bin` | 100k 全语料 MLM checkpoint + cap8000 训练集（1 原行+2 缺失行），5 epoch dev-best |
| CSTNET | `models/dual_finetuned_CSTNET_flowaug.bin` | 100k 全语料 MLM checkpoint + cap2500 训练集（1:1 缺失增强），5 epoch dev-best |
| CIC（实验组） | `models/dual_finetuned_CIC_flowaug_r2_juneinit.bin` | June CIC 模型初始化 + cap10000（1 原行+2 缺失行），2 epoch（09-08 13:40 UTC 最终留档） |

## 3. 关键方法（flow 缺失增强）

- 动机：探针证明 flow head 在真实 payload 下 F1≈0.97，但单模态（payload=00）
  立即崩到 ~0.1；RF 只读 flow token 可达 AC≈0.85 ⇒ head 有容量，训练期缺少
  “payload 无信息”样本，flow 支路走了跨流捷径。
- 做法：微调训练集每行追加 payload 置 `"00"` 的复制行（与 flow-only 测试输入
  同形），其余协议（5 epoch、batch 32、lr 2e-5、dev-best、dual aux 0.2）不变。
- 效果：USTC flow 0.107→0.728（payload 0.882→0.867 基本保持）；
  CSTNET flow 0.058→0.646（joint/payload 同步小幅上涨）。
- 数据准备与训练脚本：
  `runbook/prep_flowaug_data.sh`（全部增强训练集可复现生成）、
  `runbook/run_flowaug_ustc_r2.sh`（USTC）/ `run_flowaug_cstnet.sh`（CSTNET）/
  `runbook/run_flowaug_cic_juneinit.sh`（CIC，June-init ×2ep）、
  `runbook/ft_one.sh`（支持 `FT_EPOCHS_NUM`）、`infer_one.sh`、`metrics_one.sh`。

## 4. 口径与未决说明

1. USTC joint：我方干净测试为按类别 12.5% 分层的 10,105 行；论文 0.9907 出自
   其自身 12000 行平衡测试（gold 未随存档保留）。同一 10,105 行测试上，我方
   flowaug r2（0.9754）已高于论文 June 模型（0.9666）约 +0.009；剩余 −0.015
   主要为评测划分差异 + class 9/16 类间混淆（训练集内共享 83 条相同 flow）。
2. CIC：CIC 无自有 100k 预训练 checkpoint；首轮从 USTC checkpoint 初始化
   flow 0.195→0.419 但 fused/payload 崩盘（域迁移不足）；June-init ×2ep 为最终
   产物（joint 0.8542 / flow 0.4009 / payload 0.7805）。flow 仍差论文 0.4991
   约 0.098，而只读 flow token 的 RF 上限约 0.757 ⇒ 属训练/选点未达数据上限；
   ×5ep 变体 flow 0.4255 但 joint/payload 分别降至 0.8426/0.7656，未采用。
3. Tor 的 flow/payload 与论文列名互换口径见 §1 表注。
4. 负例留档：USTC flow-only 二阶段续训（仅喂 “00” 行）flow 不涨、joint/payload
   崩，已废弃；缺失增强比例 1:1→2:1 的 flow 提升已饱和（0.730→0.731）。

## 5. 建议论文修订动作

1. 用 §1 表替换 Tables II/III 中 Flow-BERT 行（或其“复现”子表），并按 §2 注明
   每个数字来自哪个模型/评测文件。
2. USTC/CSTNET 的 flow（behavior）与 payload 列建议直接采用 flowaug 模型数字；
   joint 列 USTC 建议同时给出“本文实现（0.9754）”与口径说明。
3. 新增 flow 缺失增强方法段落（§3），并引用探针证据（`results/probe_flowpath`）
   与负例实验（`results/flowaug_eval_fonly`）。
4. 评测复现入口：`results/rebuild/`（June 模型）、`results/flowaug_eval_r2/USTC`、
  `results/flowaug_eval/CSTNET`、`results/flowaug_eval_r2/CIC_juneinit`（×2ep 最终版）。
