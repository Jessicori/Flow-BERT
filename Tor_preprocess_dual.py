#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双流预训练语料预处理（配合 ``--encoder dual_stream_cross`` + ``pre-training/pretrain.py --target bert``）。

语料约定：每行前 N 个 token 为 flow，其余为 payload。与旧单流脚本的区别：

- 两路固定长度：``[CLS]+flow+PAD``（``flow_seq_length``）、``[CLS]+payload+PAD``（``payload_seq_length``）；
- **无 NSP**；每样本随机二选一：
  - **flow-MLM**：仅对 flow 序列 ``mask_seq``，payload 侧 **不掩盖**；
  - **payload-MLM**：仅对 payload ``mask_seq``，flow 侧 **不掩盖**。
- Pickle 静态：**5 元组**
  ``(src_flow, tgt_mlm_flow, src_payload, tgt_mlm_payload, mlm_side)``
  其中 ``mlm_side`` 为 ``0``=本样本为 flow-MLM（``tgt_mlm_payload`` 为空列表）、
  ``1``=payload-MLM（``tgt_mlm_flow`` 为空列表）。

用法示例：
  python Tor_preprocess_dual.py --corpus_path corpora/ISCX/TSV/Tor_pretrain.tsv \\
    --vocab_path models/encryptd_vocab_flow_tor.txt --dataset_path dataset_tor_dual.pt \\
    --tokenizer space --processes_num 8 --flow_tokens_num 23 \\
    --flow_seq_length 32 --payload_seq_length 128
"""

import argparse
import pickle
import random
import six
from packaging import version

from uer.utils import str2tokenizer
from uer.utils.constants import CLS_TOKEN, PAD_ID
from uer.utils.data import Dataset, mask_seq, merge_dataset
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed

assert version.parse(six.__version__) >= version.parse("1.12.0")


class TorDualStreamBertDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(TorDualStreamBertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob
        self.flow_tokens_num = args.flow_tokens_num
        self.flow_seq_length = args.flow_seq_length
        self.payload_seq_length = args.payload_seq_length
        self.with_class_label = getattr(args, "with_class_label", False)

    def worker(self, proc_id, start, end):
        print("Worker %d is building Tor dual-stream dataset ... " % proc_id)
        set_seed(self.seed + proc_id)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")

        samples = []
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1
                if pos >= end:
                    break
                if not line:
                    break

                s = line.strip()
                if not s:
                    continue
                if self.with_class_label:
                    parts = s.split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        lab = int(parts[0].strip())
                    except ValueError:
                        continue
                    text = parts[1].strip() + " " + parts[2].strip()
                else:
                    lab = None
                    text = s

                ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                if len(ids) <= self.flow_tokens_num:
                    continue

                flow = ids[: self.flow_tokens_num]
                payload = ids[self.flow_tokens_num :]
                samples.append((flow, payload))

        if len(samples) == 0:
            dataset_writer.close()
            print("Worker %d: no valid samples in line range [%s, %s)." % (proc_id, start, end))
            return

        cls_id = self.vocab.get(CLS_TOKEN)
        max_flow_body = self.flow_seq_length - 1
        max_payload_body = self.payload_seq_length - 1
        max_pair_tokens = max_flow_body + max_payload_body

        for _ in range(self.dup_factor):
            random.shuffle(samples)
            for idx, (flow, payload) in enumerate(samples):
                tokens_a = list(flow)
                tokens_b = list(payload)

                if self.short_seq_prob and random.random() < self.short_seq_prob:
                    low = max(len(tokens_a) + 1, min(8, max_pair_tokens))
                    if low < max_pair_tokens:
                        eff_max = random.randint(low, max_pair_tokens)
                    else:
                        eff_max = max_pair_tokens
                else:
                    eff_max = max_pair_tokens

                len_a = min(len(tokens_a), max_flow_body)
                len_b_cap = max(0, eff_max - len_a)
                tokens_a = tokens_a[:len_a]
                tokens_b = tokens_b[: min(len(tokens_b), len_b_cap, max_payload_body)]

                src_flow = [cls_id] + tokens_a
                src_payload = [cls_id] + tokens_b
                while len(src_flow) < self.flow_seq_length:
                    src_flow.append(PAD_ID)
                while len(src_payload) < self.payload_seq_length:
                    src_payload.append(PAD_ID)
                src_flow = src_flow[: self.flow_seq_length]
                src_payload = src_payload[: self.payload_seq_length]

                mlm_side = 0 if random.random() < 0.5 else 1
                if not self.dynamic_masking:
                    if mlm_side == 0:
                        src_flow, tgt_mlm_f = mask_seq(
                            src_flow,
                            self.tokenizer,
                            self.whole_word_masking,
                            self.span_masking,
                            self.span_geo_prob,
                            self.span_max_length,
                        )
                        tgt_mlm_p = []
                    else:
                        src_payload, tgt_mlm_p = mask_seq(
                            src_payload,
                            self.tokenizer,
                            self.whole_word_masking,
                            self.span_masking,
                            self.span_geo_prob,
                            self.span_max_length,
                        )
                        tgt_mlm_f = []
                    if self.with_class_label:
                        instance = (src_flow, tgt_mlm_f, src_payload, tgt_mlm_p, mlm_side, lab)
                    else:
                        instance = (src_flow, tgt_mlm_f, src_payload, tgt_mlm_p, mlm_side)
                else:
                    if self.with_class_label:
                        instance = (src_flow, src_payload, mlm_side, lab)
                    else:
                        instance = (src_flow, src_payload, mlm_side)

                pickle.dump(instance, dataset_writer, protocol=pickle.HIGHEST_PROTOCOL)

        dataset_writer.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Tor dual-stream preprocess: split flow/payload MLM, no NSP.",
    )

    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--vocab_path", default=None, type=str)
    parser.add_argument("--spm_model_path", default=None, type=str)
    parser.add_argument("--dataset_path", type=str, default="dataset_dual.pt")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space")
    parser.add_argument("--processes_num", type=int, default=1)
    parser.add_argument("--flow_tokens_num", type=int, default=23)
    parser.add_argument("--flow_seq_length", type=int, default=32, help="Includes [CLS] + flow + PAD.")
    parser.add_argument("--payload_seq_length", type=int, default=128, help="Includes [CLS] + payload + PAD.")
    parser.add_argument("--docs_buffer_size", type=int, default=100000, help="Unused; compatibility.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dup_factor", type=int, default=1)
    parser.add_argument("--dynamic_masking", action="store_true")
    parser.add_argument("--with-class-label", action="store_true",
                        help="Read a labeled corpus TSV "
                             "(label \\t flow_tokens \\t payload_tokens) and store the "
                             "class label in every instance for contrastive pretraining.")
    parser.add_argument("--whole_word_masking", action="store_true")
    parser.add_argument("--span_masking", action="store_true")
    parser.add_argument("--span_geo_prob", type=float, default=0.2)
    parser.add_argument("--span_max_length", type=int, default=10)

    args = parser.parse_args()
    if args.dynamic_masking:
        args.dup_factor = 1
    # `Dataset` base expects `seq_length` (merge helpers); unused for dual line packing.
    args.seq_length = max(args.flow_seq_length, args.payload_seq_length)

    tokenizer = str2tokenizer[args.tokenizer](args)
    dataset = TorDualStreamBertDataset(args, tokenizer.vocab, tokenizer)
    dataset.build_and_save(args.processes_num)


if __name__ == "__main__":
    main()
